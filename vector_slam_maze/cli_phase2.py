"""Phase 2 CLI subcommands: camera characterization, ROS 2 bridge, visual SLAM, relocalization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .cliff_guard import CliffHazard
from .atlas_lock import atlas_file_lock
from .camera_characterize import (
    assess_illumination,
    estimate_rolling_shutter_metrics,
    evaluate_visual_vs_odom_drift,
    measure_sharpness_tenengrad,
    measure_stream_timing,
)
from .movement_directives import (
    _sdk_units,
    bounded_drive,
    bounded_turn,
    execute_bounded_directive,
)
from .navmap_atlas import NavMapAtlas
from .sdk_session import (
    VectorSDKSession,
    _cancel_camera_task,
    _read_camera_image_nonblocking,
    _sdk_exception_types,
    capture_high_res_single_image,
    check_safety,
    enable_robot_image_streaming,
    poll_latest_camera_image,
    read_camera_exposure_state,
)
from .sensor_bridge import (
    CameraFrameSample,
    CameraIntrinsics,
    InertialOdomSample,
    VectorRos2Bridge,
    check_ros2_environment,
    export_slam_dataset,
    load_replay_dataset,
    pil_to_gray_numpy,
    sample_inertial_odom,
    synthesize_laserscan_from_tof_and_atlas,
)
from .types_and_math import Pose2D, normalize_angle, pose2d_from_sdk_pose
from .visual_slam import VisualPoseGraphSLAM

MAX_CAPTURE_STEPS = 400
MAX_SCAN_STEPS = 36
MAX_BG_POLLS = 2400


class _DarkRoomAbort(RuntimeError):
    """Raised when woken on-charger illumination fails the visual SLAM gate."""

    def __init__(self, illumination: Dict[str, Any]) -> None:
        super().__init__("dark_room_refuse_undock")
        self.illumination = illumination


def _default_vmap_path(args: argparse.Namespace) -> Path:
    """Resolve visual map JSON path from --visual-map or sibling of --atlas."""
    explicit = getattr(args, "visual_map", None)
    if explicit:
        return Path(explicit)
    return Path(args.atlas).with_suffix(".vslam.json")


def _probe_high_res(robot: Any, active_control: bool = False) -> Dict[str, Any]:
    """Probe single 1280x720 high-res capture when active behavior control is held."""
    if not active_control:
        return {
            "captured": False,
            "reason": "requires_behavior_control",
            "note": "wirepod_vector_sdk 0.8.1 capture_single_image requires behavior_control_level; 640x360 init_camera_feed is passive",
        }
    cam = getattr(robot, "camera", None)
    if cam is None:
        return {"captured": False, "reason": "no_camera_component"}
    _cancel_camera_task(robot)
    enable_robot_image_streaming(robot, False)
    time.sleep(0.1)
    t0 = time.monotonic()
    img = capture_high_res_single_image(robot, enable_high_res=True)
    rtt_ms = (time.monotonic() - t0) * 1000.0
    if img is None or getattr(img, "raw_image", None) is None:
        return {"captured": False, "reason": "capture_failed"}
    gray = pil_to_gray_numpy(img.raw_image)
    return {"captured": True, "width": int(gray.shape[1]), "height": int(gray.shape[0]), "rtt_ms": round(rtt_ms, 2), "illumination": assess_illumination(gray)}


def _empty_frame_illumination(tel: Dict[str, Any], exp_state: Dict[str, Any]) -> Dict[str, Any]:
    """Return honest non-viable illumination status when zero camera frames arrived."""
    calm = bool(tel.get("is_in_calm_power_mode"))
    status = "camera_paused_calm_power_mode" if calm else "no_frames_received"
    return {
        "status": status,
        "visual_slam_viable": False,
        "reason": "robot_in_calm_power_mode_on_charger" if calm else "no_frames_received",
        "frames_received": 0,
        "exposure_ms": exp_state.get("exposure_ms"),
        "gain": exp_state.get("gain"),
        "max_exposure_ms": exp_state.get("max_exposure_ms"),
        "max_gain": exp_state.get("max_gain"),
    }


def _collect_camera_stream(
    robot: Any, duration_s: float
) -> Tuple[CameraIntrinsics, List[Tuple[float, np.ndarray]], List[int], List[InertialOdomSample], Dict[str, Any]]:
    """Collect streamed grayscale frames and inertial odom over bounded wall duration."""
    deadline = time.monotonic() + max(0.2, min(30.0, duration_s))
    frames: List[Tuple[float, np.ndarray]] = []
    frame_ids: List[int] = []
    odoms: List[InertialOdomSample] = []
    last_id = -1
    intr: Optional[CameraIntrinsics] = None
    cfg = getattr(getattr(robot, "camera", None), "config", None)
    for _ in range(MAX_CAPTURE_STEPS):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        cimg = poll_latest_camera_image(robot, max_wait_s=min(0.35, remaining), step_s=0.02, min_image_id=last_id)
        if cimg is None:
            continue
        fid = int(getattr(cimg, "image_id", last_id + 1))
        if fid == last_id:
            continue
        last_id = fid
        raw_img = getattr(cimg, "raw_image", cimg)
        gray = pil_to_gray_numpy(raw_img)
        if intr is None:
            intr = CameraIntrinsics.from_sdk_config(cfg, target_width=int(gray.shape[1]), target_height=int(gray.shape[0]))
        ts = float(getattr(cimg, "image_recv_time", None) or time.time())
        frames.append((ts, gray))
        frame_ids.append(fid)
        odoms.append(sample_inertial_odom(robot, timestamp_s=ts))
    final_intr = intr or CameraIntrinsics.from_sdk_config(cfg, 640, 360)
    return (final_intr, frames, frame_ids, odoms, read_camera_exposure_state(robot))


def _wake_camera_docked(robot: Any, max_wait_s: float = 2.0) -> bool:
    """Wake Vector camera out of calm_power_mode on charger with zero wheel motion."""
    assert robot is not None and 0.0 < max_wait_s <= 10.0
    cam = getattr(robot, "camera", None)
    assert cam is not None or hasattr(robot, "status")
    if cam is None:
        return False
    if _read_camera_image_nonblocking(cam) is None and getattr(robot, "behavior", None) is not None:
        degrees_fn, _d, _s = _sdk_units()
        catch_types = (OSError, RuntimeError, TimeoutError, AttributeError) + _sdk_exception_types()
        try:
            robot.behavior.set_head_angle(degrees_fn(0.0))
        except catch_types:
            pass
    task = getattr(cam, "_camera_feed_task", None)
    need_revive = (_read_camera_image_nonblocking(cam) is None) or task is None or (hasattr(task, "done") and task.done())
    if need_revive:
        _reopen_camera_stream(robot)
    return poll_latest_camera_image(robot, max_wait_s=max_wait_s, step_s=0.05) is not None


def _reopen_camera_stream(robot: Any) -> None:
    """Cancel any active CameraFeed before calling EnableImageStreaming(True) and reopening."""
    assert robot is not None
    cam = getattr(robot, "camera", None)
    if cam is None:
        return
    _cancel_camera_task(robot)
    enable_robot_image_streaming(robot, True)
    if hasattr(cam, "init_camera_feed"):
        loop = getattr(getattr(robot, "conn", None), "loop", None)
        fn = cam.init_camera_feed
        loop.call_soon_threadsafe(fn) if (loop and hasattr(loop, "call_soon_threadsafe")) else fn()


def _snapshot_pose_and_charger(robot: Any) -> Dict[str, Any]:
    """Capture current robot SE(2) pose and world.charger SE(2) pose/visibility."""
    assert robot is not None
    r_pose = getattr(robot, "pose", None)
    r_dict = pose2d_from_sdk_pose(r_pose).to_dict() if r_pose is not None else None
    ch = getattr(getattr(robot, "world", None), "charger", None)
    ch_pose = getattr(ch, "pose", None) if ch is not None else None
    assert r_dict is None or isinstance(r_dict, dict)
    ch_dict: Optional[Dict[str, Any]] = None
    if ch is not None:
        ch_dict = {
            "object_id": getattr(ch, "object_id", None),
            "is_visible": bool(getattr(ch, "is_visible", False)),
            "pose": pose2d_from_sdk_pose(ch_pose).to_dict() if ch_pose is not None else None,
        }
    return {"robot_pose": r_dict, "charger": ch_dict}


class _BackgroundStreamCollector:
    """Continuously sample camera frames and inertial odometry while motors move."""

    def __init__(self, robot: Any, step_s: float = 0.03) -> None:
        assert robot is not None and 0.005 <= step_s <= 0.25
        assert MAX_BG_POLLS > 0
        self.robot = robot
        self.step_s = float(step_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.frames: List[Tuple[float, np.ndarray]] = []
        self.frame_ids: List[int] = []
        self.odoms: List[InertialOdomSample] = []

    def start(self) -> None:
        """Launch bounded background sampling thread."""
        assert self._thread is None, "collector already started"
        assert not self._stop.is_set()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        """Poll _latest_image by image_id until stop event or MAX_BG_POLLS."""
        assert self.robot is not None and MAX_BG_POLLS > 0
        cam = getattr(self.robot, "camera", None)
        if cam is None:
            return
        cfg = getattr(cam, "config", None)
        last_id, last_rx, last_revive = -1, time.monotonic(), 0.0
        for _ in range(MAX_BG_POLLS):
            if self._stop.is_set():
                break
            now = time.monotonic()
            new_id = self._append_if_new(_read_camera_image_nonblocking(cam), cfg, last_id)
            if new_id > last_id:
                last_id, last_rx = new_id, now
            elif (now - last_rx) >= 2.0 and (now - last_revive) >= 2.0:
                if not bool(getattr(getattr(self.robot, "status", None), "is_in_calm_power_mode", False)):
                    _reopen_camera_stream(self.robot)
                    last_revive = now
            self._stop.wait(self.step_s)

    def _append_if_new(self, cimg: Any, cfg: Any, last_id: int) -> int:
        """Decode and append sample when image_id strictly advances."""
        assert isinstance(last_id, int) and MAX_BG_POLLS > 0
        if cimg is None:
            return last_id
        fid = int(getattr(cimg, "image_id", last_id + 1))
        if fid <= last_id:
            return last_id
        raw_img = getattr(cimg, "raw_image", cimg)
        if raw_img is None:
            return last_id
        gray = pil_to_gray_numpy(raw_img)
        ts = float(getattr(cimg, "image_recv_time", None) or time.time())
        odom = sample_inertial_odom(self.robot, timestamp_s=ts)
        with self._lock:
            if self.intrinsics is None:
                self.intrinsics = CameraIntrinsics.from_sdk_config(cfg, int(gray.shape[1]), int(gray.shape[0]))
            self.frames.append((ts, gray))
            self.frame_ids.append(fid)
            self.odoms.append(odom)
        return fid

    def stop(self) -> Tuple[Optional[CameraIntrinsics], List[Tuple[float, np.ndarray]], List[int], List[InertialOdomSample]]:
        """Signal sampler stop, join thread, and return collected samples."""
        assert self._stop is not None and self._lock is not None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            return (self.intrinsics, list(self.frames), list(self.frame_ids), list(self.odoms))


def cmd_characterize_camera(args: argparse.Namespace) -> int:
    """Measure camera intrinsics, FPS, jitter, illumination, exposure, and dataset."""
    assert args is not None and hasattr(args, "serial")
    use_active = bool(getattr(args, "via_vector", False))
    hr_info: Optional[Dict[str, Any]] = None
    pose_charger: Optional[Dict[str, Any]] = None
    try:
        with VectorSDKSession(serial=args.serial, active=use_active, via_vector=use_active, enable_nav_map=False, enable_camera=True) as robot:
            if use_active:
                _wake_camera_docked(robot, max_wait_s=2.0)
            _ok, safety_reason, tel = check_safety(robot, allow_on_charger=True)
            intr, frames, fids, odoms, exp_state = _collect_camera_stream(robot, float(args.duration_s))
            pose_charger = _snapshot_pose_and_charger(robot)
            if args.high_res_probe:
                hr_info = _probe_high_res(robot, active_control=use_active)
    except CliffHazard as exc:
        print(json.dumps({"ok": False, "error": exc.hazard, "refused": exc.refused, "detail": str(exc)}, indent=2))
        return 6
    timing = measure_stream_timing([ts for ts, _ in frames], frame_ids=fids)
    illum = assess_illumination(frames[-1][1], exposure_ms=exp_state.get("exposure_ms"), max_exposure_ms=exp_state.get("max_exposure_ms"), gain=exp_state.get("gain"), max_gain=exp_state.get("max_gain")) if frames else _empty_frame_illumination(tel, exp_state)
    rs = estimate_rolling_shutter_metrics([g for _, g in frames], [o.gyro_xyz_rad_s[2] for o in odoms], [ts for ts, _ in frames], intr.fx)
    manifest = export_slam_dataset(Path(args.export_dataset), frames, odoms, intr, fps=timing["mean_fps"] or 15.0) if args.export_dataset and frames else None
    out = {
        "ok": bool(frames), "status": "sampled" if frames else illum["status"], "safety": safety_reason,
        "via_vector": use_active, "telemetry": tel, "pose_and_charger": pose_charger,
        "intrinsics": intr.to_dict(), "exposure": exp_state, "timing": timing,
        "illumination": illum, "rolling_shutter": rs, "high_res_probe": hr_info, "exported_dataset": manifest,
    }
    print(json.dumps(out, indent=2))
    return 0


def _preflight_passive_illumination(serial: str) -> Dict[str, Any]:
    """Check docked/passive camera illumination before authorizing active undock."""
    assert serial, "serial required"
    with VectorSDKSession(serial=serial, active=False, enable_nav_map=False, enable_camera=True) as robot:
        _ok, _reason, tel = check_safety(robot, allow_on_charger=True)
        _intr, frames, _fids, _odoms, exp_state = _collect_camera_stream(robot, duration_s=1.2)
    assert isinstance(tel, dict)
    if not frames:
        return _empty_frame_illumination(tel, exp_state)
    res = assess_illumination(
        frames[-1][1],
        exposure_ms=exp_state.get("exposure_ms"),
        max_exposure_ms=exp_state.get("max_exposure_ms"),
        gain=exp_state.get("gain"),
        max_gain=exp_state.get("max_gain"),
    )
    res["frames_received"] = len(frames)
    return res


def _run_slam_sequence(
    intrinsics: CameraIntrinsics,
    frames: List[Tuple[float, np.ndarray]],
    odoms: List[InertialOdomSample],
) -> Tuple[VisualPoseGraphSLAM, List[Pose2D], List[Pose2D], Dict[str, Any]]:
    """Run VisualPoseGraphSLAM over paired frames and odom samples."""
    slam = VisualPoseGraphSLAM(intrinsics)
    vis_traj: List[Pose2D] = []
    odom_traj: List[Pose2D] = []
    accepted = 0
    for idx, (ts, gray) in enumerate(frames):
        odom = odoms[min(idx, len(odoms) - 1)]
        res = slam.process_frame(gray, odom, ts)
        if res["accepted_visual"]:
            accepted += 1
        vis_traj.append(Pose2D.from_dict(res["pose"]))
        odom_traj.append(odom.pose)
    drift = evaluate_visual_vs_odom_drift(vis_traj, odom_traj, [o.pose.theta_rad for o in odoms[: len(vis_traj)]])
    rs = estimate_rolling_shutter_metrics([g for _, g in frames], [o.gyro_xyz_rad_s[2] for o in odoms], [ts for ts, _ in frames], intrinsics.fx)
    summary = {"accepted_visual_frames": accepted, "total_frames": len(frames), "drift": drift, "rolling_shutter": rs}
    return (slam, vis_traj, odom_traj, summary)


def _persist_visual_outputs(
    args: argparse.Namespace, slam: VisualPoseGraphSLAM, tag: str, origin_id: int
) -> Tuple[Path, str, Dict[str, Any]]:
    """Save visual pose-graph map, sync into NavMapAtlas, and export ROS map."""
    atlas_path, export_dir = Path(args.atlas), Path(args.export_dir)
    vmap_path = _default_vmap_path(args)
    slam.save(vmap_path)
    with atlas_file_lock(atlas_path):
        atlas = NavMapAtlas.load(atlas_path)
        sid, tof_updates = slam.sync_into_navmap_atlas(atlas, tag, origin_id=origin_id)
        atlas.save(atlas_path)
        stats = atlas.export_ros_pgm_yaml(export_dir / "grid.pgm", export_dir / "grid.yaml")
    stats["tof_updates_from_vslam"] = tof_updates
    return (vmap_path, sid, stats)


def _verify_docked_illum_or_abort(robot: Any, tel: Dict[str, Any]) -> None:
    """Wake camera on charger with 0 wheel motion and abort before undock if dark."""
    assert robot is not None and isinstance(tel, dict)
    _wake_camera_docked(robot, max_wait_s=2.5)
    _intr, frames, _ids, _ods, exp_state = _collect_camera_stream(robot, duration_s=0.8)
    assert isinstance(frames, list)
    if not frames:
        raise _DarkRoomAbort(_empty_frame_illumination(tel, exp_state))
    illum = assess_illumination(frames[-1][1], exposure_ms=exp_state.get("exposure_ms"), max_exposure_ms=exp_state.get("max_exposure_ms"), gain=exp_state.get("gain"), max_gain=exp_state.get("max_gain"))
    illum["frames_received"] = len(frames)
    if not illum["visual_slam_viable"]:
        raise _DarkRoomAbort(illum)


def _drive_baseline_and_sweep(
    robot: Any, args: argparse.Namespace, charger_obs: Dict[str, Any]
) -> None:
    """Execute optional bounded forward baseline and bounded multi-step yaw sweep."""
    assert robot is not None and isinstance(charger_obs, dict)
    baseline_mm = max(0.0, min(400.0, float(getattr(args, "baseline_mm", 0.0))))
    if baseline_mm >= 15.0:
        bounded_drive(robot, baseline_mm, speed_mmps=60.0, standoff_mm=120.0)
        time.sleep(0.25)
    steps = max(2, min(MAX_SCAN_STEPS, int(args.steps)))
    assert 2 <= steps <= MAX_SCAN_STEPS
    for idx in range(steps):
        time.sleep(0.25)
        snap = _snapshot_pose_and_charger(robot)
        if isinstance(snap.get("charger"), dict) and snap["charger"].get("is_visible"):
            charger_obs["visible_snapshots"].append(snap)
        if idx + 1 < steps:
            mr = bounded_turn(robot, math.radians(float(args.step_deg)), speed_dps=45.0)
            if not mr.ok:
                charger_obs["sweep_stopped_reason"] = mr.detail
                break


def _return_robot_to_charger(robot: Any) -> str:
    """Turn toward known charger pose if off-charger and drive onto charger."""
    assert robot is not None
    status = getattr(robot, "status", None)
    if bool(getattr(status, "is_on_charger", False)):
        return "docked"
    world_ch = getattr(getattr(robot, "world", None), "charger", None)
    assert status is not None or world_ch is None
    if world_ch is not None and getattr(world_ch, "pose", None) is not None and getattr(robot, "pose", None) is not None:
        bp, cp = pose2d_from_sdk_pose(robot.pose), pose2d_from_sdk_pose(world_ch.pose)
        bounded_turn(robot, normalize_angle(math.atan2(cp.y_mm - bp.y_mm, cp.x_mm - bp.x_mm) - bp.theta_rad), speed_dps=45.0)
        dist = math.hypot(cp.x_mm - bp.x_mm, cp.y_mm - bp.y_mm)
        if dist > 220.0:
            bounded_drive(robot, min(dist - 140.0, 280.0), speed_mmps=55.0, standoff_mm=100.0)
        time.sleep(0.3)
    robot.behavior.drive_on_charger()
    return "docked" if bool(getattr(getattr(robot, "status", None), "is_on_charger", False)) else "dock_attempted"


def _run_live_visual_scan(
    args: argparse.Namespace, check_docked_illum: bool = False
) -> Tuple[CameraIntrinsics, List[Tuple[float, np.ndarray]], List[InertialOdomSample], str, Dict[str, Any]]:
    """Execute bounded live scan with continuous camera + IMU + odom sampling."""
    assert args is not None and hasattr(args, "serial")
    dock_note = "not_requested"
    charger_obs: Dict[str, Any] = {"docked": None, "after_undock": None, "visible_snapshots": [], "post_scan": None}
    with VectorSDKSession(serial=args.serial, active=True, via_vector=args.via_vector, enable_nav_map=True, enable_camera=True) as robot:
        ok, reason, tel = check_safety(robot, allow_on_charger=True)
        if not ok:
            raise RuntimeError(reason)
        if check_docked_illum:
            _verify_docked_illum_or_abort(robot, tel)
        charger_obs["docked"] = _snapshot_pose_and_charger(robot)
        if tel["is_on_charger"]:
            if not args.undock_if_needed:
                raise RuntimeError("on_charger")
            robot.behavior.drive_off_charger()
            charger_obs["after_undock"] = _snapshot_pose_and_charger(robot)
        execute_bounded_directive(robot, "lift", {"lift_norm": 0.0})
        execute_bounded_directive(robot, "head", {"head_deg": 0.0})
        collector = _BackgroundStreamCollector(robot)
        collector.start()
        try:
            _drive_baseline_and_sweep(robot, args, charger_obs)
        finally:
            intr_out, all_frames, _ids, all_odoms = collector.stop()
        if not all_frames:
            intr_out, all_frames, _ids, all_odoms, _exp = _collect_camera_stream(robot, duration_s=0.45)
        charger_obs["post_scan"] = _snapshot_pose_and_charger(robot)
        if args.return_charger:
            dock_note = _return_robot_to_charger(robot)
    assert intr_out is not None
    return (intr_out, all_frames, all_odoms, dock_note, charger_obs)


def _acquire_live_visual_inputs(
    args: argparse.Namespace,
) -> Tuple[Optional[Tuple[Any, List[Tuple[float, np.ndarray]], List[InertialOdomSample], str, Optional[Dict[str, Any]]]], int]:
    """Run dark/calm preflight and live visual scan, returning (tuple, 0) or (None, rc)."""
    check_docked = False
    if args.undock_if_needed and not args.force_dark:
        illum = _preflight_passive_illumination(args.serial)
        if not illum["visual_slam_viable"] and int(illum.get("frames_received", 0)) > 0:
            print(json.dumps({"ok": False, "error": "dark_room_refuse_undock", "illumination": illum}, indent=2))
            return (None, 4)
        if not illum["visual_slam_viable"]:
            if not args.via_vector:
                print(json.dumps({"ok": False, "error": "camera_paused_require_via_vector", "illumination": illum}, indent=2))
                return (None, 4)
            check_docked = True
    try:
        return (_run_live_visual_scan(args, check_docked_illum=check_docked), 0)
    except _DarkRoomAbort as exc:
        print(json.dumps({"ok": False, "error": "dark_room_refuse_undock", "illumination": exc.illumination}, indent=2))
        return (None, 4)
    except CliffHazard as exc:
        print(json.dumps({"ok": False, "error": exc.hazard, "refused": exc.refused, "detail": str(exc)}, indent=2))
        return (None, 6)


def cmd_visual_slam(args: argparse.Namespace) -> int:
    """Run visual-inertial pose-graph SLAM (replay offline or live with dark preflight gate)."""
    assert args is not None and hasattr(args, "atlas")
    tag = args.session_id or f"vslam-{int(time.time())}"
    dock_note, charger_obs = "replay_offline", None
    if args.replay_dataset:
        intr, frames, odoms = load_replay_dataset(Path(args.replay_dataset))
    else:
        bundle, rc = _acquire_live_visual_inputs(args)
        if bundle is None:
            return rc
        intr, frames, odoms, dock_note, charger_obs = bundle
    assert isinstance(frames, list)
    timing = measure_stream_timing([ts for ts, _ in frames]) if frames else {}
    if args.export_dataset and frames and not args.replay_dataset:
        export_slam_dataset(Path(args.export_dataset), frames, odoms, intr, fps=float(timing.get("mean_fps") or 15.0))
    slam, _vtraj, _otraj, summary = _run_slam_sequence(intr, frames, odoms)
    oid = odoms[0].pose.origin_id if odoms else 0
    vmap_path, sid, map_stats = _persist_visual_outputs(args, slam, tag, oid)
    out = {
        "ok": True, "mode": "replay" if args.replay_dataset else "live", "dock_result": dock_note,
        "charger_observations": charger_obs, "visual_map_path": str(vmap_path), "submap_id": sid,
        "keyframes": len(slam.keyframes), "loop_closures": slam.loop_closures, "summary": summary, "map_stats": map_stats,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_visual_relocalize(args: argparse.Namespace) -> int:
    """Relocalize passive live camera frame or --replay-frame against saved visual map."""
    vmap_path = _default_vmap_path(args)
    if not vmap_path.is_file():
        print(json.dumps({"ok": False, "error": f"visual_map_not_found:{vmap_path}"}))
        return 2
    slam = VisualPoseGraphSLAM.load(vmap_path)
    exp_ms = getattr(args, "exposure_ms", None)
    if args.replay_frame:
        with Image.open(Path(args.replay_frame)) as im:
            gray = pil_to_gray_numpy(im)
        res = slam.relocalize_frame(gray, exposure_ms=exp_ms)
    else:
        with VectorSDKSession(serial=args.serial, active=False, enable_nav_map=False, enable_camera=True) as robot:
            _intr, frames, _ids, _odoms, exp_state = _collect_camera_stream(robot, duration_s=1.0)
            if not frames:
                print(json.dumps({"ok": False, "matched": False, "reason": "no_frames_received"}))
                return 2
            res = slam.relocalize_frame(frames[-1][1], exposure_ms=exp_state.get("exposure_ms"), gain=exp_state.get("gain"))
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 2


def cmd_ros2_bridge(args: argparse.Namespace) -> int:
    """Check ROS 2 readiness or export passive camera + IMU + LaserScan ROS 2 bundle."""
    env = check_ros2_environment()
    if args.check and not args.export_dataset:
        print(json.dumps({"ok": True, "ros2": env}, indent=2))
        return 0
    bridge = VectorRos2Bridge(use_rclpy=bool(args.use_rclpy))
    atlas = NavMapAtlas.load(Path(args.atlas))
    try:
        with VectorSDKSession(serial=args.serial, active=False, enable_nav_map=False, enable_camera=True) as robot:
            intr, frames, _ids, odoms, _exp = _collect_camera_stream(robot, float(args.duration_s))
        for idx, (ts, gray) in enumerate(frames):
            odom = odoms[min(idx, len(odoms) - 1)]
            bridge.publish_camera_frame(ts, intr, gray)
            bridge.publish_imu(odom)
            bridge.publish_odom(odom)
            bridge.publish_laserscan(ts, synthesize_laserscan_from_tof_and_atlas(odom.pose, odom.prox_mm, odom.prox_found, atlas))
        manifest = export_slam_dataset(Path(args.export_dataset), frames, odoms, intr) if args.export_dataset and frames else None
        out = {"ok": True, "ros2": env, "published_messages": len(bridge.published), "exported_dataset": manifest}
        print(json.dumps(out, indent=2))
        return 0
    finally:
        bridge.shutdown()


def cmd_characterize_mechanisms(args: argparse.Namespace) -> int:
    """Execute physical mechanism system-id tests (head, lift, optionally wheels) and judge realism."""
    from .mechanism_characterize import run_mechanism_characterization

    assert args is not None and hasattr(args, "serial")
    if not getattr(args, "via_vector", False):
        print(json.dumps({"ok": False, "error": "via_vector_consent_required"}))
        return 6
    inc_wheels = bool(getattr(args, "include_wheels", False)) and not bool(getattr(args, "docked_only", False))
    try:
        with VectorSDKSession(
            serial=args.serial, active=True, via_vector=True,
            enable_nav_map=False, enable_camera=False, ip=getattr(args, "ip", None),
        ) as robot:
            report = run_mechanism_characterization(
                robot,
                include_wheels=inc_wheels,
                confirm_safe_floor=bool(getattr(args, "confirm_safe_floor", False)),
                allow_occluded_tof=bool(getattr(args, "allow_occluded_tof", False)),
            )
    except CliffHazard as exc:
        print(json.dumps({"ok": False, "error": exc.hazard, "refused": exc.refused, "detail": str(exc)}, indent=2))
        return 6
    rendered = json.dumps(report, indent=2)
    out_path = getattr(args, "out", None)
    if out_path:
        Path(out_path).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.get("ok") else 6


def _add_mech_subparser(sub: argparse._SubParsersAction) -> None:
    """Register characterize-mechanisms subcommand."""
    assert sub is not None
    sm = sub.add_parser("characterize-mechanisms")
    sm.add_argument("--via-vector", action="store_true")
    sm.add_argument("--docked-only", action="store_true")
    sm.add_argument("--include-wheels", action="store_true")
    sm.add_argument("--confirm-safe-floor", action="store_true")
    sm.add_argument("--allow-occluded-tof", action="store_true")
    sm.add_argument("--ip", default=None)
    sm.add_argument("--out", default=None)


def add_phase2_subparsers(sub: argparse._SubParsersAction) -> None:
    """Register Phase 2 subcommands on the root vector-slam CLI parser."""
    assert sub is not None
    sc = sub.add_parser("characterize-camera")
    sc.add_argument("--via-vector", action="store_true")
    sc.add_argument("--duration-s", type=float, default=5.0)
    sc.add_argument("--high-res-probe", action="store_true")
    sc.add_argument("--export-dataset", default=None)
    sv = sub.add_parser("visual-slam")
    sv.add_argument("--via-vector", action="store_true")
    sv.add_argument("--undock-if-needed", action="store_true")
    sv.add_argument("--return-charger", action="store_true")
    sv.add_argument("--force-dark", action="store_true")
    sv.add_argument("--baseline-mm", type=float, default=0.0)
    sv.add_argument("--steps", type=int, default=8)
    sv.add_argument("--step-deg", type=float, default=45.0)
    sv.add_argument("--replay-dataset", default=None)
    sv.add_argument("--visual-map", default=None)
    sv.add_argument("--export-dataset", default=None)
    sr = sub.add_parser("visual-relocalize")
    sr.add_argument("--visual-map", default=None)
    sr.add_argument("--replay-frame", default=None)
    sb = sub.add_parser("ros2-bridge")
    sb.add_argument("--check", action="store_true")
    sb.add_argument("--duration-s", type=float, default=2.0)
    sb.add_argument("--export-dataset", default=None)
    sb.add_argument("--use-rclpy", action="store_true")
    _add_mech_subparser(sub)
    assert callable(cmd_characterize_camera) and callable(cmd_characterize_mechanisms)


def phase2_dispatch() -> Dict[str, Any]:
    """Return subcommand handler map for Phase 2 CLI integration."""
    return {
        "characterize-camera": cmd_characterize_camera,
        "characterize-mechanisms": cmd_characterize_mechanisms,
        "visual-slam": cmd_visual_slam,
        "visual-relocalize": cmd_visual_relocalize,
        "ros2-bridge": cmd_ros2_bridge,
    }
