"""Phase 2 CLI subcommands: camera characterization, ROS 2 bridge, visual SLAM, relocalization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .camera_characterize import (
    assess_illumination,
    estimate_rolling_shutter_metrics,
    evaluate_visual_vs_odom_drift,
    measure_sharpness_tenengrad,
    measure_stream_timing,
)
from .atlas_lock import atlas_file_lock
from .navmap_atlas import NavMapAtlas
from .sdk_session import (
    VectorSDKSession,
    capture_high_res_single_image,
    check_safety,
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
    try:
        if hasattr(cam, "image_streaming_enabled") and cam.image_streaming_enabled():
            cam.close_camera_feed()
            time.sleep(0.08)
    except (OSError, RuntimeError, TimeoutError):
        pass
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


def cmd_characterize_camera(args: argparse.Namespace) -> int:
    """Passively measure camera intrinsics, FPS, jitter, illumination, exposure, and dataset."""
    hr_info: Optional[Dict[str, Any]] = None
    with VectorSDKSession(serial=args.serial, active=False, enable_nav_map=False, enable_camera=True) as robot:
        _ok, safety_reason, tel = check_safety(robot, allow_on_charger=True)
        intr, frames, fids, odoms, exp_state = _collect_camera_stream(robot, float(args.duration_s))
        if args.high_res_probe:
            hr_info = _probe_high_res(robot, active_control=False)
    timing = measure_stream_timing([ts for ts, _ in frames], frame_ids=fids)
    illum = (
        assess_illumination(
            frames[-1][1],
            exposure_ms=exp_state.get("exposure_ms"),
            max_exposure_ms=exp_state.get("max_exposure_ms"),
            gain=exp_state.get("gain"),
            max_gain=exp_state.get("max_gain"),
        )
        if frames
        else _empty_frame_illumination(tel, exp_state)
    )
    rs = estimate_rolling_shutter_metrics([g for _, g in frames], [o.gyro_xyz_rad_s[2] for o in odoms], [ts for ts, _ in frames], intr.fx)
    manifest = export_slam_dataset(Path(args.export_dataset), frames, odoms, intr, fps=timing["mean_fps"] or 15.0) if args.export_dataset and frames else None
    out = {
        "ok": bool(frames),
        "status": "sampled" if frames else illum["status"],
        "safety": safety_reason,
        "telemetry": tel,
        "intrinsics": intr.to_dict(),
        "exposure": exp_state,
        "timing": timing,
        "illumination": illum,
        "rolling_shutter": rs,
        "high_res_probe": hr_info,
        "exported_dataset": manifest,
    }
    print(json.dumps(out, indent=2))
    return 0


def _preflight_passive_illumination(serial: str) -> Dict[str, Any]:
    """Check docked/passive camera illumination before authorizing active undock."""
    with VectorSDKSession(serial=serial, active=False, enable_nav_map=False, enable_camera=True) as robot:
        _ok, _reason, tel = check_safety(robot, allow_on_charger=True)
        _intr, frames, _fids, _odoms, exp_state = _collect_camera_stream(robot, duration_s=1.2)
    if not frames:
        return _empty_frame_illumination(tel, exp_state)
    return assess_illumination(
        frames[-1][1],
        exposure_ms=exp_state.get("exposure_ms"),
        max_exposure_ms=exp_state.get("max_exposure_ms"),
        gain=exp_state.get("gain"),
        max_gain=exp_state.get("max_gain"),
    )


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


def _run_live_visual_scan(args: argparse.Namespace) -> Tuple[CameraIntrinsics, List[Tuple[float, np.ndarray]], List[InertialOdomSample], str]:
    """Execute bounded live turn scan with camera + IMU + wheel odom (--via-vector)."""
    from anki_vector.util import degrees, radians  # type: ignore

    steps = max(2, min(MAX_SCAN_STEPS, int(args.steps)))
    all_frames: List[Tuple[float, np.ndarray]] = []
    all_odoms: List[InertialOdomSample] = []
    intr_out: Optional[CameraIntrinsics] = None
    dock_note = "not_requested"
    with VectorSDKSession(serial=args.serial, active=True, via_vector=args.via_vector, enable_nav_map=True, enable_camera=True) as robot:
        ok, reason, tel = check_safety(robot, allow_on_charger=True)
        if not ok:
            raise RuntimeError(reason)
        if tel["is_on_charger"]:
            if not args.undock_if_needed:
                raise RuntimeError("on_charger")
            robot.behavior.drive_off_charger()
        robot.behavior.set_lift_height(0.0)
        robot.behavior.set_head_angle(degrees(0.0))
        for idx in range(steps):
            intr, fs, _ids, ods, _exp = _collect_camera_stream(robot, duration_s=0.45)
            intr_out = intr
            all_frames.extend(fs)
            all_odoms.extend(ods)
            if idx + 1 < steps:
                robot.behavior.turn_in_place(degrees(float(args.step_deg)))
        if args.return_charger:
            world_ch = getattr(getattr(robot, "world", None), "charger", None)
            if world_ch is not None and getattr(world_ch, "pose", None) is not None:
                bp, cp = pose2d_from_sdk_pose(robot.pose), pose2d_from_sdk_pose(world_ch.pose)
                robot.behavior.turn_in_place(radians(normalize_angle(math.atan2(cp.y_mm - bp.y_mm, cp.x_mm - bp.x_mm) - bp.theta_rad)))
            robot.behavior.drive_on_charger()
            dock_note = "docked" if getattr(getattr(robot, "status", None), "is_on_charger", False) else "dock_attempted"
    assert intr_out is not None
    return (intr_out, all_frames, all_odoms, dock_note)


def cmd_visual_slam(args: argparse.Namespace) -> int:
    """Run visual-inertial pose-graph SLAM (replay offline or live with dark preflight gate)."""
    tag = args.session_id or f"vslam-{int(time.time())}"
    dock_note = "replay_offline"
    if args.replay_dataset:
        intr, frames, odoms = load_replay_dataset(Path(args.replay_dataset))
    else:
        if args.undock_if_needed and not args.force_dark:
            illum = _preflight_passive_illumination(args.serial)
            if not illum["visual_slam_viable"]:
                print(json.dumps({"ok": False, "error": "dark_room_refuse_undock", "illumination": illum}, indent=2))
                return 4
        intr, frames, odoms, dock_note = _run_live_visual_scan(args)
    if args.export_dataset and frames and not args.replay_dataset:
        export_slam_dataset(Path(args.export_dataset), frames, odoms, intr)
    slam, _vtraj, _otraj, summary = _run_slam_sequence(intr, frames, odoms)
    oid = odoms[0].pose.origin_id if odoms else 0
    vmap_path, sid, map_stats = _persist_visual_outputs(args, slam, tag, oid)
    out = {
        "ok": True,
        "mode": "replay" if args.replay_dataset else "live",
        "dock_result": dock_note,
        "visual_map_path": str(vmap_path),
        "submap_id": sid,
        "keyframes": len(slam.keyframes),
        "loop_closures": slam.loop_closures,
        "summary": summary,
        "map_stats": map_stats,
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


def add_phase2_subparsers(sub: argparse._SubParsersAction) -> None:
    """Register Phase 2 subcommands on the root vector-slam CLI parser."""
    sc = sub.add_parser("characterize-camera")
    sc.add_argument("--duration-s", type=float, default=5.0)
    sc.add_argument("--high-res-probe", action="store_true")
    sc.add_argument("--export-dataset", default=None)
    sv = sub.add_parser("visual-slam")
    sv.add_argument("--via-vector", action="store_true")
    sv.add_argument("--undock-if-needed", action="store_true")
    sv.add_argument("--return-charger", action="store_true")
    sv.add_argument("--force-dark", action="store_true")
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


def phase2_dispatch() -> Dict[str, Any]:
    """Return subcommand handler map for Phase 2 CLI integration."""
    return {
        "characterize-camera": cmd_characterize_camera,
        "visual-slam": cmd_visual_slam,
        "visual-relocalize": cmd_visual_relocalize,
        "ros2-bridge": cmd_ros2_bridge,
    }
