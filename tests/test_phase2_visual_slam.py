"""Offline deterministic unit tests for Vector SLAM Phase 2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vector_slam_maze import cli, cli_phase2
from vector_slam_maze.camera_characterize import (
    assess_illumination,
    estimate_rolling_shutter_metrics,
    evaluate_visual_vs_odom_drift,
    measure_stream_timing,
    phase_correlate_1d_shift,
)
from vector_slam_maze.navmap_atlas import NavMapAtlas
from vector_slam_maze.sensor_bridge import (
    CameraIntrinsics,
    InertialOdomSample,
    VectorRos2Bridge,
    check_ros2_environment,
    export_slam_dataset,
    load_replay_dataset,
    synthesize_laserscan_from_tof_and_atlas,
)
from vector_slam_maze.types_and_math import NavContent, Pose2D, normalize_angle
from vector_slam_maze.visual_slam import (
    VisualPoseGraphSLAM,
    compute_appearance_signature,
    estimate_visual_yaw_zncc,
    extract_keypoints_and_descriptors,
)


def _build_pano_canvas() -> np.ndarray:
    """Create deterministic non-repeating 360-deg panoramic scene with sharp corners."""
    rng = np.random.default_rng(42)
    h_c, w_c = 256, 4096
    canvas = np.full((h_c, w_c), 125.0, dtype=np.float32)
    xs = np.linspace(0.0, 2.0 * np.pi, w_c, endpoint=False, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h_c, dtype=np.float32)
    canvas += 25.0 * np.sin(xs + 0.3)[None, :] + 15.0 * np.cos(2.0 * xs)[None, :] * ys[:, None]
    for _ in range(320):
        x0 = int(rng.integers(0, w_c - 64))
        y0 = int(rng.integers(12, h_c - 52))
        bw, bh = int(rng.integers(14, 72)), int(rng.integers(14, 52))
        canvas[y0 : y0 + bh, x0 : x0 + bw] = float(rng.integers(20, 235))
    canvas[:, -64:] = canvas[:, :64]
    return canvas


_PANO_CANVAS = _build_pano_canvas()


def _render_pano_view(
    theta_rad: float,
    intr: CameraIntrinsics,
    omega_rad_s: float = 0.0,
    readout_s: float = 0.0,
) -> np.ndarray:
    """Render bilinear pinhole view of deterministic 360-deg canvas + optional rolling shutter."""
    h_c, w_c = _PANO_CANVAS.shape
    us = np.arange(intr.width, dtype=np.float32)
    vs = np.arange(intr.height, dtype=np.float32)
    alpha_u = np.arctan2(us - intr.cx, intr.fx)[None, :]
    theta_row = (theta_rad + omega_rad_s * (vs / float(intr.height)) * readout_s)[:, None]
    az = (theta_row - alpha_u) % (2.0 * np.pi)
    xf = az * (w_c / (2.0 * np.pi))
    yf = np.clip((0.5 + 0.45 * ((vs - intr.cy) / intr.fy)) * (h_c - 2), 0.0, float(h_c - 2))[:, None]
    x0 = np.floor(xf).astype(np.int32) % w_c
    y0 = np.floor(yf).astype(np.int32)
    dx, dy = xf - x0, yf - y0
    x1 = (x0 + 1) % w_c
    y1 = y0 + 1
    p00 = _PANO_CANVAS[y0, x0]
    p10 = _PANO_CANVAS[y0, x1]
    p01 = _PANO_CANVAS[y1, x0]
    p11 = _PANO_CANVAS[y1, x1]
    return ((1 - dy) * ((1 - dx) * p00 + dx * p10) + dy * ((1 - dx) * p01 + dx * p11)).astype(np.float32)


def _make_sample_odom(ts: float, pose: Pose2D, gz: float = 0.0, prox_mm: float = 340.0) -> InertialOdomSample:
    """Build deterministic InertialOdomSample fixture."""
    return InertialOdomSample(
        timestamp_s=ts,
        pose=pose,
        pitch_rad=0.01,
        gyro_xyz_rad_s=(0.0, 0.0, gz),
        accel_xyz_m_s2=(0.0, 0.0, 9.81),
        prox_mm=prox_mm,
        prox_found=True,
    )


class TestSensorBridgeAndCharacterize(unittest.TestCase):
    """Tests for camera intrinsics, dataset bridge, ROS 2 bus, and characterization."""

    def test_camera_intrinsics_auto_scaling(self) -> None:
        """Verify CameraConfig at 320x180 and 640x360 scales without double-scaling."""
        cfg_320 = SimpleNamespace(
            focal_length=SimpleNamespace(x=288.0, y=288.0),
            center=SimpleNamespace(x=155.0, y=111.0),
            fov_x=SimpleNamespace(degrees=90.0),
            fov_y=SimpleNamespace(degrees=56.0),
        )
        i640 = CameraIntrinsics.from_sdk_config(cfg_320, 640, 360)
        self.assertAlmostEqual(i640.fx, 576.0, places=2)
        self.assertAlmostEqual(i640.cx, 310.0, places=2)
        i1280 = CameraIntrinsics.from_sdk_config(cfg_320, 1280, 720)
        self.assertAlmostEqual(i1280.fx, 1152.0, places=2)
        cfg_640 = SimpleNamespace(focal_length=SimpleNamespace(x=576.0, y=576.0), center=SimpleNamespace(x=310.0, y=222.0))
        same = CameraIntrinsics.from_sdk_config(cfg_640, 640, 360)
        self.assertAlmostEqual(same.fx, 576.0, places=2)
        self.assertGreater(same.pixel_to_bearing_rad(same.cx + 80.0), 0.0)

    def test_virtual_laserscan_from_tof_and_atlas(self) -> None:
        """Verify center ray takes live ToF and peripheral rays hit atlas walls."""
        atlas = NavMapAtlas(cell_mm=20.0)
        layer = atlas.ensure_submap("s1:origin_0", origin_id=0)
        for dy_ang in (-0.3, -0.15, 0.15, 0.3):
            layer.integrate_tof_ray(Pose2D(0.0, 0.0, dy_ang), prox_mm=500.0, found_object=True, lift_in_fov=False)
        atlas.assume_same_frame_as_canonical("s1:origin_0")
        scan = synthesize_laserscan_from_tof_and_atlas(Pose2D(0.0, 0.0, 0.0), prox_mm=310.0, prox_found=True, atlas=atlas, num_rays=21)
        mid = len(scan["ranges"]) // 2
        self.assertAlmostEqual(scan["ranges"][mid], 0.31, places=3)
        self.assertTrue(any(0.45 <= r <= 0.60 for idx, r in enumerate(scan["ranges"]) if idx != mid))
        self.assertEqual(scan["frame_id"], "base_scan")

    def test_tum_euroc_export_and_replay_roundtrip(self) -> None:
        """Verify TUM/EuRoC/ORB-SLAM3/RTAB-Map bundle exports and replays cleanly."""
        intr = CameraIntrinsics(320, 180, 288.0, 288.0, 155.0, 111.0)
        frames = [(100.0 + i * 0.1, _render_pano_view(0.1 * i, intr)) for i in range(4)]
        odoms = [_make_sample_odom(100.0 + i * 0.1, Pose2D(0.0, 0.0, 0.1 * i), gz=1.0) for i in range(4)]
        out_dir = Path(self.id().replace(".", "_"))
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / out_dir.name
            manifest = export_slam_dataset(d, frames, odoms, intr, fps=10.0)
            self.assertEqual(manifest["frame_count"], 4)
            for fname in manifest["files"]:
                self.assertTrue((d / fname).is_file(), fname)
            yaml_txt = (d / "orb_slam3_vector.yaml").read_text(encoding="utf-8")
            self.assertIn("Camera1.fx: 288.0000", yaml_txt)
            self.assertIn("rtabmap_slam", (d / "rtabmap_ros2_launch.py").read_text(encoding="utf-8"))
            r_intr, r_frames, r_odoms = load_replay_dataset(d)
            self.assertEqual(r_intr.width, 320)
            self.assertEqual(len(r_frames), 4)
            self.assertEqual(len(r_odoms), 4)

    def test_ros2_bridge_offline_and_check(self) -> None:
        """Verify ROS 2 offline bridge bus topics and non-failing ros2-bridge --check."""
        intr = CameraIntrinsics(320, 180, 288.0, 288.0, 155.0, 111.0)
        bridge = VectorRos2Bridge(use_rclpy=False)
        gray = np.zeros((180, 320), dtype=np.float32)
        samp = _make_sample_odom(1.0, Pose2D(10.0, 20.0, 0.2), gz=0.5)
        bridge.publish_camera_frame(1.0, intr, gray)
        bridge.publish_imu(samp)
        bridge.publish_odom(samp)
        bridge.publish_laserscan(1.0, synthesize_laserscan_from_tof_and_atlas(samp.pose, 250.0, True))
        topics = [t for t, _ in bridge.published]
        self.assertEqual(topics, ["/vector/camera/camera_info", "/vector/camera/image_raw", "/vector/imu", "/vector/odom", "/vector/scan"])
        self.assertTrue(check_ros2_environment()["offline_bridge_ready"])
        self.assertEqual(cli.main(["ros2-bridge", "--check"]), 0)

    def test_phase_correlate_and_rolling_shutter_recovery(self) -> None:
        """Verify 1D subpixel shift recovery and synthetic 30ms rolling shutter estimation."""
        xs = np.arange(256, dtype=np.float64)
        sig_a = np.sin(2.0 * np.pi * xs / 32.0) + 0.6 * np.cos(2.0 * np.pi * xs / 19.0)
        shift_true = 13.4
        sig_b = np.sin(2.0 * np.pi * (xs - shift_true) / 32.0) + 0.6 * np.cos(2.0 * np.pi * (xs - shift_true) / 19.0)
        est, conf = phase_correlate_1d_shift(sig_a, sig_b)
        self.assertGreater(conf, 0.2)
        self.assertAlmostEqual(est, shift_true, delta=0.25)
        intr = CameraIntrinsics(640, 360, 576.0, 576.0, 320.0, 180.0)
        f0 = _render_pano_view(0.0, intr, omega_rad_s=0.0, readout_s=0.030)
        f1 = _render_pano_view(0.132, intr, omega_rad_s=2.0, readout_s=0.030)
        rs = estimate_rolling_shutter_metrics([f0, f1], [0.0, 2.0], [10.0, 10.066], fx=intr.fx)
        self.assertTrue(rs["evaluated"])
        self.assertAlmostEqual(rs["estimated_readout_ms"], 30.0, delta=3.5)
        timing = measure_stream_timing([1.0, 1.067, 1.134, 1.201], frame_ids=[1, 2, 3, 5])
        self.assertEqual(timing["dropped_frame_ids_est"], 1)

    def test_probe_high_res_passive_gate(self) -> None:
        """Verify passive characterize-camera skips behavior-control capture cleanly."""
        fake_cam = SimpleNamespace(image_streaming_enabled=lambda: True, close_camera_feed=lambda: (_ for _ in ()).throw(AssertionError("should not close")))
        res = cli_phase2._probe_high_res(SimpleNamespace(camera=fake_cam), active_control=False)
        self.assertFalse(res["captured"])
        self.assertEqual(res["reason"], "requires_behavior_control")

    def test_nonblocking_camera_poll(self) -> None:
        """Verify poll_latest_camera_image never triggers blocking latest_image when _latest_image exists."""
        from vector_slam_maze import sdk_session

        class BlockingCam:
            def __init__(self) -> None:
                self._latest_image = None
                self.config = None
                self.exposure_ms = 0
                self.gain = 0.0
                self.is_auto_exposure_enabled = True

            @property
            def latest_image(self) -> object:
                raise AssertionError("blocking latest_image getter must not be invoked")

        bot = SimpleNamespace(camera=BlockingCam())
        self.assertIsNone(sdk_session.poll_latest_camera_image(bot, max_wait_s=0.05, step_s=0.02))
        bot.camera._latest_image = SimpleNamespace(image_id=7, raw_image=np.zeros((16, 16), dtype=np.uint8))
        got = sdk_session.poll_latest_camera_image(bot, max_wait_s=0.05, step_s=0.02, min_image_id=5)
        self.assertEqual(got.image_id, 7)

    def test_calm_mode_no_frames_fast_exit(self) -> None:
        """Verify calm power mode with 0 frames exits cleanly within duration_s."""
        import io
        import time
        from contextlib import redirect_stdout

        cam = SimpleNamespace(_latest_image=None, config=None, exposure_ms=0, gain=0.0, is_auto_exposure_enabled=True)
        bot = SimpleNamespace(
            camera=cam,
            status=SimpleNamespace(is_on_charger=True, is_in_calm_power_mode=True, is_charging=True),
            proximity=SimpleNamespace(last_sensor_reading=None),
            pose=None,
            get_battery_state=lambda: SimpleNamespace(battery_volts=4.06, battery_level=2),
        )
        t0 = time.monotonic()
        intr, frames, _ids, _ods, _exp = cli_phase2._collect_camera_stream(bot, duration_s=0.22)
        self.assertEqual(len(frames), 0)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(intr.width, 640)

        class CalmSess:
            def __init__(self, **_kw: object) -> None:
                pass

            def __enter__(self) -> object:
                return bot

            def __exit__(self, *a: object) -> None:
                return None

        buf = io.StringIO()
        with patch("vector_slam_maze.cli_phase2.VectorSDKSession", CalmSess), redirect_stdout(buf):
            rc = cli.main(["characterize-camera", "--duration-s", "0.22", "--high-res-probe"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "camera_paused_calm_power_mode")
        self.assertEqual(payload["illumination"]["frames_received"], 0)
        self.assertFalse(payload["high_res_probe"]["captured"])


class TestVisualSlamAndDarkGating(unittest.TestCase):
    """Tests for visual ZNCC yaw, darkness gate, loop closure, and relocalization."""

    def test_visual_yaw_zncc_sign_and_precision(self) -> None:
        """Verify mutual ZNCC recovers known +0.18 rad CCW camera rotation accurately."""
        intr = CameraIntrinsics(640, 360, 576.0, 576.0, 310.0, 222.0)
        fa = _render_pano_view(0.25, intr)
        fb = _render_pano_view(0.43, intr)
        ka = extract_keypoints_and_descriptors(fa)
        kb = extract_keypoints_and_descriptors(fb)
        dyaw, info = estimate_visual_yaw_zncc(ka, kb, intr.fx, intr.cx)
        self.assertIsNotNone(dyaw)
        assert dyaw is not None
        self.assertAlmostEqual(dyaw, 0.18, delta=0.02)
        self.assertGreaterEqual(info["inliers"], 8)

    def test_dark_room_gating_and_blank_rejection(self) -> None:
        """Verify dark and flat frames never produce keyframes or false loop closures."""
        rng = np.random.default_rng(7)
        intr = CameraIntrinsics(320, 180, 288.0, 288.0, 155.0, 111.0)
        slam = VisualPoseGraphSLAM(intr)
        for i in range(6):
            dark = rng.normal(6.0, 1.8, size=(180, 320)).clip(0, 255).astype(np.float32)
            out = slam.process_frame(dark, _make_sample_odom(float(i), Pose2D(0.0, 0.0, 0.4 * i)), float(i), exposure_ms=65, gain=4.0)
            self.assertFalse(out["accepted_visual"])
            self.assertEqual(out["illumination"]["status"], "dark_room")
        flat = np.full((180, 320), 120.0, dtype=np.float32) + rng.normal(0.0, 1.2, size=(180, 320)).astype(np.float32)
        self.assertEqual(assess_illumination(flat)["status"], "low_texture")
        self.assertEqual(len(slam.keyframes), 0)
        self.assertEqual(len(slam.loop_closures), 0)
        self.assertFalse(slam.relocalize_frame(np.full((180, 320), 5.0, dtype=np.float32))["matched"])

    def test_dark_room_cli_refuses_undock(self) -> None:
        """Verify visual-slam --undock-if-needed aborts safely in dark rooms before takeover."""
        dark_img = np.full((180, 320), 4.0, dtype=np.float32)
        calls: list[bool] = []

        class FakeSess:
            def __init__(self, **kw: object) -> None:
                calls.append(bool(kw.get("active", False)))

            def __enter__(self) -> object:
                return SimpleNamespace(camera=SimpleNamespace(config=None, exposure_ms=66, gain=4.0))

            def __exit__(self, *a: object) -> None:
                return None

        fake_stream = (CameraIntrinsics(320, 180, 288.0, 288.0, 155.0, 111.0), [(1.0, dark_img)], [1], [_make_sample_odom(1.0, Pose2D(0, 0, 0))], {"exposure_ms": 66, "max_exposure_ms": 66})
        with patch("vector_slam_maze.cli_phase2.VectorSDKSession", FakeSess), patch("vector_slam_maze.cli_phase2._collect_camera_stream", return_value=fake_stream):
            rc = cli.main(["visual-slam", "--via-vector", "--undock-if-needed"])
        self.assertEqual(rc, 4)
        self.assertEqual(calls, [False])

    def test_360_loop_closure_and_cross_session_relocalize(self) -> None:
        """Verify 360-deg visual SLAM closes +15 deg yaw drift and relocalizes across sessions."""
        intr = CameraIntrinsics(480, 270, 432.0, 432.0, 240.0, 135.0)
        slam = VisualPoseGraphSLAM(intr, min_keyframe_rad=0.20, min_loop_gap=4)
        n_steps = 13  # 0 to 360 deg in 30 deg increments
        for i in range(n_steps):
            true_th = normalize_angle(i * math.radians(30.0))
            noisy_odom_th = normalize_angle(i * math.radians(31.25))  # +15 deg drift at 360 deg
            frame = _render_pano_view(true_th, intr)
            res = slam.process_frame(frame, _make_sample_odom(float(i), Pose2D(0.0, 0.0, noisy_odom_th), gz=0.5), float(i))
            self.assertTrue(res["accepted_visual"])
        self.assertGreaterEqual(len(slam.loop_closures), 1)
        final_err_deg = math.degrees(abs(normalize_angle(slam.keyframes[-1].pose.theta_rad - 0.0)))
        self.assertLess(final_err_deg, 3.0)
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / "vslam.json"
            slam.save(vpath)
            reloaded = VisualPoseGraphSLAM.load(vpath)
            reloc = reloaded.relocalize_frame(_render_pano_view(math.radians(63.0), intr))
            self.assertTrue(reloc["matched"])
            reloc_err = math.degrees(abs(normalize_angle(reloc["pose"]["theta_rad"] - math.radians(63.0))))
            self.assertLess(reloc_err, 3.0)
            atlas = NavMapAtlas()
            sid, ups = reloaded.sync_into_navmap_atlas(atlas, "s-vis")
            self.assertIn("vslam_origin_0", sid)
            self.assertGreater(ups, 0)

    def test_cli_replay_dataset_and_drift_eval(self) -> None:
        """Verify offline CLI visual-slam --replay-dataset and visual-relocalize --replay-frame."""
        intr = CameraIntrinsics(320, 180, 288.0, 288.0, 155.0, 111.0)
        frames = [(10.0 + i * 0.2, _render_pano_view(i * 0.5, intr)) for i in range(14)]
        odoms = [_make_sample_odom(10.0 + i * 0.2, Pose2D(0.0, 0.0, normalize_angle(i * 0.5)), gz=0.5) for i in range(14)]
        ev = evaluate_visual_vs_odom_drift([o.pose for o in odoms], [o.pose for o in odoms])
        self.assertEqual(ev["final_yaw_diff_deg"], 0.0)
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            ds = d / "ds"
            export_slam_dataset(ds, frames, odoms, intr)
            rc1 = cli.main(["--atlas", str(d / "atlas.json"), "--export-dir", str(d / "ros"), "visual-slam", "--replay-dataset", str(ds)])
            self.assertEqual(rc1, 0)
            self.assertTrue((d / "atlas.vslam.json").is_file())
            self.assertTrue((d / "ros" / "grid.pgm").is_file())
            rc2 = cli.main(["--atlas", str(d / "atlas.json"), "visual-relocalize", "--replay-frame", str(ds / "frames" / "frame_000001.png")])
            self.assertEqual(rc2, 0)
            self.assertEqual(compute_appearance_signature(frames[0][1]).shape, (288,))

    def test_loop_closure_keeps_odometry_translation(self) -> None:
        """Verify a yaw-only loop closure never pulls translated keyframes together.

        Driving straight past a distant scene yields near-identical frames, so
        loop closures fire. Vision here observes heading only, so the closures
        must leave the odometry positions alone.
        """
        intr = CameraIntrinsics(640, 360, 576.0, 576.0, 310.0, 222.0)
        slam = VisualPoseGraphSLAM(intr)
        view = _render_pano_view(0.25, intr)
        for i in range(12):
            slam.process_frame(view, _make_sample_odom(100.0 + 0.1 * i, Pose2D(50.0 * i, 0.0, 0.0)), 100.0 + 0.1 * i)
        self.assertGreater(len(slam.loop_closures), 0)
        self.assertEqual(len(slam.keyframes), 12)
        for i, kf in enumerate(slam.keyframes):
            self.assertAlmostEqual(kf.pose.x_mm, 50.0 * i, delta=1.0)
            self.assertAlmostEqual(kf.pose.y_mm, 0.0, delta=1.0)
        self.assertAlmostEqual(slam.current_pose.x_mm, 550.0, delta=1.0)

    def test_dark_gap_does_not_reuse_stale_keypoints(self) -> None:
        """Verify heading follows odometry across dark frames instead of jumping."""
        intr = CameraIntrinsics(640, 360, 576.0, 576.0, 310.0, 222.0)
        slam = VisualPoseGraphSLAM(intr)
        dark = np.full((360, 640), 4.0, dtype=np.float32)
        sequence = [(0.0, False)] + [(0.05 * k, True) for k in range(1, 7)] + [(0.32, False)]
        ts = 100.0
        for yaw, is_dark in sequence:
            frame = dark if is_dark else _render_pano_view(yaw, intr)
            slam.process_frame(frame, _make_sample_odom(ts, Pose2D(0.0, 0.0, yaw)), ts)
            ts += 0.1
        self.assertAlmostEqual(slam.current_pose.theta_rad, 0.32, delta=0.01)


if __name__ == "__main__":
    unittest.main()
