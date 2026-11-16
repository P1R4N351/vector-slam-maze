"""ROS 2, ORB-SLAM3, RTAB-Map, TUM, and EuRoC sensor bridge for Anki Vector."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .navmap_atlas import MAX_RAY_STEPS, OCC_THRESHOLD, NavMapAtlas
from .types_and_math import Pose2D, normalize_angle, pose2d_from_sdk_pose

MAX_SCAN_RAYS = 720
MAX_EXPORT_FRAMES = 5000
MAX_EXPORT_ODOM = 20000
DEFAULT_TOF_MAX_RANGE_M = 1.0
DEFAULT_SCAN_FOV_DEG = 60.0
DEFAULT_SCAN_RAYS = 31


def _opt_deg(val: Any, default: float) -> float:
    """Extract angle degrees from SDK Angle object or float scalar."""
    if val is None:
        return default
    deg = getattr(val, "degrees", val)
    return float(deg) if isinstance(deg, (int, float)) else default


def _infer_calib_base(cfg: Any, cx_raw: float) -> Tuple[float, float]:
    """Infer native calibration pixel dimensions from CameraConfig fields."""
    w = float(getattr(cfg, "width", 0.0) or 0.0)
    h = float(getattr(cfg, "height", 0.0) or 0.0)
    if w > 0.0 and h > 0.0:
        return (w, h)
    if cx_raw <= 220.0:
        return (320.0, 180.0)
    if cx_raw <= 440.0:
        return (640.0, 360.0)
    return (1280.0, 720.0)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole + plumb-bob camera intrinsics scaled to active frame size."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fov_x_deg: float = 90.0
    fov_y_deg: float = 56.0
    distortion_k: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_sdk_config(
        cls, cfg: Any, target_width: int = 640, target_height: int = 360
    ) -> "CameraIntrinsics":
        """Build CameraIntrinsics from anki_vector CameraConfig, scaling to frame size."""
        assert target_width > 0 and target_height > 0
        if cfg is None:
            f = float(target_width) * 0.9
            return cls(target_width, target_height, f, f, target_width * 0.5, target_height * 0.5)
        fl, ct = getattr(cfg, "focal_length", None), getattr(cfg, "center", None)
        fx_r = float(getattr(fl, "x", 288.0))
        fy_r = float(getattr(fl, "y", 288.0))
        cx_r = float(getattr(ct, "x", 155.0))
        cy_r = float(getattr(ct, "y", 111.0))
        nw, nh = _infer_calib_base(cfg, cx_r)
        sx, sy = target_width / nw, target_height / nh
        fov_x = _opt_deg(getattr(cfg, "fov_x", None), 90.0)
        fov_y = _opt_deg(getattr(cfg, "fov_y", None), 56.0)
        return cls(target_width, target_height, fx_r * sx, fy_r * sy, cx_r * sx, cy_r * sy, fov_x, fov_y)

    def pixel_to_bearing_rad(self, u: float) -> float:
        """Convert image column u to horizontal bearing (+right)."""
        return math.atan2(u - self.cx, self.fx)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize intrinsics to ROS camera_info JSON dictionary."""
        k_mat = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        return {
            "width": self.width,
            "height": self.height,
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "fov_x_deg": round(self.fov_x_deg, 3),
            "fov_y_deg": round(self.fov_y_deg, 3),
            "distortion_model": "plumb_bob",
            "D": list(self.distortion_k),
            "K": [round(v, 4) for v in k_mat],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CameraIntrinsics":
        """Reconstruct CameraIntrinsics from JSON dictionary."""
        d_raw = raw.get("D", [0.0, 0.0, 0.0, 0.0, 0.0])
        dist = tuple(float(v) for v in (list(d_raw) + [0.0] * 5)[:5])
        return cls(
            width=int(raw["width"]),
            height=int(raw["height"]),
            fx=float(raw["fx"]),
            fy=float(raw["fy"]),
            cx=float(raw["cx"]),
            cy=float(raw["cy"]),
            fov_x_deg=float(raw.get("fov_x_deg", 90.0)),
            fov_y_deg=float(raw.get("fov_y_deg", 56.0)),
            distortion_k=dist,  # type: ignore[arg-type]
        )


@dataclass
class CameraFrameSample:
    """Single timestamped camera frame metadata + optional grayscale array."""

    frame_id: int
    timestamp_s: float
    width: int
    height: int
    mean_luma: float
    contrast_std: float
    sharpness_tenengrad: float
    exposure_ms: Optional[int] = None
    gain: Optional[float] = None
    auto_exposure: Optional[bool] = None
    gray: Optional[np.ndarray] = field(default=None, repr=False)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Serialize frame metrics without pixel payload."""
        return {
            "frame_id": self.frame_id,
            "timestamp_s": round(self.timestamp_s, 6),
            "width": self.width,
            "height": self.height,
            "mean_luma": round(self.mean_luma, 3),
            "contrast_std": round(self.contrast_std, 3),
            "sharpness_tenengrad": round(self.sharpness_tenengrad, 3),
            "exposure_ms": self.exposure_ms,
            "gain": self.gain,
            "auto_exposure": self.auto_exposure,
        }


@dataclass(frozen=True)
class InertialOdomSample:
    """Synchronized wheel odometry, 6-axis IMU, pitch, and ToF sample."""

    timestamp_s: float
    pose: Pose2D
    pitch_rad: float = 0.0
    gyro_xyz_rad_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_xyz_m_s2: Tuple[float, float, float] = (0.0, 0.0, 9.81)
    left_wheel_mmps: float = 0.0
    right_wheel_mmps: float = 0.0
    prox_mm: float = 1000.0
    prox_found: bool = False
    prox_lift_in_fov: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to compact JSON dictionary."""
        return {
            "timestamp_s": round(self.timestamp_s, 6),
            "pose": self.pose.to_dict(),
            "pitch_rad": round(self.pitch_rad, 5),
            "gyro_xyz_rad_s": [round(v, 5) for v in self.gyro_xyz_rad_s],
            "accel_xyz_m_s2": [round(v, 5) for v in self.accel_xyz_m_s2],
            "left_wheel_mmps": round(self.left_wheel_mmps, 2),
            "right_wheel_mmps": round(self.right_wheel_mmps, 2),
            "prox_mm": round(self.prox_mm, 1),
            "prox_found": self.prox_found,
            "prox_lift_in_fov": self.prox_lift_in_fov,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "InertialOdomSample":
        """Rebuild InertialOdomSample from JSON dict."""
        g = tuple(float(v) for v in raw.get("gyro_xyz_rad_s", [0.0, 0.0, 0.0])[:3])
        a = tuple(float(v) for v in raw.get("accel_xyz_m_s2", [0.0, 0.0, 9.81])[:3])
        return cls(
            timestamp_s=float(raw["timestamp_s"]),
            pose=Pose2D.from_dict(raw["pose"]),
            pitch_rad=float(raw.get("pitch_rad", 0.0)),
            gyro_xyz_rad_s=g,  # type: ignore[arg-type]
            accel_xyz_m_s2=a,  # type: ignore[arg-type]
            left_wheel_mmps=float(raw.get("left_wheel_mmps", 0.0)),
            right_wheel_mmps=float(raw.get("right_wheel_mmps", 0.0)),
            prox_mm=float(raw.get("prox_mm", 1000.0)),
            prox_found=bool(raw.get("prox_found", False)),
            prox_lift_in_fov=bool(raw.get("prox_lift_in_fov", False)),
        )


def _vec3_tuple(obj: Any, scale: float = 1.0, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    """Read Vector3-like attribute safely into scaled (x, y, z) float tuple."""
    if obj is None:
        return default
    x = float(getattr(obj, "x", default[0])) * scale
    y = float(getattr(obj, "y", default[1])) * scale
    z = float(getattr(obj, "z", default[2])) * scale
    return (x, y, z)


def sample_inertial_odom(robot: Any, timestamp_s: Optional[float] = None) -> InertialOdomSample:
    """Read pose, pitch, IMU (accel mm/s^2 -> m/s^2, gyro rad/s), wheel speeds, and ToF."""
    assert robot is not None
    ts = time.time() if timestamp_s is None else float(timestamp_s)
    pose = pose2d_from_sdk_pose(getattr(robot, "pose", None))
    pitch_obj = getattr(robot, "pose_pitch_rad", 0.0)
    pitch = float(getattr(pitch_obj, "radians", pitch_obj)) if pitch_obj is not None else 0.0
    gyro = _vec3_tuple(getattr(robot, "gyro", None), scale=1.0)
    accel = _vec3_tuple(getattr(robot, "accel", None), scale=0.001, default=(0.0, 0.0, 9810.0))
    prox = getattr(getattr(robot, "proximity", None), "last_sensor_reading", None)
    dist_obj = getattr(prox, "distance", None) if prox is not None else None
    prox_mm = float(getattr(dist_obj, "distance_mm", 1000.0)) if dist_obj is not None else 1000.0
    return InertialOdomSample(
        timestamp_s=ts,
        pose=pose,
        pitch_rad=pitch,
        gyro_xyz_rad_s=gyro,
        accel_xyz_m_s2=accel,
        left_wheel_mmps=float(getattr(robot, "left_wheel_speed_mmps", 0.0) or 0.0),
        right_wheel_mmps=float(getattr(robot, "right_wheel_speed_mmps", 0.0) or 0.0),
        prox_mm=prox_mm,
        prox_found=bool(getattr(prox, "found_object", False)) if prox is not None else False,
        prox_lift_in_fov=bool(getattr(prox, "is_lift_in_fov", False)) if prox is not None else False,
    )


def pil_to_gray_numpy(raw_image: Any) -> np.ndarray:
    """Convert PIL Image or 2D/3D numpy array into float32 grayscale array [0..255]."""
    assert raw_image is not None, "raw_image required"
    if isinstance(raw_image, np.ndarray):
        arr = raw_image.astype(np.float32)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        return arr[:, :, 0]
    if hasattr(raw_image, "convert"):
        return np.asarray(raw_image.convert("L"), dtype=np.float32)
    raise TypeError(f"Unsupported image type: {type(raw_image)!r}")


def _raycast_fused_cells(
    fused: Dict[Tuple[int, int], Any], cell_mm: float, pose: Pose2D, rel_ang: float, max_mm: float
) -> float:
    """Cast one bounded ray through canonical atlas cells, returning distance in meters."""
    if not fused:
        return math.inf
    ang = normalize_angle(pose.theta_rad + rel_ang)
    dx, dy = math.cos(ang), math.sin(ang)
    max_steps = min(MAX_RAY_STEPS, max(1, int(max_mm / cell_mm)))
    for step in range(1, max_steps + 1):
        d_mm = step * cell_mm
        ix = int(math.floor((pose.x_mm + d_mm * dx) / cell_mm))
        iy = int(math.floor((pose.y_mm + d_mm * dy) / cell_mm))
        st = fused.get((ix, iy))
        if st is not None and (st.log_odds >= OCC_THRESHOLD or st.cliff_hits > 0):
            return d_mm / 1000.0
    return math.inf


def synthesize_laserscan_from_tof_and_atlas(
    pose: Pose2D,
    prox_mm: float,
    prox_found: bool,
    atlas: Optional[NavMapAtlas] = None,
    max_range_m: float = DEFAULT_TOF_MAX_RANGE_M,
    fov_deg: float = DEFAULT_SCAN_FOV_DEG,
    num_rays: int = DEFAULT_SCAN_RAYS,
) -> Dict[str, Any]:
    """Synthesize a ROS 2 LaserScan dictionary from center ToF + persisted NavMapAtlas."""
    rays = max(3, min(MAX_SCAN_RAYS, int(num_rays)))
    fov_rad = math.radians(fov_deg)
    ang_min, ang_max = -0.5 * fov_rad, 0.5 * fov_rad
    inc = fov_rad / max(1, rays - 1)
    fused = atlas.fuse_canonical_cells() if atlas is not None else {}
    cell_mm = atlas.cell_mm if atlas is not None else 20.0
    ranges: List[float] = []
    intensities: List[float] = []
    for i in range(rays):
        rel = ang_min + i * inc
        is_center = abs(rel) <= 0.55 * inc
        if is_center and prox_found and 20.0 <= prox_mm <= max_range_m * 1000.0:
            ranges.append(round(prox_mm / 1000.0, 4))
            intensities.append(1.0)
        else:
            d_m = _raycast_fused_cells(fused, cell_mm, pose, rel, max_range_m * 1000.0)
            ranges.append(round(d_m, 4) if math.isfinite(d_m) and d_m >= 0.03 else math.inf)
            intensities.append(0.5 if math.isfinite(d_m) else 0.0)
    return {
        "frame_id": "base_scan",
        "angle_min": round(ang_min, 5),
        "angle_max": round(ang_max, 5),
        "angle_increment": round(inc, 5),
        "range_min": 0.03,
        "range_max": float(max_range_m),
        "ranges": ranges,
        "intensities": intensities,
    }


def _atomic_write_text(path: Path, body: str) -> None:
    """Write UTF-8 text atomically using sibling temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def _save_frame_png(path: Path, raw_image: Any) -> None:
    """Persist PIL Image or grayscale array as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw_image, np.ndarray):
        clipped = np.clip(raw_image, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(clipped).save(path)
    else:
        raw_image.save(path)


def _write_tum_imu_odom(
    out: Path, rel_paths: List[Tuple[float, str]], odom_samples: Sequence[InertialOdomSample]
) -> None:
    """Write TUM rgb.txt, TUM groundtruth_odom.txt, EuRoC imu.csv, and odom.csv."""
    rgb_lines = ["# timestamp filename"] + [f"{ts:.6f} {rel}" for ts, rel in rel_paths]
    _atomic_write_text(out / "rgb.txt", "\n".join(rgb_lines) + "\n")
    imu_hdr = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
    odom_hdr = "timestamp_s,x_m,y_m,theta_rad,left_wheel_mps,right_wheel_mps,pitch_rad,prox_m,prox_found"
    imu_rows, odom_rows, gt_rows = [imu_hdr], [odom_hdr], ["# timestamp tx ty tz qx qy qz qw"]
    for s in list(odom_samples)[:MAX_EXPORT_ODOM]:
        ns = int(round(s.timestamp_s * 1_000_000_000.0))
        gx, gy, gz = s.gyro_xyz_rad_s
        ax, ay, az = s.accel_xyz_m_s2
        imu_rows.append(f"{ns},{gx:.6f},{gy:.6f},{gz:.6f},{ax:.6f},{ay:.6f},{az:.6f}")
        xm, ym = s.pose.x_mm / 1000.0, s.pose.y_mm / 1000.0
        lv, rv = s.left_wheel_mmps / 1000.0, s.right_wheel_mmps / 1000.0
        odom_rows.append(f"{s.timestamp_s:.6f},{xm:.6f},{ym:.6f},{s.pose.theta_rad:.6f},{lv:.4f},{rv:.4f},{s.pitch_rad:.5f},{s.prox_mm/1000.0:.4f},{int(s.prox_found)}")
        qz, qw = math.sin(0.5 * s.pose.theta_rad), math.cos(0.5 * s.pose.theta_rad)
        gt_rows.append(f"{s.timestamp_s:.6f} {xm:.6f} {ym:.6f} 0.000000 0.000000 0.000000 {qz:.6f} {qw:.6f}")
    _atomic_write_text(out / "imu.csv", "\n".join(imu_rows) + "\n")
    _atomic_write_text(out / "odom.csv", "\n".join(odom_rows) + "\n")
    _atomic_write_text(out / "groundtruth_odom.txt", "\n".join(gt_rows) + "\n")


def _format_orb_slam3_yaml(intr: CameraIntrinsics, fps: float = 15.0) -> str:
    """Build ORB-SLAM3 Monocular-Inertial settings YAML matching Vector intrinsics."""
    k = intr.distortion_k
    return "\n".join([
        "%YAML:1.0",
        "Camera.type: \"PinHole\"",
        f"Camera1.fx: {intr.fx:.4f}",
        f"Camera1.fy: {intr.fy:.4f}",
        f"Camera1.cx: {intr.cx:.4f}",
        f"Camera1.cy: {intr.cy:.4f}",
        f"Camera1.k1: {k[0]:.6f}",
        f"Camera1.k2: {k[1]:.6f}",
        f"Camera1.p1: {k[2]:.6f}",
        f"Camera1.p2: {k[3]:.6f}",
        f"Camera.width: {intr.width}",
        f"Camera.height: {intr.height}",
        f"Camera.fps: {fps:.2f}",
        "Camera.RGB: 1",
        "IMU.NoiseGyro: 1.5e-2",
        "IMU.NoiseAcc: 1.5e-1",
        "IMU.GyroWalk: 1.0e-3",
        "IMU.AccWalk: 1.0e-2",
        "IMU.Frequency: 30.0",
        "Tbc: !!opencv-matrix",
        "   rows: 4",
        "   cols: 4",
        "   dt: f",
        "   data: [0.0, 0.0, 1.0, 0.035, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.015, 0.0, 0.0, 0.0, 1.0]",
        "ORBextractor.nFeatures: 800",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 6",
        "",
    ])


def _format_rtabmap_launch() -> str:
    """Build ROS 2 Python launch file wiring RTAB-Map monocular+odom+IMU topics."""
    return "\n".join([
        "from launch import LaunchDescription",
        "from launch_ros.actions import Node",
        "",
        "def generate_launch_description():",
        "    params = [{'frame_id': 'base_link', 'subscribe_depth': False, 'subscribe_rgb': True, 'subscribe_scan': True, 'approx_sync': True}]",
        "    remaps = [('rgb/image', '/vector/camera/image_raw'), ('rgb/camera_info', '/vector/camera/camera_info'), ('odom', '/vector/odom'), ('imu', '/vector/imu'), ('scan', '/vector/scan')]",
        "    return LaunchDescription([Node(package='rtabmap_slam', executable='rtabmap', output='screen', parameters=params, remappings=remaps)])",
        "",
    ])


def export_slam_dataset(
    output_dir: Path,
    frames: Sequence[Tuple[float, Any]],
    odom_samples: Sequence[InertialOdomSample],
    intrinsics: CameraIntrinsics,
    fps: float = 15.0,
) -> Dict[str, Any]:
    """Export TUM rgb.txt, EuRoC imu.csv, odom.csv, camera_info, ORB-SLAM3 YAML, and RTAB-Map launch."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rel_paths: List[Tuple[float, str]] = []
    for idx, (ts, raw_img) in enumerate(list(frames)[:MAX_EXPORT_FRAMES]):
        rel = f"frames/frame_{idx:06d}.png"
        _save_frame_png(out / rel, raw_img)
        rel_paths.append((float(ts), rel))
    _write_tum_imu_odom(out, rel_paths, odom_samples)
    _atomic_write_text(out / "camera_info.json", json.dumps(intrinsics.to_dict(), indent=2) + "\n")
    _atomic_write_text(out / "orb_slam3_vector.yaml", _format_orb_slam3_yaml(intrinsics, fps=fps))
    _atomic_write_text(out / "rtabmap_ros2_launch.py", _format_rtabmap_launch())
    manifest = {
        "schema_version": 2,
        "frame_count": len(rel_paths),
        "odom_count": min(len(odom_samples), MAX_EXPORT_ODOM),
        "width": intrinsics.width,
        "height": intrinsics.height,
        "files": ["rgb.txt", "imu.csv", "odom.csv", "groundtruth_odom.txt", "camera_info.json", "orb_slam3_vector.yaml", "rtabmap_ros2_launch.py"],
    }
    _atomic_write_text(out / "dataset_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_imu_by_ns(imu_path: Path) -> List[Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]]:
    """Parse EuRoC imu.csv into sorted timestamp + gyro + accel tuples."""
    if not imu_path.is_file():
        return []
    out = []
    for line in imu_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split(",")
        out.append((float(p[0]) / 1e9, (float(p[1]), float(p[2]), float(p[3])), (float(p[4]), float(p[5]), float(p[6]))))
    return out


def _nearest_imu(ts: float, imu_rows: Sequence[Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Find nearest IMU sample by timestamp or return stationary gravity default."""
    if not imu_rows:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 9.81))
    best = min(imu_rows, key=lambda r: abs(r[0] - ts))
    return (best[1], best[2])


def load_replay_dataset(
    dataset_dir: Path,
) -> Tuple[CameraIntrinsics, List[Tuple[float, np.ndarray]], List[InertialOdomSample]]:
    """Load camera_info.json, TUM rgb.txt PNG frames, odom.csv, and imu.csv for replay."""
    root = Path(dataset_dir)
    intr = CameraIntrinsics.from_dict(json.loads((root / "camera_info.json").read_text(encoding="utf-8")))
    frames: List[Tuple[float, np.ndarray]] = []
    for line in (root / "rgb.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        ts_s, rel = line.split(None, 1)
        with Image.open(root / rel) as im:
            frames.append((float(ts_s), pil_to_gray_numpy(im)))
    imu_rows = _load_imu_by_ns(root / "imu.csv")
    odoms: List[InertialOdomSample] = []
    for line in (root / "odom.csv").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("timestamp_s"):
            continue
        p = line.split(",")
        ts = float(p[0])
        gyro, accel = _nearest_imu(ts, imu_rows)
        pose = Pose2D(float(p[1]) * 1000.0, float(p[2]) * 1000.0, float(p[3]))
        odoms.append(InertialOdomSample(ts, pose, float(p[6]), gyro, accel, float(p[4]) * 1000.0, float(p[5]) * 1000.0, float(p[7]) * 1000.0, bool(int(p[8]))))
    return (intr, frames, odoms)


def check_ros2_environment() -> Dict[str, Any]:
    """Inspect local Python/CLI availability of ROS 2 stack without failing when absent."""
    mods = ["rclpy", "sensor_msgs", "nav_msgs", "geometry_msgs", "cv_bridge", "rtabmap_ros", "slam_toolbox"]
    mod_status = {m: (importlib.util.find_spec(m) is not None) for m in mods}
    return {
        "ros2_cli": shutil.which("ros2"),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "modules": mod_status,
        "rclpy_ready": mod_status["rclpy"],
        "offline_bridge_ready": True,
    }


class VectorRos2Bridge:
    """Publish camera, IMU, odom, and virtual LaserScan to rclpy or deterministic test bus."""

    def __init__(self, node_name: str = "vector_slam_bridge", use_rclpy: bool = False) -> None:
        self.node_name = node_name
        self.use_rclpy = use_rclpy
        self.published: List[Tuple[str, Dict[str, Any]]] = []
        self._node: Any = None
        if use_rclpy:
            self._init_rclpy()

    def _init_rclpy(self) -> None:
        """Initialize optional live ROS 2 node if rclpy is installed."""
        import rclpy  # type: ignore

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node(self.node_name)

    def publish_camera_frame(self, ts: float, intrinsics: CameraIntrinsics, gray: np.ndarray) -> None:
        """Emit /vector/camera/camera_info and /vector/camera/image_raw payloads."""
        self.published.append(("/vector/camera/camera_info", {"ts": ts, **intrinsics.to_dict()}))
        self.published.append(("/vector/camera/image_raw", {"ts": ts, "width": int(gray.shape[1]), "height": int(gray.shape[0]), "encoding": "mono8"}))

    def publish_imu(self, sample: InertialOdomSample) -> None:
        """Emit /vector/imu payload in ROS 2 Imu coordinate layout."""
        self.published.append(("/vector/imu", {"ts": sample.timestamp_s, "angular_velocity": list(sample.gyro_xyz_rad_s), "linear_acceleration": list(sample.accel_xyz_m_s2)}))

    def publish_odom(self, sample: InertialOdomSample) -> None:
        """Emit /vector/odom payload with SI pose and wheel speeds."""
        self.published.append(("/vector/odom", {"ts": sample.timestamp_s, "x_m": sample.pose.x_mm / 1000.0, "y_m": sample.pose.y_mm / 1000.0, "theta_rad": sample.pose.theta_rad}))

    def publish_laserscan(self, ts: float, scan: Dict[str, Any]) -> None:
        """Emit /vector/scan synthetic LaserScan payload."""
        self.published.append(("/vector/scan", {"ts": ts, **scan}))

    def shutdown(self) -> None:
        """Cleanly destroy rclpy node if live ROS 2 mode was initialized."""
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
