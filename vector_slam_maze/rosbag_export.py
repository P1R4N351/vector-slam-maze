"""ROS 2 bag (.db3 sqlite3 / .mcap) exporter via PyPI ternaris/rosbags (ToF-only LaserScan)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sensor_bridge import InertialOdomSample, _load_imu_by_ns, _nearest_imu
from .types_and_math import Pose2D

MAX_EXPORT_BAG_SAMPLES = 4096
DEFAULT_SCAN_RANGE_MAX_M = 1.5
DEFAULT_SCAN_RANGE_MIN_M = 0.03


def _require_rosbags() -> Tuple[Any, Any, Any, Any]:
    """Import rosbags writer and ROS2 Humble typestore on demand."""
    try:
        from rosbags.rosbag2 import StoragePlugin, Writer  # type: ignore
        from rosbags.typesys import Stores, get_typestore  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"rosbags_missing:{exc}") from exc
    ts = get_typestore(Stores.ROS2_HUMBLE)
    assert Writer is not None and ts is not None
    assert hasattr(StoragePlugin, "SQLITE3") and hasattr(StoragePlugin, "MCAP")
    return (Writer, StoragePlugin, ts, ts.types)


def _ros_time(ts_s: float, TimeCls: Any) -> Tuple[int, Any]:
    """Convert float POSIX seconds to (timestamp_ns, builtin_interfaces/msg/Time)."""
    assert math.isfinite(ts_s) and ts_s >= 0.0, f"invalid timestamp: {ts_s}"
    sec = int(math.floor(ts_s))
    nsec = int(round((ts_s - sec) * 1_000_000_000.0))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    assert 0 <= nsec < 1_000_000_000
    return (sec * 1_000_000_000 + nsec, TimeCls(sec=sec, nanosec=nsec))


def _build_odom_msg(s: InertialOdomSample, stamp: Any, types: Dict[str, Any]) -> Any:
    """Construct nav_msgs/msg/Odometry message from InertialOdomSample."""
    assert isinstance(s, InertialOdomSample) and stamp is not None
    Header = types["std_msgs/msg/Header"]
    Point, Quat = types["geometry_msgs/msg/Point"], types["geometry_msgs/msg/Quaternion"]
    Pose, PoseCov = types["geometry_msgs/msg/Pose"], types["geometry_msgs/msg/PoseWithCovariance"]
    Vec3, Twist = types["geometry_msgs/msg/Vector3"], types["geometry_msgs/msg/Twist"]
    TwistCov, Odom = types["geometry_msgs/msg/TwistWithCovariance"], types["nav_msgs/msg/Odometry"]
    half_th = 0.5 * float(s.pose.theta_rad)
    pose = Pose(Point(s.pose.x_mm / 1000.0, s.pose.y_mm / 1000.0, 0.0), Quat(0.0, 0.0, math.sin(half_th), math.cos(half_th)))
    v_lin = 0.5 * (s.left_wheel_mmps + s.right_wheel_mmps) / 1000.0
    twist = Twist(Vec3(v_lin, 0.0, 0.0), Vec3(0.0, 0.0, float(s.gyro_xyz_rad_s[2])))
    cov36 = np.zeros(36, dtype=np.float64)
    return Odom(Header(stamp=stamp, frame_id="odom"), "base_link", PoseCov(pose, cov36), TwistCov(twist, cov36))


def _build_imu_msg(s: InertialOdomSample, stamp: Any, types: Dict[str, Any]) -> Any:
    """Construct sensor_msgs/msg/Imu message from InertialOdomSample."""
    assert isinstance(s, InertialOdomSample) and len(s.gyro_xyz_rad_s) == 3
    Header = types["std_msgs/msg/Header"]
    Quat, Vec3, Imu = types["geometry_msgs/msg/Quaternion"], types["geometry_msgs/msg/Vector3"], types["sensor_msgs/msg/Imu"]
    half_th = 0.5 * float(s.pose.theta_rad)
    gx, gy, gz = (float(v) for v in s.gyro_xyz_rad_s)
    ax, ay, az = (float(v) for v in s.accel_xyz_m_s2)
    cov9 = np.zeros(9, dtype=np.float64)
    return Imu(Header(stamp=stamp, frame_id="imu_link"), Quat(0.0, 0.0, math.sin(half_th), math.cos(half_th)), cov9, Vec3(gx, gy, gz), cov9, Vec3(ax, ay, az), cov9)


def _build_tof_scan_msg(s: InertialOdomSample, stamp: Any, types: Dict[str, Any]) -> Any:
    """Construct ToF-only 1-beam sensor_msgs/msg/LaserScan without atlas raycasting."""
    assert isinstance(s, InertialOdomSample) and stamp is not None
    Header, Scan = types["std_msgs/msg/Header"], types["sensor_msgs/msg/LaserScan"]
    valid = bool(s.prox_found and not s.prox_lift_in_fov)
    tof_m = float(s.prox_mm) / 1000.0 if valid else float("inf")
    rng = np.array([tof_m], dtype=np.float32)
    inten = np.array([1.0 if valid else 0.0], dtype=np.float32)
    return Scan(Header(stamp=stamp, frame_id="base_scan"), 0.0, 0.0, 0.0, 0.0, 0.033, DEFAULT_SCAN_RANGE_MIN_M, DEFAULT_SCAN_RANGE_MAX_M, rng, inten)


def _write_bag_samples(
    writer: Any, ts: Any, samples: Sequence[InertialOdomSample]
) -> int:
    """Add connections and serialize bounded InertialOdomSample stream into Writer."""
    assert writer is not None and ts is not None
    c_odom = writer.add_connection("/vector/odom", "nav_msgs/msg/Odometry", typestore=ts)
    c_imu = writer.add_connection("/vector/imu", "sensor_msgs/msg/Imu", typestore=ts)
    c_scan = writer.add_connection("/vector/scan", "sensor_msgs/msg/LaserScan", typestore=ts)
    count = 0
    for s in list(samples)[:MAX_EXPORT_BAG_SAMPLES]:
        ns, stamp = _ros_time(float(s.timestamp_s), ts.types["builtin_interfaces/msg/Time"])
        writer.write(c_odom, ns, ts.serialize_cdr(_build_odom_msg(s, stamp, ts.types), "nav_msgs/msg/Odometry"))
        writer.write(c_imu, ns, ts.serialize_cdr(_build_imu_msg(s, stamp, ts.types), "sensor_msgs/msg/Imu"))
        writer.write(c_scan, ns, ts.serialize_cdr(_build_tof_scan_msg(s, stamp, ts.types), "sensor_msgs/msg/LaserScan"))
        count += 1
    assert count == min(len(samples), MAX_EXPORT_BAG_SAMPLES)
    return count


def write_ros2_bag(
    bag_dir: Path, samples: Sequence[InertialOdomSample], storage: str = "sqlite3"
) -> Dict[str, Any]:
    """Write ROS 2 bag containing /vector/odom, /vector/imu, and ToF-only /vector/scan."""
    assert samples, "at least one InertialOdomSample required"
    assert storage in ("sqlite3", "mcap"), f"unsupported storage: {storage}"
    Writer, StoragePlugin, ts, _types = _require_rosbags()
    target = Path(bag_dir)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    plugin = StoragePlugin.MCAP if storage == "mcap" else StoragePlugin.SQLITE3
    with Writer(target, version=9, storage_plugin=plugin) as writer:
        count = _write_bag_samples(writer, ts, samples)
    return {"ok": True, "bag_dir": str(target), "storage": storage, "sample_count": count, "message_count": count * 3, "topics": ["/vector/odom", "/vector/imu", "/vector/scan"]}


def load_odom_samples_from_dir(dataset_dir: Path) -> List[InertialOdomSample]:
    """Load InertialOdomSample sequence from dataset_dir odom.csv and optional imu.csv."""
    root = Path(dataset_dir)
    odom_path = root / "odom.csv"
    assert odom_path.is_file(), f"odom.csv missing in {root}"
    imu_rows = _load_imu_by_ns(root / "imu.csv")
    odoms: List[InertialOdomSample] = []
    for line in odom_path.read_text(encoding="utf-8").splitlines()[:MAX_EXPORT_BAG_SAMPLES]:
        if not line or line.startswith("timestamp_s") or line.startswith("#"):
            continue
        p = line.split(",")
        ts_s = float(p[0])
        gyro, accel = _nearest_imu(ts_s, imu_rows)
        pose = Pose2D(float(p[1]) * 1000.0, float(p[2]) * 1000.0, float(p[3]))
        odoms.append(InertialOdomSample(ts_s, pose, float(p[6]), gyro, accel, float(p[4]) * 1000.0, float(p[5]) * 1000.0, float(p[7]) * 1000.0, bool(int(p[8]))))
    return odoms


def load_odom_samples_from_json(json_path: Path) -> List[InertialOdomSample]:
    """Load InertialOdomSample sequence from serialized JSON list of sample dicts."""
    assert json_path.is_file(), f"samples JSON missing: {json_path}"
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    items = raw.get("samples", raw) if isinstance(raw, dict) else raw
    assert isinstance(items, list) and items, "non-empty sample list required"
    out: List[InertialOdomSample] = []
    for item in items[:MAX_EXPORT_BAG_SAMPLES]:
        pd = item["pose"]
        pose = Pose2D(float(pd["x_mm"]), float(pd["y_mm"]), float(pd["theta_rad"]), int(pd.get("origin_id", 0)))
        out.append(InertialOdomSample(float(item["timestamp_s"]), pose, float(item.get("pitch_rad", 0.0)), tuple(item.get("gyro_xyz_rad_s", [0.0, 0.0, 0.0])), tuple(item.get("accel_xyz_m_s2", [0.0, 0.0, 9.81])), float(item.get("left_wheel_mmps", 0.0)), float(item.get("right_wheel_mmps", 0.0)), float(item.get("prox_mm", 9999.0)), bool(item.get("prox_found", False)), bool(item.get("prox_lift_in_fov", False))))
    return out


def cmd_export_rosbag(args: argparse.Namespace) -> int:
    """CLI handler exporting dataset or samples JSON to ROS 2 sqlite3/mcap bag."""
    assert args.out_bag, "--out-bag required"
    try:
        if args.dataset:
            samples = load_odom_samples_from_dir(Path(args.dataset))
        elif args.samples_json:
            samples = load_odom_samples_from_json(Path(args.samples_json))
        else:
            print(json.dumps({"ok": False, "error": "dataset_or_samples_json_required"}))
            return 2
        res = write_ros2_bag(Path(args.out_bag), samples, storage=str(args.storage))
    except (AssertionError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(json.dumps(res, indent=2))
    return 0
