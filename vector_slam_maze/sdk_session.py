"""Python 3.11-safe anki_vector session, consent gate, and safety telemetry."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from .cliff_guard import CliffGuard, CliffHazard, GuardedRobot

DEFAULT_SERIAL = os.environ.get("VECTOR_SERIAL", "0dd1cafe")
MAX_NAV_POLL_STEPS = 200
MAX_CAM_POLL_STEPS = 120
MAX_MOTOR_RETRIES = 2
MIN_SAFE_BATTERY_VOLTS = 3.45
DEFAULT_MIN_PROX_MM = 65.0
MAGENTA_EYE_HUE = 0.83
MAGENTA_EYE_SAT = 0.95


def _import_anki_vector() -> Any:
    """Import anki_vector lazily so unit tests run without SDK hardware deps."""
    import anki_vector  # type: ignore

    return anki_vector


def check_safety(
    robot: Any,
    allow_on_charger: bool = False,
    min_prox_mm: float = DEFAULT_MIN_PROX_MM,
    check_front_prox: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Inspect cliff, pickup, charger, battery, and proximity sensors."""
    assert robot is not None, "robot required"
    status = getattr(robot, "status", None)
    batt = _safe_battery_state(robot)
    prox = getattr(robot, "proximity", None)
    last_prox = getattr(prox, "last_sensor_reading", None) if prox else None
    telemetry = _extract_telemetry(status, batt, last_prox, getattr(robot, "pose", None))
    reason = _evaluate_hazard(telemetry, allow_on_charger, min_prox_mm, check_front_prox)
    return (reason == "ok", reason, telemetry)


def _safe_battery_state(robot: Any) -> Any:
    """Read cached battery state without triggering blocking waits when unavailable."""
    catch_types = (OSError, RuntimeError, TimeoutError, AttributeError) + _sdk_exception_types()
    try:
        return robot.get_battery_state()
    except catch_types:
        return None


def _extract_telemetry(status: Any, batt: Any, last_prox: Any, pose: Any) -> Dict[str, Any]:
    """Build JSON-serializable safety telemetry snapshot."""
    dist_obj = getattr(last_prox, "distance", None) if last_prox is not None else None
    has_dist = dist_obj is not None and hasattr(dist_obj, "distance_mm")
    prox_mm = float(getattr(dist_obj, "distance_mm", 9999.0)) if has_dist else 9999.0
    prox_valid = bool(has_dist and math.isfinite(prox_mm) and prox_mm >= 0.0)
    assert math.isfinite(prox_mm) or not prox_valid
    pos = getattr(pose, "position", None)
    rot = getattr(pose, "rotation", None)
    ang = getattr(rot, "angle_z", None) if rot else None
    assert status is None or not isinstance(status, dict)
    return {
        "is_cliff_detected": bool(getattr(status, "is_cliff_detected", False)),
        "is_picked_up": bool(getattr(status, "is_picked_up", False)),
        "is_falling": bool(getattr(status, "is_falling", False)),
        "is_on_charger": bool(
            getattr(status, "is_on_charger", False)
            or getattr(status, "is_on_charger_platform", False)
        ),
        "is_in_calm_power_mode": bool(getattr(status, "is_in_calm_power_mode", False)),
        "is_charging": bool(getattr(status, "is_charging", False)),
        "battery_volts": round(float(getattr(batt, "battery_volts", 0.0)), 3),
        "battery_level": int(getattr(batt, "battery_level", -1)),
        "prox_mm": round(prox_mm, 1) if math.isfinite(prox_mm) else 9999.0,
        "prox_valid": prox_valid,
        "prox_found": bool(getattr(last_prox, "found_object", False)),
        "prox_lift_in_fov": bool(getattr(last_prox, "is_lift_in_fov", False)),
        "x_mm": round(float(getattr(pos, "x", 0.0)), 2) if pos else 0.0,
        "y_mm": round(float(getattr(pos, "y", 0.0)), 2) if pos else 0.0,
        "angle_deg": round(float(getattr(ang, "degrees", 0.0)), 2) if ang else 0.0,
        "origin_id": int(getattr(pose, "origin_id", -1)) if pose else -1,
    }


def _evaluate_hazard(
    t: Dict[str, Any], allow_on_charger: bool, min_prox_mm: float, check_front_prox: bool
) -> str:
    """Return 'ok' or concrete abort reason."""
    if t["is_picked_up"]:
        return "picked_up"
    if t["is_falling"]:
        return "falling"
    if t["is_cliff_detected"]:
        return "cliff"
    if not allow_on_charger and t["is_on_charger"]:
        return "on_charger"
    lvl, volts = t["battery_level"], t["battery_volts"]
    if not t["is_on_charger"] and ((lvl >= 0 and lvl <= 1) or (0.0 < volts < MIN_SAFE_BATTERY_VOLTS)):
        return "low_battery"
    if check_front_prox and t["prox_found"] and not t["prox_lift_in_fov"] and t["prox_mm"] < min_prox_mm:
        return "proximity_blocked"
    return "ok"


def poll_fresh_nav_map(
    robot: Any, max_wait_s: float = 2.5, step_s: float = 0.1, after_map: Any = None
) -> Tuple[Any, bool]:
    """Poll robot.nav_map._latest_nav_map without blocking; if after_map given, wait for a new broadcast."""
    assert robot is not None and max_wait_s >= 0.0 and step_s > 0.0
    max_iters = min(MAX_NAV_POLL_STEPS, max(1, int(round(max_wait_s / step_s)) + 1))
    nav = getattr(robot, "nav_map", None)
    if nav is None:
        return (None, False)
    cur = None
    for _ in range(max_iters):
        cur = getattr(nav, "_latest_nav_map", None)
        if cur is not None and (after_map is None or cur is not after_map):
            return (cur, True)
        time.sleep(step_s)
    cur = getattr(nav, "_latest_nav_map", None)
    return (cur, cur is not None and (after_map is None or cur is not after_map))


def poll_latest_nav_map(
    robot: Any, max_wait_s: float = 2.5, step_s: float = 0.1, after_map: Any = None
) -> Any:
    """Poll robot.nav_map._latest_nav_map without blocking on latest_nav_map."""
    grid, _fresh = poll_fresh_nav_map(robot, max_wait_s=max_wait_s, step_s=step_s, after_map=after_map)
    return grid


def _read_camera_image_nonblocking(cam: Any) -> Any:
    """Read cached CameraImage without triggering anki_vector blocking properties."""
    raw_dict = getattr(cam, "__dict__", {})
    if "_latest_image" in raw_dict or hasattr(type(cam), "_latest_image"):
        return getattr(cam, "_latest_image", None)
    catch_types = (AttributeError,) + _sdk_exception_types()
    try:
        return getattr(cam, "latest_image", None)
    except catch_types:
        return None


def poll_latest_camera_image(
    robot: Any, max_wait_s: float = 2.0, step_s: float = 0.05, min_image_id: int = -1
) -> Any:
    """Poll robot.camera for a CameraImage newer than min_image_id within bounded wait."""
    assert robot is not None and max_wait_s >= 0.0 and step_s > 0.0
    cam = getattr(robot, "camera", None)
    if cam is None:
        return None
    max_iters = min(MAX_CAM_POLL_STEPS, max(1, int(round(max_wait_s / step_s)) + 1))
    for _ in range(max_iters):
        cur = _read_camera_image_nonblocking(cam)
        img_id = int(getattr(cur, "image_id", -1)) if cur is not None else -1
        if cur is not None and img_id > min_image_id:
            return cur
        time.sleep(step_s)
    return _read_camera_image_nonblocking(cam)


def _sdk_exception_types() -> Tuple[type, ...]:
    """Return tuple containing anki_vector VectorException if SDK is importable."""
    try:
        from anki_vector.exceptions import VectorException  # type: ignore

        return (VectorException,)
    except ImportError:
        return ()


def enable_robot_image_streaming(robot: Any, enable: bool = True, timeout_s: float = 3.0) -> bool:
    """Explicitly toggle EnableImageStreaming via gRPC when a live connection exists."""
    assert robot is not None and 0.1 <= timeout_s <= 15.0
    conn = getattr(robot, "conn", None)
    grpc_if = getattr(conn, "grpc_interface", None)
    if grpc_if is None or not hasattr(grpc_if, "EnableImageStreaming") or not hasattr(conn, "run_coroutine"):
        return False
    try:
        from anki_vector.messaging import protocol  # type: ignore

        req = protocol.EnableImageStreamingRequest(enable=bool(enable), enable_high_resolution=False)
        res = conn.run_coroutine(grpc_if.EnableImageStreaming(req)).result(timeout=timeout_s)
        return res is not None
    except (concurrent.futures.TimeoutError, TimeoutError, OSError, RuntimeError) + _sdk_exception_types():
        return False


def capture_high_res_single_image(
    robot: Any, enable_high_res: bool = True, timeout_s: float = 6.0
) -> Any:
    """Capture a single CameraImage with bounded gRPC timeout when streaming feed is off."""
    assert robot is not None and 0.2 <= timeout_s <= 30.0
    cam = getattr(robot, "camera", None)
    if cam is None:
        return None
    catch_types = (concurrent.futures.TimeoutError, OSError, RuntimeError, TimeoutError) + _sdk_exception_types()
    conn = getattr(robot, "conn", None)
    grpc_if = getattr(conn, "grpc_interface", None)
    if grpc_if is not None and hasattr(grpc_if, "CaptureSingleImage") and hasattr(conn, "run_coroutine"):
        try:
            from PIL import Image
            from anki_vector.messaging import protocol  # type: ignore

            req = protocol.CaptureSingleImageRequest(enable_high_resolution=bool(enable_high_res))
            res = conn.run_coroutine(grpc_if.CaptureSingleImage(req)).result(timeout=timeout_s)
            raw_bytes = getattr(res, "data", b"") if res is not None else b""
            if not raw_bytes:
                return None
            with Image.open(io.BytesIO(raw_bytes)) as im:
                return SimpleNamespace(raw_image=im.copy(), image_id=int(getattr(res, "image_id", 0)))
        except catch_types:
            return None
    if not hasattr(cam, "capture_single_image"):
        return None
    try:
        return cam.capture_single_image(enable_high_resolution=enable_high_res)
    except catch_types:
        return None


def read_camera_exposure_state(robot: Any) -> Dict[str, Any]:
    """Read Vector exposure, gain, auto-exposure flag, and hardware ceiling bounds."""
    cam = getattr(robot, "camera", None)
    cfg = getattr(cam, "config", None) if cam is not None else None
    return {
        "exposure_ms": _opt_int(getattr(cam, "exposure_ms", None)),
        "gain": _opt_float(getattr(cam, "gain", None)),
        "auto_exposure": getattr(cam, "is_auto_exposure_enabled", None),
        "min_exposure_ms": _opt_int(getattr(cfg, "min_exposure_time_ms", None)),
        "max_exposure_ms": _opt_int(getattr(cfg, "max_exposure_time_ms", None)),
        "min_gain": _opt_float(getattr(cfg, "min_gain", None)),
        "max_gain": _opt_float(getattr(cfg, "max_gain", None)),
    }


def _opt_int(val: Any) -> Optional[int]:
    """Convert optional numeric value to int or None."""
    return int(val) if isinstance(val, (int, float)) else None


def _opt_float(val: Any) -> Optional[float]:
    """Convert optional numeric value to rounded float or None."""
    return round(float(val), 4) if isinstance(val, (int, float)) else None


def safe_disconnect(robot: Any, active: bool = False) -> None:
    """Cleanly stop active control and cancel async feed tasks on Python 3.11+."""
    if robot is None:
        return
    if active:
        _release_active_control(robot)
    _cancel_camera_task(robot)
    _cancel_nav_map_task(robot)
    catch_types = (concurrent.futures.TimeoutError, OSError, RuntimeError, TimeoutError) + _sdk_exception_types()
    try:
        robot.disconnect()
    except catch_types as exc:
        print(f"safe_disconnect warning: {exc}", file=sys.stderr)
        _force_close_transport(robot, catch_types)


def _force_close_transport(robot: Any, catch_types: Tuple[type, ...]) -> None:
    """Close events and conn transports if disconnect aborted mid-teardown."""
    for attr in ("events", "conn"):
        comp = getattr(robot, attr, None)
        close_fn = getattr(comp, "close", None) if comp is not None else None
        if close_fn is None:
            continue
        try:
            close_fn()
        except catch_types:
            pass


def _cancel_camera_task(robot: Any) -> None:
    """Stop camera stream and cancel _camera_feed_task without 10s close_camera_feed spin."""
    cam = getattr(robot, "camera", None)
    if cam is None:
        return
    if hasattr(cam, "_enabled"):
        cam._enabled = False
    task = getattr(cam, "_camera_feed_task", None)
    if task is None:
        return
    try:
        task.cancel()
        conn = getattr(robot, "conn", None)
        if conn is not None and hasattr(conn, "run_coroutine"):
            conn.run_coroutine(task).result(timeout=2.0)
    except (concurrent.futures.CancelledError, asyncio.CancelledError, TimeoutError, RuntimeError):
        pass
    cam._camera_feed_task = None


def _release_active_control(robot: Any) -> None:
    """Stop motors and yield behavior control before socket teardown."""
    catch_types = (concurrent.futures.TimeoutError, OSError, RuntimeError, TimeoutError) + _sdk_exception_types()
    try:
        motors = getattr(robot, "motors", None)
        if motors is not None:
            res = motors.stop_all_motors()
            if hasattr(res, "result"):
                res.result(timeout=2.0)
    except catch_types:
        pass
    try:
        conn = getattr(robot, "conn", None)
        if conn is not None and hasattr(conn, "release_control"):
            fut = conn.release_control()
            if hasattr(fut, "result"):
                fut.result(timeout=2.0)
    except catch_types:
        pass


def _cancel_nav_map_task(robot: Any) -> None:
    """Pre-cancel anki_vector NavMap feed task so disconnect does not crash Py3.11."""
    nav = getattr(robot, "nav_map", None)
    task = getattr(nav, "_nav_map_feed_task", None) if nav is not None else None
    if task is None:
        return
    try:
        task.cancel()
        conn = getattr(robot, "conn", None)
        if conn is not None and hasattr(conn, "run_coroutine"):
            conn.run_coroutine(task).result(timeout=2.0)
    except (concurrent.futures.CancelledError, asyncio.CancelledError, TimeoutError, RuntimeError):
        pass
    nav._nav_map_feed_task = None


class VectorSDKSession:
    """Consent-gated SDK session handling lockfile, eye envelope, and clean teardown."""

    def __init__(
        self,
        serial: str = DEFAULT_SERIAL,
        active: bool = False,
        via_vector: bool = False,
        enable_nav_map: bool = True,
        nav_freq_hz: float = 2.0,
        enable_camera: bool = False,
        ip: Optional[str] = None,
        lock_path: Optional[str] = None,
    ) -> None:
        assert serial, "Vector serial required"
        if active and not via_vector:
            raise PermissionError("Active robot motion denied: --via-vector consent flag is required")
        self.serial = serial
        self.active = active
        self.via_vector = via_vector
        self.enable_nav_map = enable_nav_map
        self.nav_freq_hz = nav_freq_hz
        self.enable_camera = enable_camera
        self.ip = ip
        self.lock_path = lock_path or os.environ.get(
            "VECTOR_TAKEOVER_LOCK", f"/tmp/vector-{serial}-takeover.lock"
        )
        self.robot: Any = None
        self._lock_acquired = False

    def __enter__(self) -> Any:
        if self.active:
            self._acquire_lock()
        try:
            return self._open()
        except BaseException:
            self._abandon()
            raise

    def _abandon(self) -> None:
        """Undo a failed __enter__: drop any connection made and release the takeover lock.

        Without this a failed connect left the lock file behind, and every
        later session for that robot was refused until someone removed it.
        """
        robot, self.robot = self.robot, None
        try:
            safe_disconnect(robot, active=self.active)
        except Exception as exc:  # noqa: BLE001 - the failure that brought us here is the one to report
            print(f"session abandon warning: {type(exc).__name__}", file=sys.stderr)
        finally:
            if self._lock_acquired:
                self._release_lock()

    def _open(self) -> Any:
        """Connect, start the feeds and, for an active session, return the guarded robot."""
        av = _import_anki_vector()
        # DEFAULT, not OVERRIDE_BEHAVIORS: the override level suppresses the
        # robot's cliff reaction ("will drive off table"). See cliff_guard.
        ctrl = av.connection.ControlPriorityLevel.DEFAULT_PRIORITY if self.active else None
        kwargs: Dict[str, Any] = {
            "serial": self.serial,
            "behavior_control_level": ctrl,
            # Start nav_map disabled so post-connect init_nav_map_feed(frequency=...) runs on loop.
            "enable_nav_map_feed": False,
        }
        if self.ip:
            kwargs["ip"] = self.ip
        self.robot = av.Robot(**kwargs)
        self.robot.connect()
        if self.active:
            self._require_control()
        self._start_async_feeds()
        if not self.active:
            return self.robot
        self._set_takeover_eyes()
        guard = CliffGuard(self.robot, ctrl)
        guard.attach(av.events.Events.robot_state)
        return GuardedRobot(self.robot, guard)

    def _require_control(self) -> None:
        """Refuse an active session the robot has not given control to.

        At DEFAULT priority Vector does not grant control while its cliff flag
        is set and it is parked at an edge. connect() then returns without
        control and without raising, and the first behaviour call blocks: on
        2026-11-15 drive_on_charger blocked for 314 s with the robot
        motionless. Fail here, before any call can block.
        """
        granted = getattr(getattr(self.robot, "conn", None), "control_granted_event", None)
        if granted is None or granted.is_set():
            return
        cliff = bool(getattr(getattr(self.robot, "status", None), "is_cliff_detected", False))
        why = " while its cliff flag is set; move it away from the edge by hand" if cliff else ""
        raise CliffHazard(
            "session refused: the robot did not grant control at DEFAULT priority" + why,
            hazard="cliff_detected" if cliff else "control_refused",
            refused=True,
        )

    def _start_async_feeds(self) -> None:
        """Schedule NavMap and camera feed tasks on the robot event loop."""
        loop = getattr(getattr(self.robot, "conn", None), "loop", None)
        if self.enable_nav_map and hasattr(getattr(self.robot, "nav_map", None), "init_nav_map_feed"):
            fn = lambda: self.robot.nav_map.init_nav_map_feed(frequency=self.nav_freq_hz)
            loop.call_soon_threadsafe(fn) if (loop and hasattr(loop, "call_soon_threadsafe")) else fn()
        if self.enable_camera and hasattr(getattr(self.robot, "camera", None), "init_camera_feed"):
            enable_robot_image_streaming(self.robot, True)
            cam_fn = self.robot.camera.init_camera_feed
            loop.call_soon_threadsafe(cam_fn) if (loop and hasattr(loop, "call_soon_threadsafe")) else cam_fn()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            safe_disconnect(self.robot, active=self.active)
        finally:
            self.robot = None
            if self._lock_acquired:
                self._release_lock()

    def _acquire_lock(self) -> None:
        """Create non-blocking exclusive takeover lockfile."""
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd = os.open(self.lock_path, flags, 0o644)
        try:
            payload = f"pid={os.getpid()} serial={self.serial} ts={int(time.time())}\n"
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        self._lock_acquired = True

    def _release_lock(self) -> None:
        """Remove takeover lockfile if held by this session."""
        try:
            os.unlink(self.lock_path)
        except FileNotFoundError:
            pass
        self._lock_acquired = False

    def _set_takeover_eyes(self) -> None:
        """Signal active Collective control with magenta eye color (best-effort)."""
        try:
            self.robot.behavior.set_eye_color(hue=MAGENTA_EYE_HUE, saturation=MAGENTA_EYE_SAT)
        except (OSError, RuntimeError, TimeoutError, AttributeError):
            pass
