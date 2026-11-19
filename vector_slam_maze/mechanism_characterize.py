"""Empirical physical testing and system identification for all 4 Vector articulating mechanisms.

Covers left_wheel, right_wheel, head, and lift:
  - Transport latency (command -> first physical response in telemetry)
  - Closed-loop PD/PID tracking (rise time, settling time, overshoot, steady-state error)
  - Armature/inertia lag and active dynamic braking coast distance
  - Static Coulomb stiction holding under gravity / sub-breakaway rest
  - Viscous damping / tread turn scrub attenuation
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cliff_guard import CliffHazard, hazard_of
from .movement_directives import _sdk_units, check_safety

MAX_STATE_SAMPLES = 4096
MAX_POLL_STEPS = 400
MIN_SAFE_FLOOR_TOF_MM = 220.0

MIN_LATENCY_MS = 5.0
MIN_RISE_10_90_MS = 10.0
MAX_STEP_OVERSHOOT_PCT = 25.0
MIN_JOINT_INTERIOR_SAMPLES = 2
MIN_WHEEL_BRAKE_MM = 2.0
MAX_WHEEL_BRAKE_MM = 22.0
HEAD_SETTLE_TOL_DEG = 3.0
LIFT_SETTLE_TOL_MM = 3.5
WHEEL_SPEED_TOL_MMPS = 10.0
MAX_HOLD_DRIFT_DEG = 1.5
MAX_HOLD_DRIFT_MM = 2.0


def _sample_from_msg(t_s: float, msg: Any) -> Dict[str, Any]:
    """Extract normalized scalar kinematics and status flags from RobotState protobuf."""
    assert math.isfinite(t_s) and msg is not None
    pose = getattr(msg, "pose", None)
    status = int(getattr(msg, "status", 0) or 0)
    return {
        "t_s": t_s,
        "x_mm": float(getattr(pose, "x", 0.0) or 0.0),
        "y_mm": float(getattr(pose, "y", 0.0) or 0.0),
        "theta_rad": float(getattr(msg, "pose_angle_rad", 0.0) or 0.0),
        "head_deg": math.degrees(float(getattr(msg, "head_angle_rad", 0.0) or 0.0)),
        "lift_mm": float(getattr(msg, "lift_height_mm", 32.0) or 32.0),
        "lw_mmps": float(getattr(msg, "left_wheel_speed_mmps", 0.0) or 0.0),
        "rw_mmps": float(getattr(msg, "right_wheel_speed_mmps", 0.0) or 0.0),
        "status": status,
        "cliff": bool(status & 0x4000),
        "picked_up": bool(status & 0x8),
        "falling": bool(status & 0x20),
        "on_charger": bool(status & 0x1000),
        "head_in_pos": bool(status & 0x200),
        "lift_in_pos": bool(status & 0x100),
    }


class RobotStateRecorder:
    """Thread-safe bounded recorder for high-rate anki_vector robot_state events."""

    def __init__(self, max_samples: int = MAX_STATE_SAMPLES) -> None:
        assert 16 <= max_samples <= 65536
        self._max = int(max_samples)
        self._lock = threading.Lock()
        self._samples: List[Dict[str, Any]] = []
        self._attached = False

    def on_state(self, _robot: Any, _event_type: Any, msg: Any) -> None:
        """Record incoming robot_state event with monotonic wall clock."""
        if msg is None:
            return
        row = _sample_from_msg(time.monotonic(), msg)
        with self._lock:
            if len(self._samples) >= self._max:
                del self._samples[: len(self._samples) - self._max + 1]
            self._samples.append(row)

    def attach(self, robot: Any) -> bool:
        """Subscribe recorder to robot_state events."""
        assert robot is not None
        try:
            from anki_vector.events import Events  # type: ignore

            ev = Events.robot_state
        except ImportError:
            ev = "robot_state"
        events = getattr(robot, "events", None)
        if events is None or not hasattr(events, "subscribe"):
            return False
        events.subscribe(self.on_state, ev)
        self._attached = True
        return True

    def detach(self, robot: Any) -> None:
        """Best-effort unsubscribe from robot_state events."""
        if not self._attached or robot is None:
            return
        try:
            from anki_vector.events import Events  # type: ignore

            ev = Events.robot_state
        except ImportError:
            ev = "robot_state"
        events = getattr(robot, "events", None)
        if events is not None and hasattr(events, "unsubscribe"):
            try:
                events.unsubscribe(self.on_state, ev)
            except Exception:  # noqa: BLE001 - best-effort detach on teardown
                pass
        self._attached = False

    def snapshot(self, since_t: float = 0.0, until_t: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return bounded copy of recorded state rows in [since_t, until_t]."""
        with self._lock:
            return [
                dict(row)
                for row in self._samples
                if row["t_s"] >= since_t and (until_t is None or row["t_s"] <= until_t)
            ]


def _first_crossing_ms(
    samples: Sequence[Dict[str, Any]], field: str, t_cmd_s: float, y_start: float, target_delta: float, threshold: float
) -> Optional[float]:
    """Interpolate elapsed ms after t_cmd_s where signed delta reaches threshold."""
    assert threshold >= 0.0 and isinstance(field, str)
    sgn = 1.0 if target_delta >= 0.0 else -1.0
    prev_t: Optional[float] = None
    prev_d = 0.0
    for row in samples:
        t = float(row["t_s"])
        d = sgn * (float(row[field]) - y_start)
        if t >= t_cmd_s and d >= threshold:
            if prev_t is not None and d > prev_d:
                frac = (threshold - prev_d) / (d - prev_d)
                t_cross = prev_t + max(0.0, min(1.0, frac)) * (t - prev_t)
            else:
                t_cross = t
            return max(0.0, (t_cross - t_cmd_s) * 1000.0)
        prev_t, prev_d = t, d
    return None


def _count_interior(samples: Sequence[Dict[str, Any]], field: str, y_start: float, delta: float) -> int:
    """Count samples lying strictly inside (10%, 90%) of step magnitude."""
    mag = abs(delta)
    if mag <= 1e-6:
        return 0
    sgn = 1.0 if delta >= 0.0 else -1.0
    count = 0
    for row in samples:
        prog = sgn * (float(row[field]) - y_start) / mag
        if 0.10 < prog < 0.90:
            count += 1
    return count


def fit_step_response(
    samples: Sequence[Dict[str, Any]],
    field: str,
    t_cmd_s: float,
    y_start: float,
    y_target: float,
    noise_eps: float,
    settle_tol: float,
) -> Dict[str, Any]:
    """Extract latency, rise time, tau_eff, overshoot, settling time, and SSE."""
    assert math.isfinite(y_start) and math.isfinite(y_target) and settle_tol > 0.0
    win = [r for r in samples if float(r["t_s"]) >= t_cmd_s]
    delta = y_target - y_start
    mag = abs(delta)
    if len(win) < 2 or mag <= 1e-6:
        return {"sample_count": len(win), "ok": False, "reason": "insufficient_samples"}
    lat_th = max(noise_eps, 0.05 * mag)
    lat_ms = _first_crossing_ms(win, field, t_cmd_s, y_start, delta, lat_th)
    t10 = _first_crossing_ms(win, field, t_cmd_s, y_start, delta, 0.10 * mag)
    t63 = _first_crossing_ms(win, field, t_cmd_s, y_start, delta, 0.632 * mag)
    t90 = _first_crossing_ms(win, field, t_cmd_s, y_start, delta, 0.90 * mag)
    sgn = 1.0 if delta >= 0.0 else -1.0
    peak_prog = max(sgn * (float(r[field]) - y_start) for r in win)
    overshoot_pct = max(0.0, (peak_prog - mag) / mag * 100.0)
    tail = win[-min(4, len(win)) :]
    y_ss = sum(float(r[field]) for r in tail) / float(len(tail))
    ss_err = abs(y_target - y_ss)
    last_out = max((float(r["t_s"]) for r in win if abs(float(r[field]) - y_target) > settle_tol), default=None)
    settle_ms = round((last_out - t_cmd_s) * 1000.0, 2) if (last_out is not None and ss_err <= settle_tol) else (0.0 if ss_err <= settle_tol else None)
    return {
        "ok": True,
        "sample_count": len(win),
        "interior_samples": _count_interior(win, field, y_start, delta),
        "latency_ms": round(lat_ms, 2) if lat_ms is not None else None,
        "rise_10_90_ms": round(t90 - t10, 2) if (t10 is not None and t90 is not None) else None,
        "tau_eff_ms": round(max(0.0, t63 - (lat_ms or 0.0)), 2) if t63 is not None else None,
        "overshoot_pct": round(overshoot_pct, 2),
        "steady_state_value": round(y_ss, 3),
        "steady_state_error": round(ss_err, 3),
        "settling_time_ms": settle_ms,
    }


def fit_braking_response(
    samples: Sequence[Dict[str, Any]],
    speed_field: str,
    t_stop_s: float,
    v_before_mmps: float,
    single_wheel: bool = True,
) -> Dict[str, Any]:
    """Measure stop command latency, braking duration, and net post-stop braking mm."""
    assert math.isfinite(t_stop_s) and speed_field in ("lw_mmps", "rw_mmps")
    pre = [r for r in samples if float(r["t_s"]) <= t_stop_s]
    win = [r for r in samples if float(r["t_s"]) >= t_stop_s]
    if len(win) < 2 or abs(v_before_mmps) < 10.0:
        return {"ok": False, "sample_count": len(win), "braking_dist_mm": 0.0}
    thresh_drop = 0.90 * abs(v_before_mmps)
    lat_ms: Optional[float] = None
    brake_ms: Optional[float] = None
    vel_coast_mm, pose_path_mm = 0.0, 0.0
    for idx, row in enumerate(win):
        t_cur = float(row["t_s"])
        v_cur = abs(float(row[speed_field]))
        if lat_ms is None and v_cur <= thresh_drop:
            lat_ms = max(0.0, (t_cur - t_stop_s) * 1000.0)
        if brake_ms is None and v_cur <= 3.0 and (t_cur - t_stop_s) >= 0.005:
            brake_ms = max(0.0, (t_cur - t_stop_s) * 1000.0)
        if idx > 0:
            prev = win[idx - 1]
            dt = max(0.0, t_cur - float(prev["t_s"]))
            v_prev = abs(float(prev[speed_field]))
            vel_coast_mm += 0.5 * (v_prev + v_cur) * dt
            pose_path_mm += math.hypot(float(row["x_mm"]) - float(prev["x_mm"]), float(row["y_mm"]) - float(prev["y_mm"]))
            if v_prev <= 2.0 and v_cur <= 2.0:
                break
    scale = 2.0 if single_wheel else 1.0
    brake_mm = vel_coast_mm
    if pre and pose_path_mm > 0.05:
        p0 = pre[-1]
        lead_mm = math.hypot(float(win[0]["x_mm"]) - float(p0["x_mm"]), float(win[0]["y_mm"]) - float(p0["y_mm"]))
        pre_dt = max(0.0, t_stop_s - float(p0["t_s"]))
        v_meas_pre = abs(float(p0[speed_field])) or abs(v_before_mmps)
        brake_mm = max(0.0, scale * (lead_mm + pose_path_mm) - v_meas_pre * pre_dt)
    return {
        "ok": True,
        "sample_count": len(win),
        "stop_latency_ms": round(lat_ms, 2) if lat_ms is not None else None,
        "braking_time_ms": round(brake_ms, 2) if brake_ms is not None else None,
        "braking_dist_mm": round(brake_mm, 3),
    }


def _check_joint_realism(name: str, m: Dict[str, Any], tol: float, hold_key: str, hold_lim: float, failures: List[str]) -> None:
    """Validate non-ideal latency, finite ramp, PD tracking, and stiction hold on head or lift."""
    assert name in ("head", "lift") and tol > 0.0
    up = m.get("step_up") or {}
    if not up.get("ok"):
        failures.append(f"{name}:missing_step_up")
        return
    if (up.get("latency_ms") or 0.0) < MIN_LATENCY_MS:
        failures.append(f"{name}:zero_or_low_latency({up.get('latency_ms')})")
    if (up.get("rise_10_90_ms") or 0.0) < MIN_RISE_10_90_MS or int(up.get("interior_samples", 0)) < MIN_JOINT_INTERIOR_SAMPLES:
        failures.append(f"{name}:ideal_snap_no_dynamics(rise={up.get('rise_10_90_ms')},mid={up.get('interior_samples')})")
    if (up.get("steady_state_error") or 999.0) > tol:
        failures.append(f"{name}:pd_steady_error({up.get('steady_state_error')}>{tol})")
    if (up.get("overshoot_pct") or 0.0) > MAX_STEP_OVERSHOOT_PCT:
        failures.append(f"{name}:excess_overshoot({up.get('overshoot_pct')})")
    if float(m.get(hold_key, 999.0)) > hold_lim:
        failures.append(f"{name}:gravity_sag_no_stiction({m.get(hold_key)})")


def _check_wheel_realism(name: str, m: Dict[str, Any], failures: List[str]) -> None:
    """Validate non-ideal latency, closed-loop tracking, and finite braking tail on a tread."""
    assert name in ("left_wheel", "right_wheel")
    fwd = m.get("step_fwd") or {}
    brk = m.get("brake_high") or {}
    if not fwd.get("ok") or not brk.get("ok"):
        failures.append(f"{name}:missing_step_or_brake")
        return
    if (fwd.get("latency_ms") or 0.0) < MIN_LATENCY_MS:
        failures.append(f"{name}:zero_or_low_latency({fwd.get('latency_ms')})")
    if (fwd.get("steady_state_error") or 999.0) > WHEEL_SPEED_TOL_MMPS:
        failures.append(f"{name}:pd_speed_error({fwd.get('steady_state_error')})")
    dist = float(brk.get("braking_dist_mm") or 0.0)
    if dist < MIN_WHEEL_BRAKE_MM or dist > MAX_WHEEL_BRAKE_MM:
        failures.append(f"{name}:unrealistic_brake_dist({dist})")


def evaluate_physical_realism(report: Dict[str, Any], include_wheels: bool) -> Dict[str, Any]:
    """Judge whether measured mechanisms exhibit non-ideal latency, PD, armature, friction, damping."""
    assert isinstance(report, dict) and isinstance(include_wheels, bool)
    mechs = report.get("mechanisms") or {}
    failures: List[str] = []
    checked = ["head", "lift"]
    _check_joint_realism("head", mechs.get("head") or {}, HEAD_SETTLE_TOL_DEG, "hold_drift_deg", MAX_HOLD_DRIFT_DEG, failures)
    _check_joint_realism("lift", mechs.get("lift") or {}, LIFT_SETTLE_TOL_MM, "hold_drift_mm", MAX_HOLD_DRIFT_MM, failures)
    if include_wheels:
        checked.extend(["left_wheel", "right_wheel"])
        _check_wheel_realism("left_wheel", mechs.get("left_wheel") or {}, failures)
        _check_wheel_realism("right_wheel", mechs.get("right_wheel") or {}, failures)
        chs = report.get("chassis") or {}
        if abs(float(chs.get("drive_err_mm", 99.0))) > 5.0 or abs(float(chs.get("turn_err_deg", 99.0))) > 5.0:
            failures.append(f"chassis:closed_loop_accuracy(d={chs.get('drive_err_mm')},th={chs.get('turn_err_deg')})")
    return {"realistic": len(failures) == 0, "failures": failures, "checked_mechanisms": checked}


def check_wheel_bench_safety(
    robot: Any, confirm_safe_floor: bool, allow_occluded_tof: bool = False
) -> Tuple[bool, str, Dict[str, Any]]:
    """Fail-closed safety gate before commanding off-charger tread characterization."""
    assert robot is not None and isinstance(confirm_safe_floor, bool)
    if not confirm_safe_floor:
        _, _, tel = check_safety(robot, allow_on_charger=True, check_front_prox=False)
        return (False, "confirm_safe_floor_required", tel)
    ok, reason, tel = check_safety(robot, allow_on_charger=False, min_prox_mm=MIN_SAFE_FLOOR_TOF_MM, check_front_prox=True)
    if not ok:
        return (False, reason, tel)
    if not bool(tel.get("prox_valid", False)):
        return (False, "prox_unavailable", tel)
    if bool(tel.get("prox_lift_in_fov", False)) and not allow_occluded_tof:
        return (False, "prox_lift_in_fov", tel)
    return (True, "ok", tel)


def _watch_sleep(robot: Any, dur_s: float, watch_hazards: bool) -> None:
    """Sleep in ~20 ms slices up to MAX_POLL_STEPS, aborting immediately on cliff/pickup/fall."""
    assert dur_s >= 0.0
    steps = min(MAX_POLL_STEPS, max(1, int(math.ceil(dur_s / 0.02))))
    slice_s = dur_s / float(steps)
    for _ in range(steps):
        if watch_hazards:
            st = getattr(robot, "status", None)
            raw_bits = int(getattr(st, "_status", 0) or 0) if st is not None else 0
            h = hazard_of(raw_bits)
            if h is not None:
                try:
                    robot.motors.stop_all_motors()
                except Exception:  # noqa: BLE001
                    pass
                raise CliffHazard(f"mechanism_bench hazard: {h}", hazard=h)
        time.sleep(slice_s)


def _measure_head(robot: Any, rec: RobotStateRecorder) -> Dict[str, Any]:
    """Run docked-safe head angle step tests and unpowered gravity-hold verification."""
    degrees_fn, _, _ = _sdk_units()
    robot.behavior.set_head_angle(degrees_fn(-20.0))
    _watch_sleep(robot, 0.25, False)
    y0 = math.degrees(float(getattr(robot, "head_angle_rad", math.radians(-20.0)) or math.radians(-20.0)))
    t_up = time.monotonic()
    robot.behavior.set_head_angle(degrees_fn(35.0))
    _watch_sleep(robot, 0.25, False)
    up_fit = fit_step_response(rec.snapshot(t_up), "head_deg", t_up, y0, 35.0, 0.5, HEAD_SETTLE_TOL_DEG)
    y1 = up_fit.get("steady_state_value", 35.0) or 35.0
    t_dn = time.monotonic()
    robot.behavior.set_head_angle(degrees_fn(-15.0))
    _watch_sleep(robot, 0.25, False)
    dn_fit = fit_step_response(rec.snapshot(t_dn), "head_deg", t_dn, float(y1), -15.0, 0.5, HEAD_SETTLE_TOL_DEG)
    robot.behavior.set_head_angle(degrees_fn(0.0))
    robot.motors.set_head_motor(0.0)
    h0 = math.degrees(float(getattr(robot, "head_angle_rad", 0.0) or 0.0))
    _watch_sleep(robot, 0.30, False)
    h1 = math.degrees(float(getattr(robot, "head_angle_rad", 0.0) or 0.0))
    return {"step_up": up_fit, "step_down": dn_fit, "hold_drift_deg": round(abs(h1 - h0), 3)}


def _measure_lift(robot: Any, rec: RobotStateRecorder) -> Dict[str, Any]:
    """Run docked-safe lift height step tests and gravity-hold verification."""
    robot.behavior.set_lift_height(0.0)
    _watch_sleep(robot, 0.20, False)
    y0 = float(getattr(robot, "lift_height_mm", 32.0) or 32.0)
    t_up = time.monotonic()
    robot.behavior.set_lift_height(0.85)
    _watch_sleep(robot, 0.20, False)
    up_fit = fit_step_response(rec.snapshot(t_up), "lift_mm", t_up, y0, 83.0, 0.6, LIFT_SETTLE_TOL_MM)
    y1 = up_fit.get("steady_state_value", 83.0) or 83.0
    t_dn = time.monotonic()
    robot.behavior.set_lift_height(0.5)
    _watch_sleep(robot, 0.20, False)
    dn_fit = fit_step_response(rec.snapshot(t_dn), "lift_mm", t_dn, float(y1), 62.0, 0.6, LIFT_SETTLE_TOL_MM)
    robot.behavior.set_lift_height(0.0)
    robot.motors.set_lift_motor(0.0)
    l0 = float(getattr(robot, "lift_height_mm", 32.0) or 32.0)
    _watch_sleep(robot, 0.25, False)
    l1 = float(getattr(robot, "lift_height_mm", 32.0) or 32.0)
    return {"step_up": up_fit, "step_down": dn_fit, "hold_drift_mm": round(abs(l1 - l0), 3)}


def _measure_wheel(robot: Any, rec: RobotStateRecorder, side: str) -> Dict[str, Any]:
    """Run bounded single-tread speed step, reverse return, and high-speed braking test."""
    assert side in ("left", "right")
    field = "lw_mmps" if side == "left" else "rw_mmps"
    cmd_100 = (100.0, 0.0) if side == "left" else (0.0, 100.0)
    cmd_hi = (160.0, 0.0) if side == "left" else (0.0, 160.0)
    t_fwd = time.monotonic()
    robot.motors.set_wheel_motors(*cmd_100)
    _watch_sleep(robot, 0.45, True)
    t_fwd_end = time.monotonic()
    robot.motors.stop_all_motors()
    _watch_sleep(robot, 0.20, True)
    fwd_fit = fit_step_response(rec.snapshot(t_fwd, t_fwd_end), field, t_fwd, 0.0, 100.0, 4.0, WHEEL_SPEED_TOL_MMPS)
    robot.motors.set_wheel_motors(*cmd_hi)
    _watch_sleep(robot, 0.45, True)
    t_stop = time.monotonic()
    robot.motors.stop_all_motors()
    _watch_sleep(robot, 0.28, True)
    t_brk_end = time.monotonic()
    brk_fit = fit_braking_response(rec.snapshot(t_stop - 0.15, t_brk_end), field, t_stop, 160.0, single_wheel=True)
    robot.motors.set_wheel_motors(-cmd_100[0], -cmd_100[1])
    _watch_sleep(robot, 1.15, True)
    robot.motors.stop_all_motors()
    _watch_sleep(robot, 0.20, True)
    return {"step_fwd": fwd_fit, "brake_high": brk_fit}


def _measure_chassis(robot: Any) -> Dict[str, Any]:
    """Run bounded closed-loop straight drive and in-place turn accuracy verification."""
    _, dist_fn, speed_fn = _sdk_units()
    deg_fn, _, _ = _sdk_units()
    p0 = robot.pose
    robot.behavior.drive_straight(dist_fn(80.0), speed_fn(75.0))
    _watch_sleep(robot, 0.12, True)
    p1 = robot.pose
    dx = float(p1.position.x) - float(p0.position.x)
    dy = float(p1.position.y) - float(p0.position.y)
    drive_mm = math.hypot(dx, dy)
    robot.behavior.drive_straight(dist_fn(-80.0), speed_fn(75.0))
    _watch_sleep(robot, 0.12, True)
    th0 = float(robot.pose.rotation.angle_z.radians)
    robot.behavior.turn_in_place(deg_fn(60.0))
    _watch_sleep(robot, 0.12, True)
    th1 = float(robot.pose.rotation.angle_z.radians)
    turned_deg = math.degrees(math.atan2(math.sin(th1 - th0), math.cos(th1 - th0)))
    robot.behavior.turn_in_place(deg_fn(-60.0))
    _watch_sleep(robot, 0.12, True)
    return {
        "drive_measured_mm": round(drive_mm, 2),
        "drive_err_mm": round(drive_mm - 80.0, 2),
        "turn_measured_deg": round(turned_deg, 2),
        "turn_err_deg": round(turned_deg - 60.0, 2),
    }


def run_mechanism_characterization(
    robot: Any,
    *,
    include_wheels: bool = False,
    confirm_safe_floor: bool = False,
    allow_occluded_tof: bool = False,
) -> Dict[str, Any]:
    """Execute physical mechanism tests and return system-identification report."""
    assert robot is not None
    if include_wheels:
        safe_ok, safe_reason, tel = check_wheel_bench_safety(robot, confirm_safe_floor, allow_occluded_tof)
        if not safe_ok:
            return {"ok": False, "error": safe_reason, "telemetry": tel, "include_wheels": True}
    else:
        _, _, tel = check_safety(robot, allow_on_charger=True, check_front_prox=False)
    rec = RobotStateRecorder()
    rec.attach(robot)
    try:
        time.sleep(0.08)
        mechs: Dict[str, Any] = {
            "head": _measure_head(robot, rec),
            "lift": _measure_lift(robot, rec),
        }
        chassis: Optional[Dict[str, Any]] = None
        if include_wheels:
            mechs["left_wheel"] = _measure_wheel(robot, rec, "left")
            mechs["right_wheel"] = _measure_wheel(robot, rec, "right")
            chassis = _measure_chassis(robot)
    finally:
        try:
            robot.motors.stop_all_motors()
        except Exception:  # noqa: BLE001
            pass
        rec.detach(robot)
    report: Dict[str, Any] = {
        "include_wheels": include_wheels,
        "telemetry": tel,
        "samples_recorded": len(rec.snapshot()),
        "mechanisms": mechs,
        "chassis": chassis,
    }
    judgement = evaluate_physical_realism(report, include_wheels=include_wheels)
    report.update(judgement)
    report["ok"] = bool(judgement["realistic"])
    return report
