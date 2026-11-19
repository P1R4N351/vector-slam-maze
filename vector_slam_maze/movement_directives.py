"""Bounded, sensor-guarded Vector translation and movement directives.

Stdlib-only module (no numpy/PIL) providing Ariadne-compatible MotionResult,
closed-loop ToF/cliff/origin-guarded bounded_drive and bounded_turn primitives,
non-holonomic 2D vector translation (dx, dy), head/lift bounds, offline dry-run
planning, and cumulative batch budget enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cliff_guard import CliffHazard
from .sdk_session import check_safety
from .types_and_math import Pose2D, normalize_angle, pose2d_from_sdk_pose

STOP_COMPLETED = "completed"
STOP_CLIFF = "cliff"
STOP_BLOCKED = "blocked"
STOP_REFUSED = "refused"
STOP_UNSUPPORTED = "unsupported"

MAX_FORWARD_DIRECTIVE_MM = 600.0
MAX_FORWARD_CHUNK_MM = 180.0
DEFAULT_FORWARD_STANDOFF_MM = 80.0
MAX_REVERSE_DIRECTIVE_MM = 150.0
MAX_REVERSE_CHUNK_MM = 50.0
LINEAR_DEADBAND_MM = 1.0
MIN_LINEAR_SPEED_MMPS = 20.0
MAX_FORWARD_SPEED_MMPS = 160.0
MAX_REVERSE_SPEED_MMPS = 60.0
DEFAULT_FORWARD_SPEED_MMPS = 75.0
DEFAULT_REVERSE_SPEED_MMPS = 45.0
DEFAULT_SHORTFALL_TOL_MM = 15.0
DEFAULT_SHORTFALL_FRAC = 0.15
SLIP_MIN_CMD_MM = 15.0
PURE_REVERSE_LATERAL_TOL_MM = 5.0

MAX_ROTATE_DIRECTIVE_DEG = 360.0
TURN_DEADBAND_DEG = 0.5
MIN_TURN_SPEED_DPS = 15.0
MAX_TURN_SPEED_DPS = 120.0
DEFAULT_TURN_SPEED_DPS = 60.0

MIN_HEAD_DEG = -22.0
MAX_HEAD_DEG = 45.0
MIN_LIFT_NORM = 0.0
MAX_LIFT_NORM = 1.0
LIFT_MIN_MM = 32.0
LIFT_MAX_MM = 92.0

MAX_CHUNKS_PER_DIRECTIVE = 16
MAX_DIRECTIVE_STEPS = 16
MAX_PAYLOAD_KEYS = 32
MAX_BATCH_DISTANCE_MM = 1200.0
MAX_BATCH_TURN_DEG = 720.0
MAX_BATCH_DURATION_S = 30.0
VALID_DIRECTIVE_VERBS = ("move", "rotate", "turn", "translate", "head", "lift", "undock", "dock", "halt")

_COMMON_KEYS = frozenset({"verb", "strict", "dry_run"})
_MOVE_KEYS = _COMMON_KEYS | frozenset({"dx_mm", "dx_m", "speed_mmps", "speed_mps", "standoff_mm", "dtheta_deg"})
_ROTATE_KEYS = _COMMON_KEYS | frozenset({"dtheta_rad", "dtheta_deg", "angle_deg", "speed_dps"})
_TRANSLATE_KEYS = _COMMON_KEYS | frozenset({
    "dx_mm", "dy_mm", "dx_m", "dy_m", "distance_mm", "bearing_deg", "speed_mmps", "speed_mps", "standoff_mm", "preserve_heading"
})
_HEAD_KEYS = _COMMON_KEYS | frozenset({"head_deg", "angle_deg"})
_LIFT_KEYS = _COMMON_KEYS | frozenset({"height_mm", "lift_norm", "height_norm"})
_VERB_ALLOWED_KEYS: Dict[str, frozenset] = {
    "move": _MOVE_KEYS,
    "rotate": _ROTATE_KEYS,
    "turn": _ROTATE_KEYS,
    "translate": _TRANSLATE_KEYS,
    "head": _HEAD_KEYS,
    "lift": _LIFT_KEYS,
    "undock": _COMMON_KEYS,
    "dock": _COMMON_KEYS,
    "halt": _COMMON_KEYS,
}


@dataclass(frozen=True)
class MotionResult:
    """Outcome of a bounded motion or directive request (Ariadne-compatible)."""

    ok: bool
    reason: str = STOP_COMPLETED
    moved_mm: float = 0.0
    turned_rad: float = 0.0
    detail: str = ""
    chunks: int = 0
    clamped: bool = False
    prox_mm: float = 9999.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize MotionResult into a JSON-safe dictionary."""
        assert isinstance(self.ok, bool) and isinstance(self.reason, str)
        assert math.isfinite(self.moved_mm) and math.isfinite(self.turned_rad)
        return {
            "ok": self.ok,
            "reason": self.reason,
            "moved_mm": round(float(self.moved_mm), 3),
            "turned_rad": round(float(self.turned_rad), 6),
            "detail": self.detail,
            "chunks": int(self.chunks),
            "clamped": bool(self.clamped),
            "prox_mm": round(float(self.prox_mm), 1),
        }


@dataclass(frozen=True)
class _FallbackAngle:
    """Lightweight angle wrapper matching anki_vector.util.Angle."""

    degrees: float

    @property
    def radians(self) -> float:
        """Return angle in radians."""
        assert math.isfinite(self.degrees) and abs(self.degrees) <= 3600.0
        return math.radians(self.degrees)


@dataclass(frozen=True)
class _FallbackDist:
    """Lightweight distance wrapper matching anki_vector.util.Distance."""

    distance_mm: float


@dataclass(frozen=True)
class _FallbackSpeed:
    """Lightweight speed wrapper matching anki_vector.util.Speed."""

    speed_mmps: float


def _sdk_units() -> Tuple[Any, Any, Any]:
    """Return anki_vector.util.{degrees,distance_mm,speed_mmps} or test-safe fallbacks."""
    assert callable(_FallbackAngle) and callable(_FallbackDist)
    try:
        from anki_vector.util import degrees, distance_mm, speed_mmps  # type: ignore

        return (degrees, distance_mm, speed_mmps)
    except ImportError:
        return (lambda v: _FallbackAngle(float(v)), lambda v: _FallbackDist(float(v)), lambda v: _FallbackSpeed(float(v)))


def _invoke_turn_in_place(behavior: Any, degrees_fn: Any, cmd_deg: float, spd_dps: float) -> None:
    """Invoke turn_in_place with optional speed kwarg matching receiver signature."""
    assert behavior is not None and math.isfinite(cmd_deg)
    assert math.isfinite(spd_dps) and spd_dps > 0.0
    ang = degrees_fn(cmd_deg)
    fn = behavior.turn_in_place
    try:
        params = inspect.signature(fn).parameters
        has_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if "speed" in params or has_kw:
            fn(ang, speed=degrees_fn(spd_dps))
            return
    except (ValueError, TypeError):
        pass
    fn(ang)


def _map_safety_stop(reason: str) -> str:
    """Map sdk_session check_safety reason onto Ariadne STOP_* taxonomy."""
    assert isinstance(reason, str) and reason != ""
    assert STOP_CLIFF == "cliff"
    if reason == "cliff":
        return STOP_CLIFF
    if reason == "proximity_blocked":
        return STOP_BLOCKED
    return STOP_REFUSED


def _as_finite_float(val: Any, name: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert scalar input to finite float or return structured validation error."""
    assert isinstance(name, str) and len(name) > 0
    assert val is None or not isinstance(val, dict), f"unexpected dict for {name}"
    if isinstance(val, bool):
        return (None, f"invalid_numeric:{name}")
    try:
        num = float(val)
    except (TypeError, ValueError):
        return (None, f"invalid_numeric:{name}")
    if not math.isfinite(num):
        return (None, f"invalid_numeric:{name}")
    return (num, None)


def _safe_pose(robot: Any) -> Optional[Pose2D]:
    """Read robot pose safely when complete finite planar pose is present, else None."""
    assert robot is not None
    raw = getattr(robot, "pose", None)
    assert raw is None or isinstance(raw, object)
    pos = getattr(raw, "position", None) if raw is not None else None
    rot = getattr(raw, "rotation", None) if raw is not None else None
    ang = getattr(rot, "angle_z", None) if rot is not None else None
    if pos is None or ang is None or not (hasattr(pos, "x") and hasattr(pos, "y") and hasattr(ang, "radians")):
        return None
    try:
        x_v, y_v, th_v = float(pos.x), float(pos.y), float(ang.radians)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x_v) and math.isfinite(y_v) and math.isfinite(th_v)):
        return None
    return pose2d_from_sdk_pose(raw)


def _unwrap_turned_rad(before_rad: float, after_rad: float, cmd_deg: float) -> float:
    """Compute signed turned radians from odometry without fabricating zero-motion turns."""
    assert math.isfinite(before_rad) and math.isfinite(after_rad)
    assert math.isfinite(cmd_deg)
    raw = normalize_angle(after_rad - before_rad)
    if abs(cmd_deg) > 180.0:
        cmd_rad = math.radians(cmd_deg)
        cand_pos = raw + 2.0 * math.pi if raw < 0.0 else raw
        cand_neg = raw - 2.0 * math.pi if raw > 0.0 else raw
        return cand_pos if cmd_deg > 0.0 and abs(cand_pos - cmd_rad) < abs(raw - cmd_rad) else (
            cand_neg if cmd_deg < 0.0 and abs(cand_neg - cmd_rad) < abs(raw - cmd_rad) else raw
        )
    return raw


def _bound_turn_args(delta_rad: float, speed_dps: float, strict: bool) -> Tuple[Optional[float], float, bool, Optional[str]]:
    """Validate and clamp or strictly reject angular turn parameters."""
    assert isinstance(strict, bool) and MAX_ROTATE_DIRECTIVE_DEG == 360.0
    ang_r, err_a = _as_finite_float(delta_rad, "delta_rad")
    spd_v, err_s = _as_finite_float(speed_dps, "speed_dps")
    if err_a or err_s or ang_r is None or spd_v is None:
        return (None, DEFAULT_TURN_SPEED_DPS, False, "invalid_numeric")
    deg = math.degrees(ang_r)
    clamped = abs(deg) > MAX_ROTATE_DIRECTIVE_DEG or spd_v < MIN_TURN_SPEED_DPS or spd_v > MAX_TURN_SPEED_DPS
    if strict and clamped:
        return (None, DEFAULT_TURN_SPEED_DPS, False, "bounds_exceeded")
    cmd_deg = max(-MAX_ROTATE_DIRECTIVE_DEG, min(MAX_ROTATE_DIRECTIVE_DEG, deg))
    cmd_spd = max(MIN_TURN_SPEED_DPS, min(MAX_TURN_SPEED_DPS, spd_v))
    return (cmd_deg, cmd_spd, clamped, None)


def bounded_turn(
    robot: Any, delta_rad: float, *, speed_dps: float = DEFAULT_TURN_SPEED_DPS, strict: bool = False, allow_on_charger: bool = False
) -> MotionResult:
    """Execute a bounded in-place rotation with pre/post safety, pose, and origin checks."""
    assert robot is not None, "robot required"
    assert isinstance(strict, bool) and isinstance(allow_on_charger, bool)
    cmd_deg, cmd_spd, clamped, err = _bound_turn_args(delta_rad, speed_dps, strict)
    if err is not None or cmd_deg is None:
        return MotionResult(False, STOP_REFUSED, detail=err or "invalid_numeric")
    ok, reason, pre = check_safety(robot, allow_on_charger=allow_on_charger, check_front_prox=False)
    if not ok:
        return MotionResult(False, _map_safety_stop(reason), detail=reason, clamped=clamped, prox_mm=pre["prox_mm"])
    p0 = _safe_pose(robot)
    if p0 is None:
        return MotionResult(False, STOP_REFUSED, turned_rad=0.0, detail="pose_unavailable", clamped=clamped, prox_mm=pre["prox_mm"])
    if abs(cmd_deg) < TURN_DEADBAND_DEG:
        return MotionResult(True, STOP_COMPLETED, turned_rad=0.0, detail="deadband", clamped=clamped, prox_mm=pre["prox_mm"])
    degrees_fn, _d, _s = _sdk_units()
    hazard: Optional[Tuple[str, str]] = None
    try:
        _invoke_turn_in_place(robot.behavior, degrees_fn, cmd_deg, cmd_spd)
    except CliffHazard as exc:
        hazard = _hazard_stop(exc)
    p1 = _safe_pose(robot)
    ok2, reason2, post = check_safety(robot, allow_on_charger=allow_on_charger, check_front_prox=False)
    if hazard is not None:  # the guard stopped the turn: report it, with the yaw that was measured
        swung = _unwrap_turned_rad(p0.theta_rad, p1.theta_rad, cmd_deg) if (p1 is not None and p1.origin_id == p0.origin_id) else 0.0
        return MotionResult(False, hazard[0], turned_rad=swung, detail=hazard[1], chunks=1, clamped=clamped, prox_mm=post["prox_mm"])
    if p1 is None:
        stop_r = _map_safety_stop(reason2) if not ok2 else STOP_REFUSED
        return MotionResult(False, stop_r, turned_rad=0.0, detail=reason2 if not ok2 else "pose_unavailable", chunks=1, clamped=clamped, prox_mm=post["prox_mm"])
    if p1.origin_id != p0.origin_id:
        return MotionResult(False, STOP_REFUSED, turned_rad=0.0, detail="origin_id_changed", chunks=1, clamped=clamped, prox_mm=post["prox_mm"])
    turned = _unwrap_turned_rad(p0.theta_rad, p1.theta_rad, cmd_deg)
    if not ok2:
        return MotionResult(False, _map_safety_stop(reason2), turned_rad=turned, detail=reason2, chunks=1, clamped=clamped, prox_mm=post["prox_mm"])
    return MotionResult(True, STOP_COMPLETED, turned_rad=turned, detail="goal", chunks=1, clamped=clamped, prox_mm=post["prox_mm"])


def _bound_drive_args(
    distance_mm: float, speed_mmps: Optional[float], max_chunk_mm: Optional[float], min_prox_mm: float, standoff_mm: float, strict: bool
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and bound forward/reverse drive distance, speed, and chunk caps."""
    assert isinstance(strict, bool) and MAX_FORWARD_DIRECTIVE_MM >= MAX_REVERSE_DIRECTIVE_MM
    dist, e1 = _as_finite_float(distance_mm, "distance_mm")
    mp, e2 = _as_finite_float(min_prox_mm, "min_prox_mm")
    so, e3 = _as_finite_float(standoff_mm, "standoff_mm")
    if e1 or e2 or e3 or dist is None or mp is None or so is None or mp < 0.0 or so < 0.0:
        return (None, "invalid_numeric")
    fwd = dist >= 0.0
    max_d = MAX_FORWARD_DIRECTIVE_MM if fwd else MAX_REVERSE_DIRECTIVE_MM
    max_spd = MAX_FORWARD_SPEED_MMPS if fwd else MAX_REVERSE_SPEED_MMPS
    chunk_def = MAX_FORWARD_CHUNK_MM if fwd else MAX_REVERSE_CHUNK_MM
    spd_in = (DEFAULT_FORWARD_SPEED_MMPS if fwd else DEFAULT_REVERSE_SPEED_MMPS) if speed_mmps is None else speed_mmps
    spd, e4 = _as_finite_float(spd_in, "speed_mmps")
    chk_v, e5 = _as_finite_float(chunk_def if max_chunk_mm is None else max_chunk_mm, "max_chunk_mm")
    if e4 or e5 or spd is None or chk_v is None or chk_v <= 0.0:
        return (None, "invalid_numeric")
    clamped = abs(dist) > max_d or spd < MIN_LINEAR_SPEED_MMPS or spd > max_spd or chk_v > chunk_def
    if strict and clamped:
        return (None, "bounds_exceeded")
    return ({
        "sign": 1.0 if fwd else -1.0,
        "mag_mm": min(abs(dist), max_d),
        "speed_mmps": max(MIN_LINEAR_SPEED_MMPS, min(max_spd, spd)),
        "chunk_mm": min(chk_v, chunk_def),
        "clamped": clamped,
    }, None)


def _compute_drive_step(
    rem_mm: float, sign: float, chunk_cap: float, tel: Dict[str, Any], min_prox_mm: float, standoff_mm: float
) -> Tuple[float, bool]:
    """Compute bounded step length and whether front ToF standoff limits this chunk."""
    assert rem_mm > 0.0 and chunk_cap > 0.0
    assert sign in (-1.0, 1.0)
    step = min(rem_mm, chunk_cap)
    if sign < 0.0:
        return (step, False)
    if bool(tel.get("prox_found")) and not bool(tel.get("prox_lift_in_fov")):
        eff_standoff = max(float(min_prox_mm), float(standoff_mm))
        avail = float(tel["prox_mm"]) - eff_standoff
        if avail < LINEAR_DEADBAND_MM:
            return (0.0, True)
        if avail < rem_mm:
            trimmed = min(step, avail)
            return (trimmed, trimmed >= avail - 1e-6)
    return (step, False)


def _hazard_stop(exc: CliffHazard) -> Tuple[str, str]:
    """Map a cliff-guard hazard onto (stop reason, detail) for a MotionResult.

    At DEFAULT priority only a mandatory physical reaction takes control away
    or cancels an action, so lost control and a failed action count as a cliff.
    """
    assert isinstance(exc, CliffHazard) and STOP_CLIFF == "cliff"
    assert isinstance(exc.hazard, str) and exc.hazard
    if exc.refused:
        return (STOP_REFUSED, "cliff_lockout")
    if exc.hazard in ("picked_up", "falling"):
        return (STOP_REFUSED, exc.hazard)
    if exc.hazard == "cliff_detected":
        return (STOP_CLIFF, "cliff")
    return (STOP_CLIFF, "control_lost")


def _run_drive_chunk(
    robot: Any, step_mm: float, sign: float, spd_mmps: float, dist_fn: Any, speed_fn: Any, start_oid: int, tol_opt: Optional[float]
) -> Tuple[float, str, str, float]:
    """Execute one bounded straight chunk and verify pose, origin, safety, and shortfall."""
    assert robot is not None and step_mm > 0.0
    assert sign in (-1.0, 1.0) and spd_mmps > 0.0
    p0 = _safe_pose(robot)
    if p0 is None:
        _ok0, _r0, pre0 = check_safety(robot, allow_on_charger=True, check_front_prox=False)
        return (0.0, STOP_REFUSED, "pose_unavailable", float(pre0["prox_mm"]))
    hazard: Optional[Tuple[str, str]] = None
    try:
        robot.behavior.drive_straight(dist_fn(sign * step_mm), speed_fn(spd_mmps))
    except CliffHazard as exc:
        hazard = _hazard_stop(exc)
    p1 = _safe_pose(robot)
    _ok2, reason2, post = check_safety(robot, allow_on_charger=True, check_front_prox=False)
    if hazard is not None:  # the guard stopped or refused the chunk: report it, with what was measured
        same_frame = p1 is not None and (start_oid < 0 or p1.origin_id == start_oid)
        went = ((p1.x_mm - p0.x_mm) * math.cos(p0.theta_rad) + (p1.y_mm - p0.y_mm) * math.sin(p0.theta_rad)) if same_frame else 0.0
        return (went, hazard[0], hazard[1], float(post["prox_mm"]))
    if p1 is None:
        if reason2 in ("cliff", "picked_up", "falling"):
            return (0.0, _map_safety_stop(reason2), reason2, float(post["prox_mm"]))
        return (0.0, STOP_REFUSED, "pose_unavailable", float(post["prox_mm"]))
    if start_oid >= 0 and p1.origin_id != start_oid:
        return (0.0, STOP_REFUSED, "origin_id_changed", float(post["prox_mm"]))
    moved = (p1.x_mm - p0.x_mm) * math.cos(p0.theta_rad) + (p1.y_mm - p0.y_mm) * math.sin(p0.theta_rad)
    if reason2 in ("cliff", "picked_up", "falling"):
        return (moved, _map_safety_stop(reason2), reason2, float(post["prox_mm"]))
    tol = float(tol_opt) if tol_opt is not None else max(DEFAULT_SHORTFALL_TOL_MM, step_mm * DEFAULT_SHORTFALL_FRAC)
    if step_mm >= SLIP_MIN_CMD_MM and (step_mm - moved * sign) > tol:
        return (moved, STOP_BLOCKED, "wheel_slip_or_blocked", float(post["prox_mm"]))
    return (moved, STOP_COMPLETED, "ok", float(post["prox_mm"]))


def _check_forward_prox_usable(tel: Dict[str, Any], allow_occluded_tof: bool) -> Optional[Tuple[str, str]]:
    """Fail-closed when front ToF telemetry is missing or occluded by the lift."""
    assert isinstance(tel, dict) and isinstance(allow_occluded_tof, bool)
    assert STOP_REFUSED == "refused"
    if not bool(tel.get("prox_valid", False)):
        return (STOP_REFUSED, "prox_unavailable")
    if bool(tel.get("prox_lift_in_fov", False)) and not allow_occluded_tof:
        return (STOP_REFUSED, "prox_occluded_lift_in_fov")
    return None


def _drive_chunks_loop(
    robot: Any, b: Dict[str, Any], min_prox_mm: float, standoff_mm: float, shortfall_tol_mm: Optional[float], start_oid: int, pre_tel: Dict[str, Any], allow_occluded_tof: bool = False
) -> MotionResult:
    """Iterate sensor-guarded chunks up to MAX_CHUNKS_PER_DIRECTIVE until target or hazard."""
    assert robot is not None and b["mag_mm"] >= LINEAR_DEADBAND_MM
    assert isinstance(allow_occluded_tof, bool)
    _a, dist_fn, speed_fn = _sdk_units()
    rem, moved_tot, chunks, last_prox = float(b["mag_mm"]), 0.0, 0, float(pre_tel["prox_mm"])
    for idx in range(MAX_CHUNKS_PER_DIRECTIVE):
        if rem < LINEAR_DEADBAND_MM:
            break
        ok, reason, tel = (True, "ok", pre_tel) if idx == 0 else check_safety(robot, allow_on_charger=False, min_prox_mm=min_prox_mm, check_front_prox=(b["sign"] > 0.0))
        last_prox = float(tel["prox_mm"])
        if not ok:
            return MotionResult(False, _map_safety_stop(reason), moved_mm=moved_tot, detail=reason, chunks=chunks, clamped=b["clamped"], prox_mm=last_prox)
        if idx > 0 and b["sign"] > 0.0:
            unusable = _check_forward_prox_usable(tel, allow_occluded_tof)
            if unusable is not None:
                return MotionResult(False, unusable[0], moved_mm=moved_tot, detail=unusable[1], chunks=chunks, clamped=b["clamped"], prox_mm=last_prox)
        step, at_standoff = _compute_drive_step(rem, b["sign"], b["chunk_mm"], tel, min_prox_mm, standoff_mm)
        if step < LINEAR_DEADBAND_MM:
            return MotionResult(False, STOP_BLOCKED, moved_mm=moved_tot, detail="proximity_standoff", chunks=chunks, clamped=b["clamped"], prox_mm=last_prox)
        dm, rcode, detail, post_prox = _run_drive_chunk(robot, step, b["sign"], b["speed_mmps"], dist_fn, speed_fn, start_oid, shortfall_tol_mm)
        moved_tot, chunks, last_prox = moved_tot + dm, chunks + 1, post_prox
        rem -= step
        if rcode != STOP_COMPLETED:
            return MotionResult(False, rcode, moved_mm=moved_tot, detail=detail, chunks=chunks, clamped=b["clamped"], prox_mm=last_prox)
        if at_standoff:
            return MotionResult(False, STOP_BLOCKED, moved_mm=moved_tot, detail="proximity_standoff", chunks=chunks, clamped=b["clamped"], prox_mm=last_prox)
    if rem >= LINEAR_DEADBAND_MM:
        return MotionResult(False, STOP_REFUSED, moved_mm=moved_tot, detail="chunk_budget_exhausted", chunks=chunks, clamped=b["clamped"], prox_mm=last_prox)
    return MotionResult(True, STOP_COMPLETED, moved_mm=moved_tot, detail="goal", chunks=chunks, clamped=b["clamped"], prox_mm=last_prox)


def bounded_drive(
    robot: Any,
    distance_mm: float,
    *,
    speed_mmps: Optional[float] = None,
    min_prox_mm: float = 35.0,
    standoff_mm: float = DEFAULT_FORWARD_STANDOFF_MM,
    max_chunk_mm: Optional[float] = None,
    shortfall_tol_mm: Optional[float] = None,
    strict: bool = False,
    allow_occluded_tof: bool = False,
) -> MotionResult:
    """Drive straight in ToF/cliff/origin-guarded chunks with hard asymmetric bounds."""
    assert robot is not None and isinstance(allow_occluded_tof, bool), "robot and bool flag required"
    assert isinstance(strict, bool)
    b, err = _bound_drive_args(distance_mm, speed_mmps, max_chunk_mm, min_prox_mm, standoff_mm, strict)
    if err is not None or b is None:
        return MotionResult(False, STOP_REFUSED, detail=err or "invalid_numeric")
    fwd_move = b["sign"] > 0.0 and b["mag_mm"] >= LINEAR_DEADBAND_MM
    ok, reason, pre = check_safety(robot, allow_on_charger=False, min_prox_mm=min_prox_mm, check_front_prox=fwd_move)
    if not ok:
        return MotionResult(False, _map_safety_stop(reason), detail=reason, clamped=b["clamped"], prox_mm=pre["prox_mm"])
    if fwd_move:
        unusable = _check_forward_prox_usable(pre, allow_occluded_tof)
        if unusable is not None:
            return MotionResult(False, unusable[0], detail=unusable[1], clamped=b["clamped"], prox_mm=pre["prox_mm"])
    p0 = _safe_pose(robot)
    if p0 is None:
        return MotionResult(False, STOP_REFUSED, moved_mm=0.0, detail="pose_unavailable", clamped=b["clamped"], prox_mm=pre["prox_mm"])
    if b["mag_mm"] < LINEAR_DEADBAND_MM:
        return MotionResult(True, STOP_COMPLETED, moved_mm=0.0, detail="deadband", clamped=b["clamped"], prox_mm=pre["prox_mm"])
    return _drive_chunks_loop(robot, b, min_prox_mm, standoff_mm, shortfall_tol_mm, p0.origin_id, pre, allow_occluded_tof=allow_occluded_tof)


def _resolve_strict_and_keys(verb: str, payload: Dict[str, Any], caller_strict: bool) -> Tuple[bool, Optional[str]]:
    """Enforce caller strict floor, boolean flags, and per-verb payload key whitelist."""
    assert isinstance(payload, dict) and isinstance(caller_strict, bool)
    assert MAX_PAYLOAD_KEYS == 32 and verb in VALID_DIRECTIVE_VERBS
    if len(payload) > MAX_PAYLOAD_KEYS:
        return (True, "payload_too_many_keys")
    for bkey in ("strict", "dry_run"):
        if bkey in payload and not isinstance(payload[bkey], bool):
            return (True, f"invalid_boolean:{bkey}")
    allowed = _VERB_ALLOWED_KEYS[verb]
    extra = [str(k) for k in list(payload.keys())[:MAX_PAYLOAD_KEYS] if k not in allowed]
    if extra:
        return (True, f"unknown_keys:{','.join(sorted(extra))}")
    return (bool(caller_strict) or bool(payload.get("strict", False)), None)


def _resolve_standoff_mm(
    payload: Dict[str, Any], caller_standoff_mm: float, min_prox_mm: float, strict: bool
) -> Tuple[Optional[float], bool, Optional[str]]:
    """Enforce caller standoff_mm floor; payloads may raise standoff but never lower it."""
    assert isinstance(payload, dict) and isinstance(strict, bool)
    assert caller_standoff_mm >= 0.0 and min_prox_mm >= 0.0
    floor_so = max(float(min_prox_mm), float(caller_standoff_mm))
    if "standoff_mm" not in payload:
        return (floor_so, False, None)
    req_so, err = _as_finite_float(payload["standoff_mm"], "standoff_mm")
    if err or req_so is None or req_so < 0.0:
        return (None, False, "invalid_numeric")
    so_clamped = req_so < floor_so
    if strict and so_clamped:
        return (None, False, "bounds_exceeded")
    return (max(floor_so, req_so), so_clamped, None)


def _parse_optional_linear_speed(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """Extract optional linear speed in mm/s from speed_mmps or speed_mps."""
    assert isinstance(payload, dict)
    assert MIN_LINEAR_SPEED_MMPS > 0.0
    if "speed_mmps" in payload and "speed_mps" in payload:
        return (None, "conflicting_speed_keys")
    if "speed_mmps" in payload:
        spd, err = _as_finite_float(payload["speed_mmps"], "speed_mmps")
        return (spd, err) if not err else (None, "invalid_numeric")
    if "speed_mps" in payload:
        mps, err = _as_finite_float(payload["speed_mps"], "speed_mps")
        return (None if mps is None else mps * 1000.0, err) if not err else (None, "invalid_numeric")
    return (None, None)


def _normalize_move_payload(
    payload: Dict[str, Any], strict: bool, *, min_prox_mm: float = 35.0, caller_standoff_mm: float = DEFAULT_FORWARD_STANDOFF_MM
) -> Dict[str, Any]:
    """Validate and normalize a straight move payload requiring explicit distance."""
    assert isinstance(payload, dict) and isinstance(strict, bool)
    assert min_prox_mm >= 0.0 and caller_standoff_mm >= 0.0
    has_mm, has_m = "dx_mm" in payload, "dx_m" in payload
    if not has_mm and not has_m:
        return {"ok": False, "stopped_reason": "missing_distance", "error": "missing_distance"}
    if has_mm and has_m:
        return {"ok": False, "stopped_reason": "conflicting_distance_keys", "error": "conflicting_distance_keys"}
    if "dtheta_deg" in payload:
        dth, e_th = _as_finite_float(payload["dtheta_deg"], "dtheta_deg")
        if e_th or dth is None:
            return {"ok": False, "stopped_reason": "invalid_numeric", "error": e_th}
        if abs(dth) > 1e-6:
            return {"ok": False, "stopped_reason": "nonzero_dtheta_on_move", "error": "nonzero_dtheta_on_move"}
    dist, e1 = _as_finite_float(payload["dx_mm"], "dx_mm") if has_mm else _as_finite_float(payload["dx_m"], "dx_m")
    if not has_mm and dist is not None:
        dist *= 1000.0
    raw_spd, e_spd = _parse_optional_linear_speed(payload)
    so, so_clamped, e_so = _resolve_standoff_mm(payload, caller_standoff_mm, min_prox_mm, strict)
    if e1 or dist is None or e_spd is not None or e_so is not None or so is None:
        reason = e_spd if e_spd == "conflicting_speed_keys" else (e_so or "invalid_numeric")
        return {"ok": False, "stopped_reason": reason, "error": e1 or e_spd or e_so or "invalid_numeric"}
    b, err = _bound_drive_args(dist, raw_spd, None, min_prox_mm, so, strict)
    if err or b is None:
        return {"ok": False, "stopped_reason": err or "bounds_exceeded", "error": err or "bounds_exceeded"}
    signed_mm = b["sign"] * b["mag_mm"]
    return {"ok": True, "dist_mm": signed_mm, "speed_mmps": b["speed_mmps"], "chunk_mm": b["chunk_mm"], "standoff_mm": so, "clamped": bool(b["clamped"] or so_clamped), "distance_mm": b["mag_mm"], "turn_deg": 0.0}


def _normalize_rotate_payload(payload: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    """Validate and normalize a rotate/turn payload into degrees + deg/s."""
    assert isinstance(payload, dict) and isinstance(strict, bool)
    assert MAX_ROTATE_DIRECTIVE_DEG == 360.0
    if sum(1 for k in ("dtheta_rad", "dtheta_deg", "angle_deg") if k in payload) > 1:
        return {"ok": False, "stopped_reason": "conflicting_angle_keys", "error": "conflicting_angle_keys"}
    if "dtheta_rad" in payload:
        rad, e1 = _as_finite_float(payload["dtheta_rad"], "dtheta_rad")
        deg = math.degrees(rad) if rad is not None else None
    else:
        raw_d = payload["angle_deg"] if "angle_deg" in payload else payload.get("dtheta_deg", 0.0)
        deg, e1 = _as_finite_float(raw_d, "dtheta_deg")
    spd, e2 = _as_finite_float(payload.get("speed_dps", DEFAULT_TURN_SPEED_DPS), "speed_dps")
    if e1 or e2 or deg is None or spd is None:
        return {"ok": False, "stopped_reason": "invalid_numeric", "error": e1 or e2}
    cmd_deg, cmd_spd, clamped, err = _bound_turn_args(math.radians(deg), spd, strict)
    if err or cmd_deg is None:
        return {"ok": False, "stopped_reason": err or "bounds_exceeded", "error": err or "bounds_exceeded"}
    return {"ok": True, "dtheta_deg": cmd_deg, "speed_dps": cmd_spd, "clamped": clamped, "distance_mm": 0.0, "turn_deg": abs(cmd_deg)}


def _extract_translate_xy(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Extract body-frame (dx_mm, dy_mm) from Cartesian or polar translate fields."""
    assert isinstance(payload, dict)
    assert PURE_REVERSE_LATERAL_TOL_MM > 0.0
    polar = any(k in payload for k in ("distance_mm", "bearing_deg"))
    cart = any(k in payload for k in ("dx_mm", "dy_mm", "dx_m", "dy_m"))
    if not polar and not cart:
        return (None, None, "missing_translation")
    if polar and cart:
        return (None, None, "conflicting_coordinate_keys")
    if polar:
        if "distance_mm" not in payload:
            return (None, None, "missing_distance")
        dist, e1 = _as_finite_float(payload["distance_mm"], "distance_mm")
        brg, e2 = _as_finite_float(payload.get("bearing_deg", 0.0), "bearing_deg")
        if e1 or e2 or dist is None or brg is None:
            return (None, None, "invalid_numeric")
        rad = math.radians(brg)
        return (dist * math.cos(rad), dist * math.sin(rad), None)
    if ("dx_mm" in payload and "dx_m" in payload) or ("dy_mm" in payload and "dy_m" in payload):
        return (None, None, "conflicting_coordinate_keys")
    dx, e1 = _as_finite_float(payload.get("dx_mm", 0.0), "dx_mm") if "dx_m" not in payload else _as_finite_float(payload["dx_m"], "dx_m")
    dy, e2 = _as_finite_float(payload.get("dy_mm", 0.0), "dy_mm") if "dy_m" not in payload else _as_finite_float(payload["dy_m"], "dy_m")
    if e1 or e2 or dx is None or dy is None:
        return (None, None, "invalid_numeric")
    return (dx * (1000.0 if "dx_m" in payload else 1.0), dy * (1000.0 if "dy_m" in payload else 1.0), None)


def _normalize_translate_payload(
    payload: Dict[str, Any], strict: bool, *, min_prox_mm: float = 35.0, caller_standoff_mm: float = DEFAULT_FORWARD_STANDOFF_MM
) -> Dict[str, Any]:
    """Normalize 2D translation (dx, dy) into bounded pure-reverse or turn-drive-restore."""
    assert isinstance(payload, dict) and isinstance(strict, bool)
    assert min_prox_mm >= 0.0 and caller_standoff_mm >= 0.0
    dx, dy, err = _extract_translate_xy(payload)
    if err or dx is None or dy is None:
        return {"ok": False, "stopped_reason": err or "invalid_numeric", "error": err or "invalid_numeric"}
    if "preserve_heading" in payload and not isinstance(payload["preserve_heading"], bool):
        return {"ok": False, "stopped_reason": "invalid_boolean:preserve_heading", "error": "invalid_boolean:preserve_heading"}
    spd_raw, e_spd = _parse_optional_linear_speed(payload)
    so, so_clamped, e_so = _resolve_standoff_mm(payload, caller_standoff_mm, min_prox_mm, strict)
    if e_spd is not None or e_so is not None or so is None:
        reason = e_spd if e_spd == "conflicting_speed_keys" else (e_so or "invalid_numeric")
        return {"ok": False, "stopped_reason": reason, "error": e_spd or e_so or "invalid_numeric"}
    pure_rev = dx < 0.0 and abs(dy) <= PURE_REVERSE_LATERAL_TOL_MM
    b, b_err = _bound_drive_args(dx if pure_rev else math.hypot(dx, dy), spd_raw, None, min_prox_mm, so, strict)
    if b_err or b is None:
        return {"ok": False, "stopped_reason": b_err or "bounds_exceeded", "error": b_err or "bounds_exceeded"}
    brg_deg = 0.0 if (pure_rev or b["mag_mm"] < LINEAR_DEADBAND_MM) else math.degrees(math.atan2(dy, dx))
    if abs(brg_deg) < TURN_DEADBAND_DEG:
        brg_deg = 0.0
    preserve = bool(payload.get("preserve_heading", False)) and not pure_rev and abs(brg_deg) >= TURN_DEADBAND_DEG
    return {
        "ok": True, "dx_mm": b["sign"] * b["mag_mm"] * math.cos(math.radians(brg_deg)), "dy_mm": 0.0 if pure_rev else b["mag_mm"] * math.sin(math.radians(brg_deg)),
        "pure_reverse": pure_rev, "bearing_deg": brg_deg, "drive_mm": b["sign"] * b["mag_mm"], "speed_mmps": b["speed_mmps"], "standoff_mm": so,
        "preserve_heading": preserve, "clamped": bool(b["clamped"] or so_clamped), "distance_mm": b["mag_mm"], "turn_deg": abs(brg_deg) * (2.0 if preserve else 1.0),
    }


def _normalize_head_lift_payload(verb: str, payload: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    """Validate and clamp head angle (-22..45 deg) or lift height (0..1 norm / 32..92 mm)."""
    assert verb in ("head", "lift") and isinstance(strict, bool)
    assert MIN_HEAD_DEG < MAX_HEAD_DEG and MIN_LIFT_NORM < MAX_LIFT_NORM
    if verb == "head":
        if "head_deg" not in payload and "angle_deg" not in payload:
            return {"ok": False, "stopped_reason": "missing_head_angle", "error": "missing_head_angle"}
        deg, err = _as_finite_float(payload["head_deg"] if "head_deg" in payload else payload["angle_deg"], "head_deg")
        if err or deg is None:
            return {"ok": False, "stopped_reason": "invalid_numeric", "error": err}
        clamped = deg < MIN_HEAD_DEG or deg > MAX_HEAD_DEG
        if strict and clamped:
            return {"ok": False, "stopped_reason": "bounds_exceeded", "error": "head_deg_out_of_bounds"}
        return {"ok": True, "head_deg": max(MIN_HEAD_DEG, min(MAX_HEAD_DEG, deg)), "clamped": clamped, "distance_mm": 0.0, "turn_deg": 0.0}
    if not any(k in payload for k in ("height_mm", "lift_norm", "height_norm")):
        return {"ok": False, "stopped_reason": "missing_lift_height", "error": "missing_lift_height"}
    if "height_mm" in payload:
        hmm, err = _as_finite_float(payload["height_mm"], "height_mm")
        norm = None if (err or hmm is None) else (hmm - LIFT_MIN_MM) / (LIFT_MAX_MM - LIFT_MIN_MM)
    else:
        norm, err = _as_finite_float(payload["lift_norm"] if "lift_norm" in payload else payload["height_norm"], "lift_norm")
    if err or norm is None:
        return {"ok": False, "stopped_reason": "invalid_numeric", "error": err}
    clamped = norm < MIN_LIFT_NORM or norm > MAX_LIFT_NORM
    if strict and clamped:
        return {"ok": False, "stopped_reason": "bounds_exceeded", "error": "lift_out_of_bounds"}
    return {"ok": True, "lift_norm": max(MIN_LIFT_NORM, min(MAX_LIFT_NORM, norm)), "clamped": clamped, "distance_mm": 0.0, "turn_deg": 0.0}


def _decompose_primitives(verb: str, norm: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
    """Expand a normalized directive into bounded primitive steps and duration estimate."""
    assert verb in VALID_DIRECTIVE_VERBS and norm.get("ok") is True
    prims: List[Dict[str, Any]] = []
    if verb == "move":
        _append_drive_chunks(prims, norm["dist_mm"], norm["speed_mmps"])
    elif verb in ("rotate", "turn") and abs(norm["dtheta_deg"]) >= TURN_DEADBAND_DEG:
        prims.append({"op": "turn_in_place", "angle_deg": round(norm["dtheta_deg"], 3), "speed_dps": round(norm["speed_dps"], 2)})
    elif verb == "translate":
        if abs(norm["bearing_deg"]) >= TURN_DEADBAND_DEG:
            prims.append({"op": "turn_in_place", "angle_deg": round(norm["bearing_deg"], 3), "speed_dps": DEFAULT_TURN_SPEED_DPS})
        _append_drive_chunks(prims, norm["drive_mm"], norm["speed_mmps"])
        if norm["preserve_heading"]:
            prims.append({"op": "turn_in_place", "angle_deg": round(-norm["bearing_deg"], 3), "speed_dps": DEFAULT_TURN_SPEED_DPS})
    elif verb in ("head", "lift", "undock", "dock", "halt"):
        prims.append({"op": verb, **{k: v for k, v in norm.items() if k not in ("ok", "clamped", "distance_mm", "turn_deg")}})
    dur = (norm.get("distance_mm", 0.0) / max(MIN_LINEAR_SPEED_MMPS, float(norm.get("speed_mmps", DEFAULT_FORWARD_SPEED_MMPS)))) + (norm.get("turn_deg", 0.0) / DEFAULT_TURN_SPEED_DPS)
    return (prims, round(dur, 3))


def _append_drive_chunks(prims: List[Dict[str, Any]], signed_mm: float, spd_mmps: float) -> None:
    """Append bounded forward/reverse chunk primitives into list."""
    assert math.isfinite(signed_mm) and spd_mmps > 0.0
    if abs(signed_mm) < LINEAR_DEADBAND_MM:
        return
    cap = MAX_FORWARD_CHUNK_MM if signed_mm >= 0.0 else MAX_REVERSE_CHUNK_MM
    sign, rem = (1.0 if signed_mm >= 0.0 else -1.0), abs(signed_mm)
    for _ in range(MAX_CHUNKS_PER_DIRECTIVE):
        if rem < LINEAR_DEADBAND_MM:
            break
        step = min(rem, cap)
        prims.append({"op": "drive_straight", "distance_mm": round(sign * step, 3), "speed_mmps": round(spd_mmps, 2)})
        rem -= step


def plan_directive(
    verb: str, payload: Optional[Dict[str, Any]] = None, *, strict: bool = False, min_prox_mm: float = 35.0, standoff_mm: float = DEFAULT_FORWARD_STANDOFF_MM
) -> Dict[str, Any]:
    """Validate, bound, and decompose one movement directive without robot hardware."""
    assert isinstance(strict, bool) and min_prox_mm >= 0.0 and standoff_mm >= 0.0
    if verb not in VALID_DIRECTIVE_VERBS:
        return {"ok": False, "dry_run": True, "verb": verb, "stopped_reason": f"unknown_verb:{verb}", "error": "unknown_verb"}
    if payload is not None and not isinstance(payload, dict):
        return {"ok": False, "dry_run": True, "verb": verb, "stopped_reason": "invalid_payload", "error": "payload_must_be_dict"}
    pl = dict(payload or {})
    st_flag, k_err = _resolve_strict_and_keys(verb, pl, strict)
    if k_err is not None:
        return {"ok": False, "dry_run": True, "verb": verb, "stopped_reason": k_err, "error": k_err}
    if verb == "move":
        norm = _normalize_move_payload(pl, st_flag, min_prox_mm=min_prox_mm, caller_standoff_mm=standoff_mm)
    elif verb in ("rotate", "turn"):
        norm = _normalize_rotate_payload(pl, st_flag)
    elif verb == "translate":
        norm = _normalize_translate_payload(pl, st_flag, min_prox_mm=min_prox_mm, caller_standoff_mm=standoff_mm)
    elif verb in ("head", "lift"):
        norm = _normalize_head_lift_payload(verb, pl, st_flag)
    else:
        norm = {"ok": True, "clamped": False, "distance_mm": 0.0, "turn_deg": 0.0}
    if not norm.get("ok"):
        return {"ok": False, "dry_run": True, "verb": verb, "stopped_reason": norm["stopped_reason"], "error": norm.get("error", norm["stopped_reason"])}
    prims, est_s = _decompose_primitives(verb, norm)
    return {
        "ok": True, "dry_run": True, "verb": verb, "clamped": bool(norm["clamped"]),
        "normalized": {k: v for k, v in norm.items() if k not in ("ok", "clamped")},
        "primitives": prims, "estimated_duration_s": est_s,
        "bounds": {"max_forward_mm": MAX_FORWARD_DIRECTIVE_MM, "max_reverse_mm": MAX_REVERSE_DIRECTIVE_MM, "max_turn_deg": MAX_ROTATE_DIRECTIVE_DEG},
    }


def plan_directive_batch(
    directives: Sequence[Dict[str, Any]], *, strict: bool = False, min_prox_mm: float = 35.0, standoff_mm: float = DEFAULT_FORWARD_STANDOFF_MM
) -> Dict[str, Any]:
    """Validate and bound a sequence of movement directives against cumulative caps."""
    assert isinstance(strict, bool) and MAX_DIRECTIVE_STEPS == 16
    if not isinstance(directives, (list, tuple)) or len(directives) == 0:
        return {"ok": False, "dry_run": True, "stopped_reason": "empty_batch", "error": "empty_batch"}
    if len(directives) > MAX_DIRECTIVE_STEPS:
        return {"ok": False, "dry_run": True, "stopped_reason": "batch_steps_exceeded", "error": f"steps>{MAX_DIRECTIVE_STEPS}"}
    plans, tot_d, tot_t, clamped = [], 0.0, 0.0, False
    for idx, entry in enumerate(directives):
        if not isinstance(entry, dict) or not entry.get("verb"):
            return {"ok": False, "dry_run": True, "stopped_reason": f"invalid_step:{idx}", "error": "missing_verb"}
        sub_pl = entry["args"] if ("args" in entry and isinstance(entry["args"], dict)) else entry
        p = plan_directive(str(entry["verb"]), sub_pl, strict=strict, min_prox_mm=min_prox_mm, standoff_mm=standoff_mm)
        if not p["ok"]:
            return {"ok": False, "dry_run": True, "failed_step": idx, "stopped_reason": p["stopped_reason"], "error": p["error"]}
        tot_d += float(p["normalized"].get("distance_mm", 0.0))
        tot_t += float(p["normalized"].get("turn_deg", 0.0))
        clamped = clamped or bool(p["clamped"])
        if tot_d > MAX_BATCH_DISTANCE_MM or tot_t > MAX_BATCH_TURN_DEG:
            return {"ok": False, "dry_run": True, "failed_step": idx, "stopped_reason": "batch_budget_exceeded", "error": "cumulative_budget_exceeded"}
        plans.append(p)
    return {"ok": True, "dry_run": True, "verb": "directives", "step_count": len(plans), "total_distance_mm": round(tot_d, 2), "total_turn_deg": round(tot_t, 2), "clamped": clamped, "plans": plans}


def _execute_translate(
    robot: Any, norm: Dict[str, Any], min_prox_mm: float, standoff_mm: float, strict: bool
) -> Tuple[MotionResult, int]:
    """Execute decomposed 2D translation (turn -> chunked drive -> optional counter-turn)."""
    assert robot is not None and isinstance(norm, dict)
    total_moved, total_turned, chunks, clamped = 0.0, 0.0, 0, bool(norm.get("clamped", False))
    if abs(norm["bearing_deg"]) >= TURN_DEADBAND_DEG:
        tr = bounded_turn(robot, math.radians(norm["bearing_deg"]), strict=strict)
        total_turned += tr.turned_rad
        chunks += tr.chunks
        if not tr.ok:
            return (MotionResult(False, tr.reason, total_moved, total_turned, tr.detail, chunks, clamped, tr.prox_mm), chunks)
    dr = bounded_drive(robot, norm["drive_mm"], speed_mmps=norm["speed_mmps"], min_prox_mm=min_prox_mm, standoff_mm=standoff_mm, strict=strict)
    total_moved, chunks = dr.moved_mm, chunks + dr.chunks
    if not dr.ok:
        return (MotionResult(False, dr.reason, total_moved, total_turned, dr.detail, chunks, clamped or dr.clamped, dr.prox_mm), chunks)
    if norm["preserve_heading"]:
        tr2 = bounded_turn(robot, math.radians(-norm["bearing_deg"]), strict=strict)
        total_turned += tr2.turned_rad
        chunks += tr2.chunks
        if not tr2.ok:
            return (MotionResult(False, tr2.reason, total_moved, total_turned, tr2.detail, chunks, clamped, tr2.prox_mm), chunks)
        return (MotionResult(True, STOP_COMPLETED, total_moved, total_turned, "goal", chunks, clamped, tr2.prox_mm), chunks)
    return (MotionResult(True, STOP_COMPLETED, total_moved, total_turned, dr.detail, chunks, clamped or dr.clamped, dr.prox_mm), chunks)


def _execute_head_or_lift(robot: Any, verb: str, norm: Dict[str, Any]) -> MotionResult:
    """Execute bounded head angle or lift height command after safety check."""
    assert robot is not None and verb in ("head", "lift")
    degrees_fn, _d, _s = _sdk_units()
    ok, reason, tel = check_safety(robot, allow_on_charger=True, check_front_prox=False)
    if not ok:
        return MotionResult(False, _map_safety_stop(reason), detail=reason, clamped=bool(norm["clamped"]), prox_mm=tel["prox_mm"])
    if verb == "head":
        robot.behavior.set_head_angle(degrees_fn(norm["head_deg"]))
    else:
        robot.behavior.set_lift_height(float(norm["lift_norm"]))
    return MotionResult(True, STOP_COMPLETED, detail="goal", chunks=1, clamped=bool(norm["clamped"]), prox_mm=tel["prox_mm"])


def _execute_dock_or_halt(robot: Any, verb: str) -> MotionResult:
    """Execute undock, dock, or halt safety action."""
    assert robot is not None and verb in ("undock", "dock", "halt")
    if verb == "halt":
        robot.motors.stop_all_motors()
        _ok, _r, tel = check_safety(robot, allow_on_charger=True, check_front_prox=False)
        return MotionResult(True, STOP_COMPLETED, detail="halt", prox_mm=tel["prox_mm"])
    ok, reason, tel = check_safety(robot, allow_on_charger=True, check_front_prox=False)
    if not ok:
        return MotionResult(False, _map_safety_stop(reason), detail=reason, prox_mm=tel["prox_mm"])
    if verb == "undock":
        robot.behavior.drive_off_charger()
        return MotionResult(True, STOP_COMPLETED, detail="goal", chunks=1, prox_mm=tel["prox_mm"])
    resp = robot.behavior.drive_on_charger()
    dock_res = getattr(getattr(resp, "result", None), "name", None) or str(resp or "dock_attempted")
    return MotionResult(True, STOP_COMPLETED, detail=dock_res, chunks=1, prox_mm=tel["prox_mm"])


def _format_act_response(p0: Optional[Pose2D], p1: Optional[Pose2D], mr: MotionResult, tel: Dict[str, Any]) -> Dict[str, Any]:
    """Format Cartographer + Ariadne compatible response JSON dictionary."""
    assert isinstance(mr, MotionResult) and isinstance(tel, dict)
    dx_m = round(((p1.x_mm - p0.x_mm) / 1000.0) if (p0 and p1) else (mr.moved_mm / 1000.0), 4)
    dy_m = round(((p1.y_mm - p0.y_mm) / 1000.0) if (p0 and p1) else 0.0, 4)
    dth = round(normalize_angle(p1.theta_rad - p0.theta_rad) if (p0 and p1) else mr.turned_rad, 5)
    stopped = mr.detail if not mr.ok else ("halt" if mr.detail == "halt" else "goal")
    return {
        "ok": mr.ok,
        "stopped_reason": stopped,
        "prox_mm": round(float(tel["prox_mm"]), 1),
        "actual_delta": {"x": dx_m, "y": dy_m, "theta": dth},
        "clamped": mr.clamped,
        "chunks_executed": mr.chunks,
        "motion_result": mr.to_dict(),
    }


def execute_bounded_directive(
    robot: Any, verb: str, payload: Optional[Dict[str, Any]] = None, *, min_prox_mm: float = 35.0, standoff_mm: float = DEFAULT_FORWARD_STANDOFF_MM, strict: bool = False
) -> Dict[str, Any]:
    """Validate bounds, execute a bounded directive on robot, and return telemetry."""
    assert robot is not None and min_prox_mm >= 0.0
    plan = plan_directive(verb, payload, strict=strict, min_prox_mm=min_prox_mm, standoff_mm=standoff_mm)
    _ok, _r, pre_tel = check_safety(robot, allow_on_charger=True, check_front_prox=False)
    if not plan["ok"]:
        mr = MotionResult(False, STOP_REFUSED, detail=plan["stopped_reason"], prox_mm=pre_tel["prox_mm"])
        return _format_act_response(None, None, mr, pre_tel)
    p0 = _safe_pose(robot)
    st_flag = bool(strict) or bool((payload or {}).get("strict", False))
    norm = {**plan["normalized"], "clamped": bool(plan["clamped"])}
    if verb == "move":
        mr = bounded_drive(robot, norm["dist_mm"], speed_mmps=norm["speed_mmps"], min_prox_mm=min_prox_mm, standoff_mm=norm["standoff_mm"], strict=st_flag)
    elif verb in ("rotate", "turn"):
        mr = bounded_turn(robot, math.radians(norm["dtheta_deg"]), speed_dps=norm["speed_dps"], strict=st_flag)
    elif verb == "translate":
        mr, _ = _execute_translate(robot, norm, min_prox_mm, norm["standoff_mm"], st_flag)
    elif verb in ("head", "lift"):
        mr = _execute_head_or_lift(robot, verb, norm)
    else:
        mr = _execute_dock_or_halt(robot, verb)
    if plan["clamped"] and not mr.clamped:
        mr = replace(mr, clamped=True)
    p1 = _safe_pose(robot)
    _ok2, _r2, post_tel = check_safety(robot, allow_on_charger=True, check_front_prox=False)
    return _format_act_response(p0, p1, mr, post_tel)


def execute_directive_batch(
    robot: Any, directives: Sequence[Dict[str, Any]], *, min_prox_mm: float = 35.0, standoff_mm: float = DEFAULT_FORWARD_STANDOFF_MM, strict: bool = False
) -> Dict[str, Any]:
    """Execute a bounded sequence of directives, halting on budget, timeout, or hazard."""
    assert robot is not None and MAX_BATCH_DURATION_S > 0.0
    batch_plan = plan_directive_batch(directives, strict=strict, min_prox_mm=min_prox_mm, standoff_mm=standoff_mm)
    _ok, _r, tel0 = check_safety(robot, allow_on_charger=True, check_front_prox=False)
    if not batch_plan["ok"]:
        return {"ok": False, "stopped_reason": batch_plan["stopped_reason"], "steps_completed": 0, "results": [], "prox_mm": tel0["prox_mm"]}
    t0, p0, results = time.monotonic(), _safe_pose(robot), []
    for entry in directives[:MAX_DIRECTIVE_STEPS]:
        if time.monotonic() - t0 > MAX_BATCH_DURATION_S:
            return {"ok": False, "stopped_reason": "batch_timeout", "steps_completed": len(results), "results": results, "prox_mm": tel0["prox_mm"]}
        sub_pl = entry["args"] if ("args" in entry and isinstance(entry["args"], dict)) else entry
        step_res = execute_bounded_directive(robot, str(entry["verb"]), sub_pl, min_prox_mm=min_prox_mm, standoff_mm=standoff_mm, strict=strict)
        results.append(step_res)
        if not step_res["ok"]:
            return {"ok": False, "stopped_reason": step_res["stopped_reason"], "steps_completed": len(results) - 1, "results": results, "prox_mm": step_res["prox_mm"]}
    p1 = _safe_pose(robot)
    _ok2, _r2, post_tel = check_safety(robot, allow_on_charger=True, check_front_prox=False)
    return {
        "ok": True,
        "stopped_reason": "goal",
        "steps_completed": len(results),
        "results": results,
        "prox_mm": post_tel["prox_mm"],
        "actual_delta": {
            "x": round(((p1.x_mm - p0.x_mm) / 1000.0) if (p0 and p1) else 0.0, 4),
            "y": round(((p1.y_mm - p0.y_mm) / 1000.0) if (p0 and p1) else 0.0, 4),
            "theta": round(normalize_angle(p1.theta_rad - p0.theta_rad) if (p0 and p1) else 0.0, 5),
        },
    }
