"""Closed-loop FloodFillPlanner helpers, CustomObjectMarker localization, ToF peek, and speed-run."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .landmarks import MazeMarkerAnchor, estimate_marker_transform
from .navmap_atlas import (
    MAX_TOF_MM,
    NavMapAtlas,
    SubmapLayer,
    extract_quadtree_leaves,
    observe_world_anchors,
)
from .sdk_session import check_safety, poll_latest_nav_map
from .types_and_math import Pose2D, Transform2D, normalize_angle, pose2d_from_sdk_pose

NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8
ALL_DIRS = (NORTH, EAST, SOUTH, WEST)
DIR_DELTA = {NORTH: (1, 0), EAST: (0, 1), SOUTH: (-1, 0), WEST: (0, -1)}
DELTA_TO_DIR = {delta: d for d, delta in DIR_DELTA.items()}
DIR_THETA = {EAST: 0.0, NORTH: math.pi * 0.5, WEST: math.pi, SOUTH: -math.pi * 0.5}
TURN_RIGHT_OF = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}
TURN_LEFT_OF = {NORTH: WEST, WEST: SOUTH, SOUTH: EAST, EAST: NORTH}
MAX_CLOSED_LOOP_STEPS = 64
MAX_SPEED_SEGMENTS = 64
MAX_CHUNKS_PER_SEGMENT = 32
MAX_SUBMAP_PIVOT_CELLS = 4096
TURN_DEADBAND_DEG = 0.5
MAX_STEER_OFFSET_DEG = 20.0
MAX_LATERAL_CORR_FRAC = 0.35
SPEED_RUN_MAX_CHUNK_MM = 540.0
SPEED_RUN_STANDOFF_MM = 120.0
MARKER_OUTLIER_MM = 120.0
MARKER_OUTLIER_DEG = 15.0
MARKER_MAX_GAIN = 0.12
MARKER_MAX_STEP_MM = 5.0
MARKER_MAX_STEP_DEG = 0.5
MARKER_MIN_GAIN = 0.10
MARKER_CONSENSUS_MM = 60.0
MARKER_CONSENSUS_DEG = 10.0
MARKER_RELOCK_COUNT = 3


@dataclass(frozen=True)
class _FallbackAngle:
    """Lightweight angle holder matching anki_vector.util.Angle interface."""

    degrees: float


@dataclass(frozen=True)
class _FallbackDist:
    """Lightweight distance holder matching anki_vector.util.Distance interface."""

    distance_mm: float


@dataclass(frozen=True)
class _FallbackSpeed:
    """Lightweight speed holder matching anki_vector.util.Speed interface."""

    speed_mmps: float


def _sdk_units() -> Tuple[Any, Any, Any]:
    """Return anki_vector.util.{degrees,distance_mm,speed_mmps} or test-safe fallbacks."""
    try:
        from anki_vector.util import degrees, distance_mm, speed_mmps  # type: ignore

        return (degrees, distance_mm, speed_mmps)
    except ImportError:
        return (lambda v: _FallbackAngle(float(v)), lambda v: _FallbackDist(float(v)), lambda v: _FallbackSpeed(float(v)))


def heading_from_theta(theta_rad: float) -> int:
    """Quantize canonical yaw radians to nearest cardinal MazeGrid heading."""
    assert math.isfinite(theta_rad), "theta_rad must be finite"
    best = min(ALL_DIRS, key=lambda d: abs(normalize_angle(theta_rad - DIR_THETA[d])))
    assert best in ALL_DIRS
    return best


def canonical_to_robot_theta(canonical_rad: float, tf_can: Optional[Transform2D]) -> float:
    """Transform desired canonical yaw into the active robot origin frame."""
    assert math.isfinite(canonical_rad), "canonical_rad must be finite"
    dtheta = float(tf_can.dtheta_rad) if tf_can is not None else 0.0
    out = normalize_angle(canonical_rad - dtheta)
    assert math.isfinite(out)
    return out


def turn_to_canonical_theta(robot: Any, target_canonical_rad: float, tf_can: Optional[Transform2D]) -> float:
    """Rotate robot in place to match canonical yaw, compensating submap rotation."""
    assert robot is not None and math.isfinite(target_canonical_rad)
    degrees_fn, _d, _s = _sdk_units()
    cur = pose2d_from_sdk_pose(robot.pose)
    robot_target = canonical_to_robot_theta(target_canonical_rad, tf_can)
    delta = normalize_angle(robot_target - cur.theta_rad)
    assert math.isfinite(delta)
    if abs(delta) > math.radians(TURN_DEADBAND_DEG):
        robot.behavior.turn_in_place(degrees_fn(math.degrees(delta)))
    return delta


def resolve_submap_id(base_submap_id: str, origin_id: int) -> str:
    """Build per-origin submap identifier sharing canonical session prefix."""
    assert base_submap_id, "base_submap_id required"
    prefix = base_submap_id.split(":origin_", 1)[0]
    assert prefix
    return f"{prefix}:origin_{int(origin_id)}"


def _submap_pivot_mm(submap: SubmapLayer) -> Tuple[float, float]:
    """Compute local-frame centroid (mm) of integrated cells as SE(2) blend pivot."""
    assert submap is not None and submap.cell_mm > 0.0
    if not submap.cells:
        return (0.0, 0.0)
    sx = sy = 0.0
    n = 0
    for ix, iy in submap.cells:
        sx += (ix + 0.5) * submap.cell_mm
        sy += (iy + 0.5) * submap.cell_mm
        n += 1
        if n >= MAX_SUBMAP_PIVOT_CELLS:
            break
    assert n > 0
    return (sx / n, sy / n)


def _is_pose_outlier(tf_prev: Transform2D, tf_cand: Transform2D, rpose: Pose2D) -> bool:
    """Return True when candidate transform jumps beyond outlier position or heading gates."""
    assert tf_prev is not None and tf_cand is not None
    assert rpose is not None
    c_prev = tf_prev.apply_pose(rpose, target_origin_id=0)
    c_cand = tf_cand.apply_pose(rpose, target_origin_id=0)
    pos_err = math.hypot(c_cand.x_mm - c_prev.x_mm, c_cand.y_mm - c_prev.y_mm)
    deg_err = abs(math.degrees(normalize_angle(c_cand.theta_rad - c_prev.theta_rad)))
    return pos_err > MARKER_OUTLIER_MM or deg_err > MARKER_OUTLIER_DEG


def _pivot_blend_se2(
    tf_a: Transform2D, tf_b: Transform2D, px: float, py: float, weight: float, max_step_mm: float = 0.0, max_step_deg: float = 0.0
) -> Transform2D:
    """Blend tf_a toward tf_b around world-map pivot (px, py) with optional per-step caps."""
    assert tf_a is not None and tf_b is not None
    assert 0.0 <= weight <= 1.0
    cx0, cy0 = tf_a.apply_point(px, py)
    cx1, cy1 = tf_b.apply_point(px, py)
    dx, dy = weight * (cx1 - cx0), weight * (cy1 - cy0)
    if max_step_mm > 0.0:
        step = math.hypot(dx, dy)
        if step > max_step_mm and step > 1e-9:
            dx, dy = dx * (max_step_mm / step), dy * (max_step_mm / step)
    dth = weight * normalize_angle(tf_b.dtheta_rad - tf_a.dtheta_rad)
    if max_step_deg > 0.0:
        max_th = math.radians(max_step_deg)
        dth = max(-max_th, min(max_th, dth))
    nt = normalize_angle(tf_a.dtheta_rad + dth)
    c, s = math.cos(nt), math.sin(nt)
    return Transform2D((cx0 + dx) - (c * px - s * py), (cy0 + dy) - (s * px + c * py), nt)


def _advance_outlier_streak(
    streaks: Dict[str, Tuple[Transform2D, int]], sid: str, tf_prev: Transform2D, tf_cand: Transform2D, rpose: Pose2D
) -> Tuple[Transform2D, bool, bool]:
    """Track consecutive mutually-consistent outlier reads and relock once quorum is met."""
    assert streaks is not None and tf_prev is not None
    assert tf_cand is not None and rpose is not None
    prior = streaks.get(sid)
    if prior is not None:
        tf_ref, cnt = prior
        c_ref, c_cand = tf_ref.apply_pose(rpose, target_origin_id=0), tf_cand.apply_pose(rpose, target_origin_id=0)
        if math.hypot(c_cand.x_mm - c_ref.x_mm, c_cand.y_mm - c_ref.y_mm) <= MARKER_CONSENSUS_MM and abs(math.degrees(normalize_angle(c_cand.theta_rad - c_ref.theta_rad))) <= MARKER_CONSENSUS_DEG:
            cnt += 1
            tf_avg = _pivot_blend_se2(tf_ref, tf_cand, rpose.x_mm, rpose.y_mm, 1.0 / float(cnt))
            if cnt >= MARKER_RELOCK_COUNT:
                streaks.pop(sid, None)
                return (tf_avg, True, False)
            streaks[sid] = (tf_avg, cnt)
            return (tf_prev, False, True)
    streaks[sid] = (tf_cand, 1)
    return (tf_prev, False, True)


def _blend_marker_transform(
    tf_prev: Optional[Transform2D], tf_cand: Transform2D, rpose: Pose2D, submap: SubmapLayer, had_prior_frame: bool, obs_count: int, outlier_streaks: Optional[Dict[str, Tuple[Transform2D, int]]] = None, submap_id: str = ""
) -> Tuple[Transform2D, bool, bool]:
    """Reject isolated outliers, relock on consistent clusters, and blend SE(2) with gain floor."""
    assert tf_cand is not None and rpose is not None
    assert submap is not None
    if not had_prior_frame or tf_prev is None:
        if outlier_streaks is not None and submap_id:
            outlier_streaks.pop(submap_id, None)
        return (tf_cand, True, False)
    if _is_pose_outlier(tf_prev, tf_cand, rpose):
        if outlier_streaks is None:
            return (tf_prev, False, True)
        return _advance_outlier_streak(outlier_streaks, submap_id, tf_prev, tf_cand, rpose)
    if outlier_streaks is not None and submap_id:
        outlier_streaks.pop(submap_id, None)
    px, py = _submap_pivot_mm(submap)
    gain = max(MARKER_MIN_GAIN, min(MARKER_MAX_GAIN, 1.0 / float(max(1, obs_count) + 1)))
    return (_pivot_blend_se2(tf_prev, tf_cand, px, py, gain, MARKER_MAX_STEP_MM, MARKER_MAX_STEP_DEG), True, False)


def _apply_localization_result(
    atlas: NavMapAtlas, submap: SubmapLayer, active_sid: str, rpose: Pose2D, tf_m: Optional[Transform2D], had_prior_frame: bool, is_base_sid: bool
) -> Dict[str, Any]:
    """Update submap transform via outlier-gated filter and build canonical pose dict."""
    assert atlas is not None and rpose is not None
    assert active_sid in atlas.submaps
    tf_prev = atlas.transforms_to_canonical.get(active_sid)
    obs_n = atlas.marker_obs_counts.get(active_sid, 0)
    if tf_m is not None:
        was_outlier = bool(had_prior_frame and tf_prev is not None and _is_pose_outlier(tf_prev, tf_m, rpose))
        tf_used, accepted, rejected = _blend_marker_transform(tf_prev, tf_m, rpose, submap, had_prior_frame, obs_n, atlas.marker_outlier_streaks, active_sid)
        relocked = bool(accepted and was_outlier)
        if accepted:
            atlas.transforms_to_canonical[active_sid] = tf_used
            atlas.marker_obs_counts[active_sid] = MARKER_RELOCK_COUNT if relocked else (obs_n + 1)
        cpose = tf_used.apply_pose(rpose, target_origin_id=0)
        return {"submap_id": active_sid, "origin_id": rpose.origin_id, "rpose": rpose, "cpose": cpose, "tf_can": tf_used, "relocalized_via": "custom_marker" if accepted else None, "localized": True, "marker_rejected": rejected, "marker_relocked": relocked}
    cpose = tf_prev.apply_pose(rpose, target_origin_id=0) if tf_prev else rpose
    return {"submap_id": active_sid, "origin_id": rpose.origin_id, "rpose": rpose, "cpose": cpose, "tf_can": tf_prev, "relocalized_via": None, "localized": tf_prev is not None or is_base_sid, "marker_rejected": False, "marker_relocked": False}


def fuse_and_localize_step(
    robot: Any,
    atlas: NavMapAtlas,
    base_submap_id: str,
    marker_map: Optional[Dict[str, MazeMarkerAnchor]] = None,
) -> Dict[str, Any]:
    """Fuse ToF + NavMap + world/marker anchors and return current canonical pose."""
    assert robot is not None and atlas is not None
    rpose = pose2d_from_sdk_pose(robot.pose)
    active_sid = resolve_submap_id(base_submap_id, rpose.origin_id)
    had_prior_obs = (active_sid in atlas.submaps and bool(atlas.submaps[active_sid].cells)) or atlas.marker_obs_counts.get(active_sid, 0) > 0
    had_prior_frame = (active_sid in atlas.transforms_to_canonical) and had_prior_obs
    submap = atlas.ensure_submap(active_sid, rpose.origin_id)
    _ok, _r, tel = check_safety(robot, allow_on_charger=True)
    submap.integrate_tof_ray(rpose, float(tel["prox_mm"]), bool(tel["prox_found"]), bool(tel["prox_lift_in_fov"]))
    nm = poll_latest_nav_map(robot, max_wait_s=0.25)
    if nm is not None:
        submap.integrate_leaves(extract_quadtree_leaves(nm))
    anchors = observe_world_anchors(robot)
    for aid, apose in anchors.items():
        atlas.record_anchor(active_sid, aid, apose)
    atlas.resolve_registrations()
    tf_m = estimate_marker_transform(anchors, marker_map, rpose.origin_id) if marker_map else None
    assert active_sid in atlas.submaps
    return _apply_localization_result(atlas, submap, active_sid, rpose, tf_m, had_prior_frame, active_sid == base_submap_id)


def peek_unknown_edges_tof(
    robot: Any,
    atlas: NavMapAtlas,
    grid: Any,
    curr: Tuple[int, int],
    heading: int,
    base_submap_id: str,
    cell_size_mm: float,
    marker_map: Optional[Dict[str, MazeMarkerAnchor]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Measure unvisited front/left/right cell edges with short 0/+-90deg ToF peeks."""
    assert robot is not None and grid.in_bounds(curr[0], curr[1])
    events: List[Dict[str, Any]] = []
    last_loc = fuse_and_localize_step(robot, atlas, base_submap_id, marker_map=marker_map)
    for d in (heading, TURN_LEFT_OF[heading], TURN_RIGHT_OF[heading]):
        if grid.known[curr[0]][curr[1]] & d:
            continue
        turn_to_canonical_theta(robot, DIR_THETA[d], last_loc["tf_can"])
        ok, reason, tel = check_safety(robot, allow_on_charger=False)
        if not ok:
            return (reason, events, last_loc)
        wall = bool(tel["prox_found"] and not tel["prox_lift_in_fov"] and tel["prox_mm"] < cell_size_mm * 0.85)
        grid.set_wall(curr[0], curr[1], d, wall)
        last_loc = fuse_and_localize_step(robot, atlas, base_submap_id, marker_map=marker_map)
        events.append({"event": "peek_edge", "cell": list(curr), "dir": d, "wall": wall, "prox_mm": tel["prox_mm"]})
    return (None, events, last_loc)


def coalesce_collinear_runs(path: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Compress discrete cell path into maximal straight collinear segments."""
    if len(path) < 2:
        return []
    assert len(path) <= MAX_CLOSED_LOOP_STEPS * 4, "path exceeds bounded segment count"
    segments: List[Dict[str, Any]] = []
    for i in range(len(path) - 1):
        delta = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        d = DELTA_TO_DIR.get(delta)
        assert d is not None, f"non-adjacent cells in path: {path[i]} -> {path[i + 1]}"
        if segments and segments[-1]["dir"] == d:
            segments[-1]["cells"] += 1
            segments[-1]["end"] = list(path[i + 1])
        else:
            segments.append({"dir": d, "cells": 1, "start": list(path[i]), "end": list(path[i + 1])})
    return segments


def _handle_origin_transition(
    prev_oid: int, loc: Dict[str, Any], step_idx: int, trace: List[Dict[str, Any]]
) -> Optional[str]:
    """Record custom-marker relocalization or halt safely on blind origin jump."""
    assert "origin_id" in loc and "localized" in loc
    new_oid = int(loc["origin_id"])
    if new_oid == prev_oid:
        return None
    if loc["relocalized_via"] == "custom_marker":
        trace.append({"step": step_idx, "event": "relocalized_marker", "old_origin_id": prev_oid, "new_origin_id": new_oid, "cpose": loc["cpose"].to_dict()})
        return None
    if not loc["localized"]:
        return "unlocalized_origin_change"
    trace.append({"step": step_idx, "event": "relocalized_anchor", "old_origin_id": prev_oid, "new_origin_id": new_oid})
    return None


def _compute_steered_step(
    cpose: Pose2D,
    grid: Any,
    curr: Tuple[int, int],
    nd: int,
    cell_size_mm: float,
    front_prox_mm: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute canonical steer angle and drive distance toward next cell center with ToF trim."""
    assert cpose is not None and nd in DIR_DELTA
    assert cell_size_mm > 0.0 and grid.in_bounds(curr[0], curr[1])
    dr, dc = DIR_DELTA[nd]
    tx_mm, ty_mm = grid.cell_center_mm(curr[0] + dr, curr[1] + dc)
    dx, dy = tx_mm - cpose.x_mm, ty_mm - cpose.y_mm
    ux, uy = math.cos(DIR_THETA[nd]), math.sin(DIR_THETA[nd])
    lx, ly = -uy, ux
    along = max(0.25 * cell_size_mm, min(1.5 * cell_size_mm, dx * ux + dy * uy))
    max_lat = MAX_LATERAL_CORR_FRAC * cell_size_mm
    lat = max(-max_lat, min(max_lat, dx * lx + dy * ly))
    max_off = math.radians(MAX_STEER_OFFSET_DEG)
    steer_off = max(-max_off, min(max_off, math.atan2(lat, along)))
    if front_prox_mm is not None and 0.85 * cell_size_mm <= front_prox_mm <= 1.85 * cell_size_mm:
        along = min(along, max(0.25 * cell_size_mm, float(front_prox_mm) - 0.5 * cell_size_mm))
    return (normalize_angle(DIR_THETA[nd] + steer_off), along / max(0.5, math.cos(steer_off)))


def _drive_and_update_cell(
    robot: Any,
    atlas: NavMapAtlas,
    grid: Any,
    base_submap_id: str,
    curr: Tuple[int, int],
    goal: Tuple[int, int],
    nd: int,
    cell_size_mm: float,
    marker_map: Optional[Dict[str, MazeMarkerAnchor]],
    state: Dict[str, Any],
    idx: int,
    front_prox_mm: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Steer toward next cell center, fuse localization, and detect slip or arrival."""
    _d, dist_fn, speed_fn = _sdk_units()
    assert grid.in_bounds(curr[0], curr[1]) and cell_size_mm > 0.0
    assert "loc" in state and state["loc"] is not None
    loc = state["loc"]
    steer_theta, drive_dist = _compute_steered_step(loc["cpose"], grid, curr, nd, cell_size_mm, front_prox_mm)
    turn_to_canonical_theta(robot, steer_theta, loc["tf_can"])
    robot.behavior.drive_straight(dist_fn(drive_dist), speed_fn(65.0))
    new_loc = fuse_and_localize_step(robot, atlas, base_submap_id, marker_map=marker_map)
    err = _handle_origin_transition(state["origin_id"], new_loc, idx, state["trace"])
    state["loc"], state["origin_id"] = new_loc, int(new_loc["origin_id"])
    if err is not None:
        return {"status": "halted", "reason": err, "steps": len(state["trace"]), "trace": state["trace"]}
    after_cell = grid.world_to_cell(new_loc["cpose"].x_mm, new_loc["cpose"].y_mm)
    state["trace"].append({"step": idx, "event": "moved", "cell": list(after_cell), "dir": nd, "slipped": after_cell == curr})
    if after_cell == goal:
        return {"status": "reached", "goal": list(goal), "curr": list(after_cell), "steps": len(state["trace"]), "trace": state["trace"]}
    return None


def _run_floodfill_step(
    robot: Any, atlas: NavMapAtlas, grid: Any, planner: Any, base_submap_id: str,
    goal: Tuple[int, int], cell_size_mm: float, marker_map: Optional[Dict[str, MazeMarkerAnchor]],
    peek_unknown: bool, state: Dict[str, Any], idx: int,
) -> Optional[Dict[str, Any]]:
    """Execute one closed-loop perceive-plan-act step and return terminal dict if done."""
    assert robot is not None and grid is not None
    assert planner is not None and "loc" in state
    loc = state["loc"]
    curr = grid.world_to_cell(loc["cpose"].x_mm, loc["cpose"].y_mm)
    heading = heading_from_theta(loc["cpose"].theta_rad)
    if curr == goal:
        return {"status": "reached", "goal": list(goal), "curr": list(curr), "steps": len(state["trace"]), "trace": state["trace"]}
    if peek_unknown:
        reason, pevs, loc = peek_unknown_edges_tof(robot, atlas, grid, curr, heading, base_submap_id, cell_size_mm, marker_map)
        state["loc"] = loc
        state["trace"].extend(pevs)
        if reason is not None:
            return {"status": "halted", "reason": reason, "steps": len(state["trace"]), "trace": state["trace"]}
        curr = grid.world_to_cell(loc["cpose"].x_mm, loc["cpose"].y_mm)
        heading = heading_from_theta(loc["cpose"].theta_rad)
    nd = planner.next_direction(curr, [goal], heading=heading)
    if nd is None:
        return {"status": "unreachable", "curr": list(curr), "steps": len(state["trace"]), "trace": state["trace"]}
    turn_to_canonical_theta(robot, DIR_THETA[nd], loc["tf_can"])
    ok, reason, tel = check_safety(robot, allow_on_charger=False)
    if not ok:
        return {"status": "halted", "reason": reason, "steps": len(state["trace"]), "trace": state["trace"]}
    if tel["prox_found"] and not tel["prox_lift_in_fov"] and tel["prox_mm"] < cell_size_mm * 0.85:
        grid.set_wall(curr[0], curr[1], nd, True)
        state["loc"] = fuse_and_localize_step(robot, atlas, base_submap_id, marker_map=marker_map)
        state["trace"].append({"step": idx, "event": "wall_discovered", "cell": list(curr), "dir": nd, "prox_mm": tel["prox_mm"]})
        return None
    front_mm = float(tel["prox_mm"]) if (tel["prox_found"] and not tel["prox_lift_in_fov"]) else None
    return _drive_and_update_cell(robot, atlas, grid, base_submap_id, curr, goal, nd, cell_size_mm, marker_map, state, idx, front_mm)


def run_floodfill_loop(
    robot: Any, atlas: NavMapAtlas, grid: Any, planner: Any, base_submap_id: str,
    goal: Tuple[int, int], max_steps: int, cell_size_mm: float,
    marker_map: Optional[Dict[str, MazeMarkerAnchor]], peek_unknown: bool,
) -> Dict[str, Any]:
    """Iterate bounded closed-loop flood-fill steps until reached, halted, or capped."""
    assert robot is not None and grid.in_bounds(goal[0], goal[1])
    assert int(max_steps) >= 1 and cell_size_mm > 0.0
    init_loc = fuse_and_localize_step(robot, atlas, base_submap_id, marker_map=marker_map)
    state: Dict[str, Any] = {"loc": init_loc, "origin_id": int(init_loc["origin_id"]), "trace": []}
    for idx in range(max(1, min(int(max_steps), MAX_CLOSED_LOOP_STEPS))):
        done = _run_floodfill_step(robot, atlas, grid, planner, base_submap_id, goal, cell_size_mm, marker_map, peek_unknown, state, idx)
        if done is not None:
            return done
    curr = grid.world_to_cell(state["loc"]["cpose"].x_mm, state["loc"]["cpose"].y_mm)
    return {"status": "completed", "curr": list(curr), "steps": len(state["trace"]), "trace": state["trace"]}


def _compute_corridor_chunk_drive(
    rem_mm: float, cell_size_mm: float, tel: Dict[str, Any]
) -> Tuple[float, Optional[str]]:
    """Compute ToF-capped chunk drive distance and optional unexpected-obstacle halt reason."""
    assert math.isfinite(rem_mm) and cell_size_mm > 0.0
    assert isinstance(tel, dict)
    if rem_mm <= 0.25 * cell_size_mm:
        return (0.0, None)
    max_chunk = min(SPEED_RUN_MAX_CHUNK_MM, MAX_TOF_MM, rem_mm)
    prox_mm = float(tel.get("prox_mm", 9999.0))
    obstacle_seen = bool(tel.get("prox_found")) and not bool(tel.get("prox_lift_in_fov")) and prox_mm < rem_mm + 0.20 * cell_size_mm
    if not obstacle_seen:
        return (max_chunk, None)
    safe_cells = int(math.floor(max(0.0, prox_mm - SPEED_RUN_STANDOFF_MM) / cell_size_mm))
    safe_mm = min(max_chunk, float(safe_cells) * cell_size_mm)
    if safe_mm < 0.5 * cell_size_mm:
        return (0.0, "unexpected_obstacle")
    return (safe_mm, None)


def _execute_single_speed_segment(
    robot: Any, atlas: NavMapAtlas, grid: Any, submap_id: str,
    seg: Dict[str, Any], idx: int, cell_size_mm: float,
    marker_map: Optional[Dict[str, MazeMarkerAnchor]], loc: Dict[str, Any], trace: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Drive one collinear segment in <=540 mm ToF-rechecked chunks, halting short of obstacles."""
    _d, dist_fn, speed_fn = _sdk_units()
    assert seg["dir"] in DIR_THETA and int(seg["cells"]) >= 1
    assert math.isfinite(cell_size_mm) and cell_size_mm > 0.0
    seg_dir, done_mm = seg["dir"], 0.0
    ex, ey = grid.cell_center_mm(seg["end"][0], seg["end"][1])
    ux, uy = math.cos(DIR_THETA[seg_dir]), math.sin(DIR_THETA[seg_dir])
    for _chunk in range(MAX_CHUNKS_PER_SEGMENT):
        rem_mm = (ex - loc["cpose"].x_mm) * ux + (ey - loc["cpose"].y_mm) * uy
        if rem_mm <= 0.25 * cell_size_mm:
            break
        steer_off = max(-math.radians(MAX_STEER_OFFSET_DEG), min(math.radians(MAX_STEER_OFFSET_DEG), normalize_angle(math.atan2(ey - loc["cpose"].y_mm, ex - loc["cpose"].x_mm) - DIR_THETA[seg_dir])))
        turn_to_canonical_theta(robot, normalize_angle(DIR_THETA[seg_dir] + steer_off), loc["tf_can"])
        ok, reason, tel = check_safety(robot, allow_on_charger=False)
        if not ok:
            return (loc, {"status": "halted", "reason": reason, "segments": trace})
        drive_mm, halt_reason = _compute_corridor_chunk_drive(rem_mm, cell_size_mm, tel)
        if halt_reason is not None:
            curr = grid.world_to_cell(loc["cpose"].x_mm, loc["cpose"].y_mm)
            grid.set_wall(curr[0], curr[1], seg_dir, True)
            loc = fuse_and_localize_step(robot, atlas, submap_id, marker_map=marker_map)
            return (loc, {"status": "halted", "reason": halt_reason, "prox_mm": tel["prox_mm"], "segments": trace})
        robot.behavior.drive_straight(dist_fn(drive_mm), speed_fn(100.0))
        done_mm += drive_mm
        loc = fuse_and_localize_step(robot, atlas, submap_id, marker_map=marker_map)
    curr = grid.world_to_cell(loc["cpose"].x_mm, loc["cpose"].y_mm)
    trace.append({"segment": idx, "dir": seg_dir, "cells": seg["cells"], "dist_mm": done_mm, "cell_after": list(curr)})
    return (loc, None)


def execute_speed_run_segments(
    robot: Any, atlas: NavMapAtlas, grid: Any, planner: Any, submap_id: str,
    goal_cell: Optional[Tuple[int, int]] = None, cell_size_mm: float = 180.0,
    marker_map: Optional[Dict[str, MazeMarkerAnchor]] = None,
) -> Dict[str, Any]:
    """Plan shortest known-only path, coalesce collinear segments, and execute chunked speed run."""
    assert robot is not None and cell_size_mm > 0.0
    assert grid is not None and planner is not None
    loc = fuse_and_localize_step(robot, atlas, submap_id, marker_map=marker_map)
    curr = grid.world_to_cell(loc["cpose"].x_mm, loc["cpose"].y_mm)
    goal = goal_cell or (grid.rows - 1, grid.cols - 1)
    path = planner.plan_path(curr, [goal], initial_heading=heading_from_theta(loc["cpose"].theta_rad), known_only=True)
    if not path:
        return {"status": "unreachable", "curr": list(curr), "goal": list(goal), "segments": []}
    segs, trace = coalesce_collinear_runs(path), []
    for idx, seg in enumerate(segs[:MAX_SPEED_SEGMENTS]):
        loc, halted = _execute_single_speed_segment(robot, atlas, grid, submap_id, seg, idx, cell_size_mm, marker_map, loc, trace)
        if halted is not None:
            return halted
    curr = grid.world_to_cell(loc["cpose"].x_mm, loc["cpose"].y_mm)
    return {"status": "reached" if curr == goal else "completed", "curr": list(curr), "goal": list(goal), "segment_count": len(trace), "segments": trace}
