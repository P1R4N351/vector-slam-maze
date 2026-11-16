"""Command-line interface for Vector persistent SLAM atlas and maze solving."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .atlas_lock import AtlasLockTimeoutError, atlas_file_lock
from .cert_onboard import DEFAULT_BOT_SDK_INFO, DEFAULT_SDK_DIR, DEFAULT_SESSION_CERTS_DIR, cmd_onboard_cert
from .landmarks import load_marker_map, register_custom_wall_markers
from .maze_solver import (
    EAST,
    FloodFillPlanner,
    MazeGrid,
    execute_floodfill_run,
    execute_speed_run,
    execute_wall_follow_run,
)
from .movement_directives import (
    DEFAULT_FORWARD_STANDOFF_MM,
    execute_bounded_directive,
    execute_directive_batch,
    plan_directive,
    plan_directive_batch,
)
from .navmap_atlas import (
    NavMapAtlas,
    compress_occupied_to_boxes,
    extract_quadtree_leaves,
    observe_world_anchors,
    rehydrate_onboard_obstacles,
)
from .rosbag_export import cmd_export_rosbag
from .sdk_session import DEFAULT_SERIAL, VectorSDKSession, check_safety, poll_latest_nav_map
from .types_and_math import normalize_angle, pose2d_from_sdk_pose

DEFAULT_ATLAS = Path(os.environ.get("VECTOR_ATLAS_PATH", "/home/pi/maps/demo-persistent-atlas.json"))
DEFAULT_EXPORT_DIR = Path(os.environ.get("VECTOR_EXPORT_DIR", "/home/pi/maps/demo-ros"))
MAX_SCAN_STEPS = 36


def _session_submap_id(session_tag: str, origin_id: int) -> str:
    """Build stable submap key combining session tag and firmware origin_id."""
    return f"{session_tag}:origin_{origin_id}"


def _sample_robot_into_atlas(
    robot: Any, atlas: NavMapAtlas, session_tag: str, assume_canonical: bool = False
) -> Dict[str, Any]:
    """Capture pose, ToF ray, NavMapGrid leaves, and anchors into atlas."""
    pose = pose2d_from_sdk_pose(robot.pose)
    sid = _session_submap_id(session_tag, pose.origin_id)
    submap = atlas.ensure_submap(sid, pose.origin_id)
    if assume_canonical:
        atlas.assume_same_frame_as_canonical(sid)
    _ok, _reason, tel = check_safety(robot, allow_on_charger=True)
    nm = poll_latest_nav_map(robot, max_wait_s=2.0)
    leaves = extract_quadtree_leaves(nm)
    leaf_hits = submap.integrate_leaves(leaves)
    tof_hits = submap.integrate_tof_ray(pose, float(tel["prox_mm"]), bool(tel["prox_found"]), bool(tel["prox_lift_in_fov"]))
    anchors = observe_world_anchors(robot)
    for aid, apose in anchors.items():
        atlas.record_anchor(sid, aid, apose, assume_canonical=assume_canonical)
    return {
        "submap_id": sid,
        "origin_id": pose.origin_id,
        "leaves_read": len(leaves),
        "leaf_cell_updates": leaf_hits,
        "tof_cell_updates": tof_hits,
        "anchors": {k: v.to_dict() for k, v in anchors.items()},
        "registered": sid in atlas.transforms_to_canonical,
        "telemetry": tel,
    }


def _default_session_tag(prefix: str) -> str:
    """Build PID-disambiguated default session tag across concurrent CLI runs."""
    assert prefix, "prefix required"
    return f"{prefix}-{int(time.time())}-{os.getpid()}"


def _load_atlas_snapshot(atlas_path: Path) -> Tuple[NavMapAtlas, NavMapAtlas]:
    """Load baseline and working NavMapAtlas copies under a brief advisory lock."""
    assert atlas_path is not None
    with atlas_file_lock(atlas_path):
        return (NavMapAtlas.load(atlas_path), NavMapAtlas.load(atlas_path))


def _commit_session_atlas(
    atlas_path: Path,
    export_dir: Optional[Path],
    baseline: NavMapAtlas,
    working: NavMapAtlas,
    explicit_tf_sids: Set[str],
    session_rec: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Reload disk atlas under short lock, merge working deltas, save, and export."""
    assert atlas_path is not None and isinstance(working, NavMapAtlas)
    recs = (session_rec,) if session_rec is not None else ()
    with atlas_file_lock(atlas_path):
        fresh = NavMapAtlas.load(atlas_path)
        fresh.merge_session_from(working, baseline=baseline, explicit_tf_sids=explicit_tf_sids, new_sessions=recs)
        fresh.save(atlas_path)
        return fresh.export_ros_pgm_yaml(export_dir / "grid.pgm", export_dir / "grid.yaml") if export_dir else None


def cmd_probe(args: argparse.Namespace) -> int:
    """Passive robot state + anchor + NavMapGrid check (zero motion)."""
    with VectorSDKSession(serial=args.serial, active=False, enable_nav_map=True) as robot:
        _ok, reason, tel = check_safety(robot, allow_on_charger=True)
        nm = poll_latest_nav_map(robot, max_wait_s=float(args.wait_s))
        leaves = extract_quadtree_leaves(nm)
        anchors = observe_world_anchors(robot)
        out = {
            "ok": True,
            "serial": args.serial,
            "safety": reason,
            "telemetry": tel,
            "nav_map": {"present": nm is not None, "size_mm": getattr(nm, "size", None), "leaf_count": len(leaves)},
            "anchors": {k: v.to_dict() for k, v in anchors.items()},
        }
        print(json.dumps(out, indent=2))
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Read live NavMapGrid + anchors lock-free, then merge-save atlas under short lock."""
    atlas_path = Path(args.atlas)
    export_dir = Path(args.export_dir)
    tag = args.session_id or _default_session_tag("snap")
    baseline, working = _load_atlas_snapshot(atlas_path)
    with VectorSDKSession(serial=args.serial, active=False, enable_nav_map=True) as robot:
        obs = _sample_robot_into_atlas(robot, working, tag, assume_canonical=args.assume_docked_canonical)
    explicit = {obs["submap_id"]} if args.assume_docked_canonical else set()
    rec = {"tag": tag, "kind": "snapshot", "ts": int(time.time()), "origin_id": obs["origin_id"], "submap_id": obs["submap_id"]}
    stats = _commit_session_atlas(atlas_path, export_dir, baseline, working, explicit, rec)
    assert stats is not None
    print(json.dumps({"ok": True, "atlas_path": str(atlas_path), "export_dir": str(export_dir), "observation": obs, "map_stats": stats}, indent=2))
    return 0


def _prepare_robot_for_scan(robot: Any, undock_if_needed: bool) -> Tuple[bool, str]:
    """Undock if authorized and set level head + lowered lift for clean ToF/camera view."""
    from anki_vector.util import degrees  # type: ignore

    ok, reason, tel = check_safety(robot, allow_on_charger=True)
    if not ok:
        return (False, reason)
    if tel["is_on_charger"]:
        if not undock_if_needed:
            return (False, "on_charger")
        robot.behavior.drive_off_charger()
    robot.behavior.set_lift_height(0.0)
    robot.behavior.set_head_angle(degrees(0.0))
    register_custom_wall_markers(robot)
    return (True, "ready")


def _try_return_charger(robot: Any) -> str:
    """Turn toward charger if known and drive onto charger safely."""
    from anki_vector.util import radians  # type: ignore

    if getattr(getattr(robot, "status", None), "is_on_charger", False):
        return "already_on_charger"
    world = getattr(robot, "world", None)
    charger = getattr(world, "charger", None) if world else None
    if charger is not None and getattr(charger, "pose", None) is not None and getattr(robot, "pose", None) is not None:
        bp = pose2d_from_sdk_pose(robot.pose)
        cp = pose2d_from_sdk_pose(charger.pose)
        if bp.origin_id == cp.origin_id:
            dth = normalize_angle(math.atan2(cp.y_mm - bp.y_mm, cp.x_mm - bp.x_mm) - bp.theta_rad)
            robot.behavior.turn_in_place(radians(dth))
    try:
        robot.behavior.drive_on_charger()
    except Exception as exc:
        return f"dock_failed:{exc}"
    return "docked" if getattr(getattr(robot, "status", None), "is_on_charger", False) else "dock_attempted"


def _run_scan_steps(
    robot: Any, working: NavMapAtlas, tag: str, steps: int, step_deg: float, assume_canonical: bool
) -> Tuple[int, List[Dict[str, Any]], str]:
    """Execute bounded 360-degree scan turns without holding any file lock."""
    from anki_vector.util import degrees  # type: ignore

    samples: List[Dict[str, Any]] = []
    for idx in range(steps):
        ok, hreason, _ = check_safety(robot, allow_on_charger=False)
        if not ok:
            return (6, samples, hreason)
        samples.append(_sample_robot_into_atlas(robot, working, tag, assume_canonical=assume_canonical))
        if idx + 1 < steps:
            robot.behavior.turn_in_place(degrees(float(step_deg)))
    return (0, samples, "ok")


def cmd_scan(args: argparse.Namespace) -> int:
    """Active 360-degree ToF + NavMapGrid + landmark scan (--via-vector required)."""
    atlas_path, export_dir = Path(args.atlas), Path(args.export_dir)
    tag = args.session_id or _default_session_tag("scan")
    steps = max(1, min(int(args.steps), MAX_SCAN_STEPS))
    dock_note = "not_requested"
    baseline, working = _load_atlas_snapshot(atlas_path)
    with VectorSDKSession(serial=args.serial, active=True, via_vector=args.via_vector, enable_nav_map=True) as robot:
        ready, r_reason = _prepare_robot_for_scan(robot, args.undock_if_needed)
        if not ready:
            print(json.dumps({"ok": False, "error": r_reason}))
            return 3
        rc_step, step_samples, fail_reason = _run_scan_steps(robot, working, tag, steps, float(args.step_deg), bool(args.assume_docked_canonical))
        if rc_step != 0:
            print(json.dumps({"ok": False, "aborted_step": len(step_samples), "reason": fail_reason}))
            return rc_step
        if args.return_charger:
            dock_note = _try_return_charger(robot)
    last = step_samples[-1]
    explicit = {s["submap_id"] for s in step_samples} if args.assume_docked_canonical else set()
    rec = {"tag": tag, "kind": "scan", "ts": int(time.time()), "origin_id": last["origin_id"], "submap_id": last["submap_id"]}
    stats = _commit_session_atlas(atlas_path, export_dir, baseline, working, explicit, rec)
    print(json.dumps({"ok": True, "steps_done": len(step_samples), "dock_result": dock_note, "last_observation": last, "map_stats": stats}, indent=2))
    return 0


def _create_test_virtual_box(robot: Any) -> bool:
    """Place a temporary 100x40x80 mm fixed box 220 mm ahead to probe NavMap injection."""
    from anki_vector.util import Pose, degrees  # type: ignore

    pose = Pose(220.0, 0.0, 40.0, angle_z=degrees(0.0))
    obj = robot.world.create_custom_fixed_object(
        pose,
        x_size_mm=100.0,
        y_size_mm=40.0,
        z_size_mm=80.0,
        relative_to_robot=True,
        use_robot_origin=True,
    )
    return obj is not None


def cmd_rehydrate(args: argparse.Namespace) -> int:
    """Inject persisted obstacles into onboard origin without holding disk lock during robot IO."""
    atlas_path = Path(args.atlas)
    tag = args.session_id or _default_session_tag("rehydrate")
    _baseline, working = _load_atlas_snapshot(atlas_path)
    with VectorSDKSession(serial=args.serial, active=True, via_vector=args.via_vector, enable_nav_map=True) as robot:
        obs = _sample_robot_into_atlas(robot, working, tag, assume_canonical=args.assume_docked_canonical)
        sid = obs["submap_id"]
        before_occ = _count_occ_leaves(extract_quadtree_leaves(poll_latest_nav_map(robot, max_wait_s=1.0)))
        created = rehydrate_onboard_obstacles(robot, working, sid, max_boxes=int(args.max_boxes))
        test_box_created = _create_test_virtual_box(robot) if args.test_box else False
        time.sleep(1.2)
        after_leaves = extract_quadtree_leaves(poll_latest_nav_map(robot, max_wait_s=1.5))
        after_occ = _count_occ_leaves(after_leaves)
        if args.cleanup:
            robot.world.delete_custom_objects(delete_custom_marker_objects=False, delete_fixed_custom_objects=True, delete_custom_object_archetypes=False)
    print(json.dumps({"ok": True, "submap_id": sid, "fixed_boxes_created": created, "test_box_created": test_box_created, "before_occ_leaves": before_occ, "after_occ_leaves": after_occ, "total_leaves_after": len(after_leaves)}, indent=2))
    return 0


def _count_occ_leaves(leaves: list) -> int:
    """Count obstacle leaves (types 3..7) in a leaf list."""
    return sum(1 for lf in leaves if int(lf.content) in (3, 4, 5, 6, 7))


def cmd_plan_maze(args: argparse.Namespace) -> int:
    """Run offline UK MarsBot flood-fill solver over a persisted NavMapAtlas."""
    atlas = NavMapAtlas.load(Path(args.atlas))
    grid = MazeGrid.from_atlas(atlas, cell_size_mm=float(args.cell_mm))
    planner = FloodFillPlanner(grid)
    start = grid.world_to_cell(float(args.start_x_mm), float(args.start_y_mm))
    goal = (int(args.goal_row), int(args.goal_col)) if args.goal_row is not None and args.goal_col is not None else (grid.rows - 1, grid.cols - 1)
    path = planner.plan_path(start, [goal], initial_heading=EAST)
    out = {
        "ok": True,
        "rows": grid.rows,
        "cols": grid.cols,
        "cell_mm": grid.cell_size_mm,
        "start": list(start),
        "goal": list(goal),
        "path_len": len(path),
        "path": [list(p) for p in path],
        "obstacle_boxes": len(compress_occupied_to_boxes(atlas)),
        "ascii": grid.render_ascii(path=path, start=start),
    }
    print(json.dumps(out, indent=2))
    return 0


def _dispatch_solve_mode(robot: Any, atlas: NavMapAtlas, sid: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Run selected maze solver mode against live robot and atlas."""
    assert robot is not None and sid, "robot and submap_id required"
    markers = load_marker_map(Path(args.marker_map)) if args.marker_map else None
    goal = (int(args.goal_row), int(args.goal_col)) if args.goal_row is not None and args.goal_col is not None else None
    if args.mode == "wall-follow":
        return execute_wall_follow_run(robot, atlas, sid, max_steps=int(args.max_steps), step_mm=float(args.cell_mm))
    if args.mode == "speed-run":
        return execute_speed_run(robot, atlas, sid, goal_cell=goal, cell_size_mm=float(args.cell_mm), marker_map=markers)
    return execute_floodfill_run(robot, atlas, sid, goal_cell=goal, max_steps=int(args.max_steps), cell_size_mm=float(args.cell_mm), marker_map=markers, peek_unknown=bool(args.peek_unknown), rehydrate_before_solve=bool(args.rehydrate_before_solve), two_pass=bool(args.two_pass))


def cmd_solve_maze(args: argparse.Namespace) -> int:
    """Execute bounded live maze solver lock-free, then merge session submaps under short lock."""
    atlas_path, export_dir = Path(args.atlas), Path(args.export_dir)
    tag = args.session_id or _default_session_tag("maze")
    baseline, working = _load_atlas_snapshot(atlas_path)
    pre_tf_sids = set(working.transforms_to_canonical.keys())
    with VectorSDKSession(serial=args.serial, active=True, via_vector=args.via_vector, enable_nav_map=True) as robot:
        ready, r_reason = _prepare_robot_for_scan(robot, args.undock_if_needed)
        if not ready:
            print(json.dumps({"ok": False, "error": r_reason}))
            return 3
        sid = _session_submap_id(tag, pose2d_from_sdk_pose(robot.pose).origin_id)
        if args.assume_docked_canonical:
            working.assume_same_frame_as_canonical(sid)
        res = _dispatch_solve_mode(robot, working, sid, args)
    explicit = {sid} if args.assume_docked_canonical else set()
    if args.marker_map:
        explicit.update(k for k in working.transforms_to_canonical if k not in pre_tf_sids)
    rec = {"tag": tag, "kind": f"maze-{args.mode}", "ts": int(time.time()), "submap_id": sid, "status": res.get("status")}
    stats = _commit_session_atlas(atlas_path, export_dir, baseline, working, explicit, rec)
    print(json.dumps({"ok": True, "mode": args.mode, "result": res, "map_stats": stats}, indent=2))
    return 0


def _plan_act_dry_run(verb: str, payload: Dict[str, Any], prox_min_mm: float, strict: bool) -> Dict[str, Any]:
    """Validate and decompose act verb offline without opening a robot session."""
    assert isinstance(verb, str) and isinstance(payload, dict)
    assert prox_min_mm >= 0.0
    if verb == "directives":
        seq = payload.get("directives") if isinstance(payload.get("directives"), list) else []
        return plan_directive_batch(seq, strict=strict, min_prox_mm=prox_min_mm)
    return plan_directive(verb, payload, strict=strict, min_prox_mm=prox_min_mm)


def cmd_act(args: argparse.Namespace) -> int:
    """Bounded actuator verb (move, rotate, translate, head, lift, directives, dock, halt)."""
    assert args is not None and isinstance(args.verb, str)
    verb = args.verb
    try:
        payload = json.loads(args.args_json or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "stopped_reason": "invalid_json", "error": str(exc)}))
        return 2
    if not isinstance(payload, dict):
        print(json.dumps({"ok": False, "stopped_reason": "invalid_payload", "error": "args_json must be a JSON object"}))
        return 2
    strict = bool(getattr(args, "strict", False))
    if bool(getattr(args, "dry_run", False)) or payload.get("dry_run") is True:
        plan = _plan_act_dry_run(verb, payload, float(args.prox_min_mm), strict)
        print(json.dumps(plan))
        return 0 if plan.get("ok") else 6
    if verb == "halt":
        with VectorSDKSession(serial=args.serial, active=args.via_vector, via_vector=args.via_vector, enable_nav_map=False) as robot:
            robot.motors.stop_all_motors()
        print(json.dumps({"ok": True, "stopped_reason": "halt"}))
        return 0
    with VectorSDKSession(serial=args.serial, active=True, via_vector=args.via_vector, enable_nav_map=False) as robot:
        res = _run_act_motion(robot, verb, payload, float(args.prox_min_mm), strict=strict)
    print(json.dumps(res))
    return 0 if res.get("ok") else 6


def _run_act_motion(robot: Any, verb: str, payload: Dict[str, Any], prox_min_mm: float, strict: bool = False) -> Dict[str, Any]:
    """Execute bounded actuator verb or directive batch safely."""
    assert robot is not None and isinstance(verb, str)
    assert isinstance(payload, dict) and prox_min_mm >= 0.0
    if verb == "dock":
        _ok, _r, pre = check_safety(robot, allow_on_charger=True, min_prox_mm=prox_min_mm, check_front_prox=False)
        return {"ok": True, "stopped_reason": _try_return_charger(robot), "prox_mm": pre["prox_mm"]}
    if verb == "directives":
        seq = payload.get("directives") if isinstance(payload.get("directives"), list) else []
        return execute_directive_batch(robot, seq, min_prox_mm=prox_min_mm, strict=strict)
    return execute_bounded_directive(robot, verb, payload, min_prox_mm=prox_min_mm, strict=strict)


def _translate_payload_from_args(args: argparse.Namespace) -> Tuple[str, Any]:
    """Build ('directives', list) or ('translate', dict) from translate CLI arguments."""
    assert args is not None and hasattr(args, "prox_min_mm")
    if args.directives_json:
        seq = json.loads(args.directives_json)
        assert isinstance(seq, list), "--directives-json must decode to JSON array"
        return ("directives", seq)
    pl: Dict[str, Any] = {"preserve_heading": bool(args.preserve_heading), "strict": bool(args.strict)}
    if args.distance_mm is not None:
        pl["distance_mm"] = float(args.distance_mm)
        pl["bearing_deg"] = float(args.bearing_deg)
    else:
        pl["dx_mm"] = 0.0 if args.dx_mm is None else float(args.dx_mm)
        pl["dy_mm"] = float(args.dy_mm)
    if args.speed_mmps is not None:
        pl["speed_mmps"] = float(args.speed_mmps)
    if args.standoff_mm is not None:
        pl["standoff_mm"] = float(args.standoff_mm)
    assert isinstance(pl, dict) and "strict" in pl
    return ("translate", pl)


def cmd_translate(args: argparse.Namespace) -> int:
    """Validate, dry-run plan, or execute bounded 2D translation / directive batch."""
    assert args is not None and float(args.prox_min_mm) >= 0.0
    try:
        mode, data = _translate_payload_from_args(args)
    except (AssertionError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"ok": False, "stopped_reason": "invalid_args", "error": str(exc)}))
        return 2
    prox_min, standoff = float(args.prox_min_mm), float(args.standoff_mm)
    if args.dry_run:
        plan = plan_directive_batch(data, strict=bool(args.strict), min_prox_mm=prox_min, standoff_mm=standoff) if mode == "directives" else plan_directive(mode, data, strict=bool(args.strict), min_prox_mm=prox_min, standoff_mm=standoff)
        print(json.dumps(plan, indent=2))
        return 0 if plan.get("ok") else 6
    with VectorSDKSession(serial=args.serial, active=True, via_vector=args.via_vector, enable_nav_map=False) as robot:
        res = execute_directive_batch(robot, data, min_prox_mm=prox_min, standoff_mm=standoff, strict=bool(args.strict)) if mode == "directives" else execute_bounded_directive(robot, mode, data, min_prox_mm=prox_min, standoff_mm=standoff, strict=bool(args.strict))
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 6


def _add_translate_subparser(sub: Any) -> None:
    """Register top-level bounded translate subcommand."""
    assert sub is not None and DEFAULT_FORWARD_STANDOFF_MM > 0.0
    st = sub.add_parser("translate")
    st.add_argument("--dx-mm", type=float, default=None)
    st.add_argument("--dy-mm", type=float, default=0.0)
    st.add_argument("--distance-mm", type=float, default=None)
    st.add_argument("--bearing-deg", type=float, default=0.0)
    st.add_argument("--speed-mmps", type=float, default=None)
    st.add_argument("--standoff-mm", type=float, default=DEFAULT_FORWARD_STANDOFF_MM)
    st.add_argument("--preserve-heading", action="store_true")
    st.add_argument("--strict", action="store_true")
    st.add_argument("--prox-min-mm", type=float, default=35.0)
    st.add_argument("--directives-json", default=None)
    st.add_argument("--dry-run", action="store_true")
    st.add_argument("--via-vector", action="store_true")


def _add_maze_subparsers(sub: Any) -> None:
    """Register plan-maze, solve-maze, act, and translate subcommands."""
    spm = sub.add_parser("plan-maze")
    spm.add_argument("--cell-mm", type=float, default=180.0)
    spm.add_argument("--start-x-mm", type=float, default=0.0)
    spm.add_argument("--start-y-mm", type=float, default=0.0)
    spm.add_argument("--goal-row", type=int, default=None)
    spm.add_argument("--goal-col", type=int, default=None)
    ssm = sub.add_parser("solve-maze")
    ssm.add_argument("--via-vector", action="store_true")
    ssm.add_argument("--undock-if-needed", action="store_true")
    ssm.add_argument("--mode", choices=["flood-fill", "wall-follow", "speed-run"], default="flood-fill")
    ssm.add_argument("--max-steps", type=int, default=8)
    ssm.add_argument("--cell-mm", type=float, default=180.0)
    ssm.add_argument("--goal-row", type=int, default=None)
    ssm.add_argument("--goal-col", type=int, default=None)
    ssm.add_argument("--marker-map", default=None)
    ssm.add_argument("--peek-unknown", action="store_true")
    ssm.add_argument("--rehydrate-before-solve", action="store_true")
    ssm.add_argument("--two-pass", action="store_true")
    sa = sub.add_parser("act")
    sa.add_argument("verb", choices=["undock", "dock", "move", "rotate", "turn", "translate", "head", "lift", "directives", "halt"])
    sa.add_argument("--via-vector", action="store_true")
    sa.add_argument("--args-json", default="{}")
    sa.add_argument("--prox-min-mm", type=float, default=35.0)
    sa.add_argument("--strict", action="store_true")
    sa.add_argument("--dry-run", action="store_true")
    _add_translate_subparser(sub)


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for all vector-slam subcommands."""
    p = argparse.ArgumentParser(prog="vector-slam", description="Anki Vector persistent SLAM + maze solver")
    p.add_argument("--serial", default=DEFAULT_SERIAL)
    p.add_argument("--atlas", default=str(DEFAULT_ATLAS))
    p.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    p.add_argument("--session-id", default=None)
    p.add_argument("--assume-docked-canonical", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("probe")
    sp.add_argument("--wait-s", type=float, default=1.2)
    sub.add_parser("snapshot")
    ss = sub.add_parser("scan")
    ss.add_argument("--via-vector", action="store_true")
    ss.add_argument("--undock-if-needed", action="store_true")
    ss.add_argument("--return-charger", action="store_true")
    ss.add_argument("--steps", type=int, default=8)
    ss.add_argument("--step-deg", type=float, default=45.0)
    sr = sub.add_parser("rehydrate")
    sr.add_argument("--via-vector", action="store_true")
    sr.add_argument("--max-boxes", type=int, default=24)
    sr.add_argument("--test-box", action="store_true")
    sr.add_argument("--cleanup", action="store_true")
    _add_maze_subparsers(sub)
    so = sub.add_parser("onboard-cert")
    so.add_argument("--bot-sdk-info", default=str(DEFAULT_BOT_SDK_INFO))
    so.add_argument("--session-certs-dir", default=str(DEFAULT_SESSION_CERTS_DIR))
    so.add_argument("--sdk-dir", default=str(DEFAULT_SDK_DIR))
    so.add_argument("--sdk-config", default=None)
    so.add_argument("--robot-name", default=None)
    so.add_argument("--timeout-s", type=float, default=2.0)
    so.add_argument("--dry-run", action="store_true")
    so.add_argument("--allow-live-cert-fallback", action="store_true")
    sb = sub.add_parser("export-rosbag")
    sb.add_argument("--dataset", default=None)
    sb.add_argument("--samples-json", default=None)
    sb.add_argument("--out-bag", required=True)
    sb.add_argument("--storage", choices=["sqlite3", "mcap"], default="sqlite3")
    from .cli_phase2 import add_phase2_subparsers

    add_phase2_subparsers(sub)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch CLI subcommand and return POSIX exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    from .cli_phase2 import phase2_dispatch

    dispatch = {"probe": cmd_probe, "snapshot": cmd_snapshot, "scan": cmd_scan, "rehydrate": cmd_rehydrate, "plan-maze": cmd_plan_maze, "solve-maze": cmd_solve_maze, "act": cmd_act, "translate": cmd_translate, "onboard-cert": cmd_onboard_cert, "export-rosbag": cmd_export_rosbag}
    dispatch.update(phase2_dispatch())
    try:
        return dispatch[args.cmd](args)
    except AtlasLockTimeoutError as exc:
        print(json.dumps({"ok": False, "error": "atlas_lock_timeout", "detail": str(exc)}))
        return 4


def entrypoint() -> None:
    """Subprocess CLI entry that flushes stdio and always calls os._exit despite gRPC threads."""
    code = 1
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
