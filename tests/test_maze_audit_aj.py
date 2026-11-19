"""Unit tests for GPTPCS-Astra audit findings A-J in maze_closed_loop and navmap_atlas."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vector_slam_maze.landmarks import MazeMarkerAnchor
from vector_slam_maze.maze_closed_loop import (
    EAST,
    MARKER_MAX_STEP_MM,
    NORTH,
    SOUTH,
    WEST,
    _blend_marker_transform,
    _compute_corridor_chunk_drive,
    _execute_single_speed_segment,
    _pivot_blend_se2,
    check_rotational_clearance,
    execute_speed_run_segments,
    fuse_and_localize_step,
    peek_unknown_edges_tof,
    run_floodfill_loop,
    turn_to_canonical_theta_checked,
)
from vector_slam_maze.maze_solver import FloodFillPlanner, MazeGrid
from vector_slam_maze.navmap_atlas import CellStats, NavMapAtlas, SubmapLayer
from vector_slam_maze.types_and_math import NavContent, Pose2D, Transform2D, normalize_angle

from test_phase3_closed_loop_maze import (  # noqa: E402
    _Archetype,
    _MarkerEnum,
    _MarkerObj,
    _MockMazeRobot,
    _make_pose,
)


class _StampedMarkerObj(_MarkerObj):
    """Marker mock carrying SDK freshness attributes."""

    def __init__(self, name: str, pose: object, *, stamp_ns: int, is_visible: bool = True) -> None:
        super().__init__(_Archetype(_MarkerEnum(name)), pose)
        self.last_observed_time_ns = stamp_ns
        self.is_visible = is_visible


class MazeAuditAJTests(unittest.TestCase):
    """Targeted regression tests for findings A through J."""

    def test_finding_a_corridor_chunk_eliminates_end_of_segment_standoff_gap(self) -> None:
        tel = {"prox_valid": True, "prox_found": True, "prox_lift_in_fov": False, "prox_mm": 220.0}
        dist, reason = _compute_corridor_chunk_drive(180.0, 180.0, tel)
        self.assertEqual((dist, reason), (0.0, "unexpected_obstacle"))

    def test_finding_b_missing_or_occluded_tof_fails_closed(self) -> None:
        d1, r1 = _compute_corridor_chunk_drive(500.0, 180.0, {"prox_valid": False, "prox_found": False, "prox_lift_in_fov": False, "prox_mm": 9999.0})
        self.assertEqual((d1, r1), (0.0, "prox_unavailable"))
        d2, r2 = _compute_corridor_chunk_drive(500.0, 180.0, {"prox_valid": True, "prox_found": True, "prox_lift_in_fov": True, "prox_mm": 50.0})
        self.assertEqual((d2, r2), (0.0, "prox_occluded_lift_in_fov"))

    def test_finding_c_chunk_exhaustion_and_lateral_drift_halt(self) -> None:
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("c:origin_0")
        # 1) Lateral drift beyond 0.40 * cell_size_mm (72 mm) must halt before segment completion.
        robot_lat = _MockMazeRobot(grid, Pose2D(0.0, 80.0, 0.0, 0))
        loc_lat = fuse_and_localize_step(robot_lat, atlas, "c:origin_0")
        trace: list = []
        _loc_out, halt_lat = _execute_single_speed_segment(
            robot_lat, atlas, grid, "c:origin_0", {"dir": EAST, "cells": 2, "start": [2, 2], "end": [2, 4]}, 0, 180.0, None, loc_lat, trace
        )
        self.assertIsNotNone(halt_lat)
        self.assertEqual(halt_lat["reason"], "lateral_drift_exceeded")

        # 2) Chunk budget exhaustion (MAX_CHUNKS_PER_SEGMENT=32) must halt, never return success.
        grid_long = MazeGrid(3, 128, 180.0, -90.0, -90.0)
        robot_long = _MockMazeRobot(grid_long, Pose2D(0.0, 180.0, 0.0, 0))
        loc_long = fuse_and_localize_step(robot_long, atlas, "c:origin_0")
        trace2: list = []
        _loc2, halt_bud = _execute_single_speed_segment(
            robot_long, atlas, grid_long, "c:origin_0", {"dir": EAST, "cells": 110, "start": [1, 0], "end": [1, 110]}, 0, 180.0, None, loc_long, trace2
        )
        self.assertIsNotNone(halt_bud)
        self.assertEqual(halt_bud["reason"], "segment_chunk_budget_exhausted")

    def test_finding_d_turn_uses_bounded_turn_and_rejects_unverified_heading(self) -> None:
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("d:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0))
        robot.status.is_on_charger = True
        _d, mr_chg = turn_to_canonical_theta_checked(robot, math.pi * 0.5, Transform2D(0.0, 0.0, 0.0))
        self.assertFalse(mr_chg.ok)
        self.assertEqual(mr_chg.detail, "on_charger")

        robot2 = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0))
        robot2.behavior.turn_in_place = lambda *_a, **_k: None  # no-op turn leaves heading wrong
        res = run_floodfill_loop(robot2, atlas, grid, FloodFillPlanner(grid), "d:origin_0", (3, 2), 3, 180.0, None, False)
        self.assertEqual(res["status"], "halted")
        self.assertEqual(res["reason"], "turn_heading_mismatch")
        self.assertEqual(robot2.move_count, 0)

    def test_finding_e_requires_valid_canonical_localization_before_motion(self) -> None:
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.ensure_submap("prior:origin_1", 1)  # canonical_submap_id is now prior:origin_1
        robot = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0))
        res_spd = execute_speed_run_segments(robot, atlas, grid, FloodFillPlanner(grid), "new:origin_0", goal_cell=(2, 4))
        self.assertEqual((res_spd["status"], res_spd["reason"]), ("halted", "unlocalized"))
        self.assertEqual(robot.move_count, 0)

        res_ff = run_floodfill_loop(robot, atlas, grid, FloodFillPlanner(grid), "new:origin_0", (2, 4), 4, 180.0, None, True)
        self.assertEqual((res_ff["status"], res_ff["reason"]), ("halted", "unlocalized"))
        self.assertEqual(robot.move_count, 0)
        self.assertEqual(robot.turn_history_deg, [])

    def test_finding_f_fuse_surfaces_safety_failure_and_skips_invalid_tof(self) -> None:
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("f:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0))
        robot.status.is_cliff_detected = True
        robot.proximity = None
        loc = fuse_and_localize_step(robot, atlas, "f:origin_0")
        self.assertFalse(loc["safety_ok"])
        self.assertEqual(loc["safety_reason"], "cliff")
        self.assertEqual(len(atlas.submaps["f:origin_0"].cells), 0)

        res = run_floodfill_loop(robot, atlas, grid, FloodFillPlanner(grid), "f:origin_0", (2, 4), 4, 180.0, None, False)
        self.assertEqual((res["status"], res["reason"]), ("halted", "cliff"))

    def test_finding_g_swept_footprint_rotational_clearance_blocks_offcenter_turn(self) -> None:
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        grid.set_wall(2, 2, EAST, True)
        ok_center, _ = check_rotational_clearance(Pose2D(0.0, 0.0, 0.0, 0), grid)
        self.assertTrue(ok_center)
        # 50 mm east of cell center leaves 40 mm to EAST wall (< 52+5 mm swept clearance).
        ok_off, reason_off = check_rotational_clearance(Pose2D(50.0, 0.0, 0.0, 0), grid)
        self.assertFalse(ok_off)
        self.assertEqual(reason_off, "swept_clearance_wall")

        atlas = NavMapAtlas(cell_mm=20.0)
        atlas.assume_same_frame_as_canonical("g:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(50.0, 0.0, 0.0, 0))
        res_pk, _evs, _loc = peek_unknown_edges_tof(robot, atlas, grid, (2, 2), EAST, "g:origin_0", 180.0)
        self.assertEqual(res_pk, "swept_clearance_wall")
        self.assertEqual(robot.turn_history_deg, [])

        # Atlas-only occupied obstacle near off-center robot also blocks turn.
        grid_open = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        sub = atlas.ensure_submap("g:origin_0", 0)
        sub.cells[(4, 0)] = CellStats(log_odds=2.5, obs_count=2, last_content=int(NavContent.OBSTACLE_PROXIMITY))
        ok_atl, reason_atl = check_rotational_clearance(Pose2D(45.0, 0.0, 0.0, 0), grid_open, atlas=atlas)
        self.assertFalse(ok_atl)
        self.assertEqual(reason_atl, "swept_clearance_obstacle")

    def test_finding_h_fresh_marker_stamps_and_streak_decay_on_misses(self) -> None:
        atlas = NavMapAtlas(cell_mm=20.0)
        markers = {"Circles2": MazeMarkerAnchor("Circles2", 2, 4, 360.0, 0.0, 0.0)}
        robot = _MockMazeRobot(MazeGrid(5, 5, 180.0, -450.0, -450.0), Pose2D(0.0, 0.0, 0.0, 7))
        robot.markers_by_origin[7] = [_StampedMarkerObj("Circles2", _make_pose(360.0, 0.0, 0.0, 7), stamp_ns=100)]
        fuse_and_localize_step(robot, atlas, "h:origin_0", marker_map=markers)
        atlas.marker_obs_counts["h:origin_7"] = 10

        # Same timestamp 101 repeated 4 times must NOT reach MARKER_RELOCK_COUNT = 3.
        tf_bump = Transform2D(260.0, -80.0, math.radians(22.0))
        m_inv = tf_bump.invert().apply_pose(Pose2D(360.0, 0.0, 0.0, 0), target_origin_id=7)
        for _ in range(4):
            robot.markers_by_origin[7] = [_StampedMarkerObj("Circles2", _make_pose(m_inv.x_mm, m_inv.y_mm, m_inv.theta_rad, 7), stamp_ns=101)]
            loc = fuse_and_localize_step(robot, atlas, "h:origin_0", marker_map=markers)
            self.assertFalse(loc["marker_relocked"])

        # Advancing timestamps: 2 outliers -> 1 missed frame (decays streak 2->1) -> 1 outlier -> STILL no relock.
        robot.markers_by_origin[7] = [_StampedMarkerObj("Circles2", _make_pose(m_inv.x_mm, m_inv.y_mm, m_inv.theta_rad, 7), stamp_ns=102)]
        loc2 = fuse_and_localize_step(robot, atlas, "h:origin_0", marker_map=markers)
        self.assertTrue(loc2["marker_rejected"])
        self.assertEqual(atlas.marker_outlier_streaks["h:origin_7"][1], 2)

        robot.markers_by_origin[7] = []  # missed frame decays 2 -> 1
        fuse_and_localize_step(robot, atlas, "h:origin_0", marker_map=markers)
        self.assertEqual(atlas.marker_outlier_streaks["h:origin_7"][1], 1)

        robot.markers_by_origin[7] = [_StampedMarkerObj("Circles2", _make_pose(m_inv.x_mm, m_inv.y_mm, m_inv.theta_rad, 7), stamp_ns=103)]
        loc3 = fuse_and_localize_step(robot, atlas, "h:origin_0", marker_map=markers)
        self.assertFalse(loc3["marker_relocked"])
        self.assertEqual(atlas.marker_outlier_streaks["h:origin_7"][1], 2)

        # Next fresh outlier reaches quorum 3 and relocks.
        robot.markers_by_origin[7] = [_StampedMarkerObj("Circles2", _make_pose(m_inv.x_mm, m_inv.y_mm, m_inv.theta_rad, 7), stamp_ns=104)]
        loc4 = fuse_and_localize_step(robot, atlas, "h:origin_0", marker_map=markers)
        self.assertTrue(loc4["marker_relocked"])

    def test_finding_i_lever_arm_displacement_bounded_at_robot_pose(self) -> None:
        tf_a = Transform2D(0.0, 0.0, 0.0)
        tf_rot5 = Transform2D(0.0, 0.0, math.radians(5.0))
        tf_inlier = Transform2D(0.0, 0.0, math.radians(3.4))
        sub = SubmapLayer(cell_mm=20.0, origin_id=0)
        sub.cells[(0, 0)] = CellStats(log_odds=-1.0, obs_count=1)  # centroid at (10, 10) mm
        for dist_mm in (1000.0, 2000.0):
            rpose = Pose2D(dist_mm, 0.0, 0.0, 0)
            rx0, ry0 = tf_a.apply_point(rpose.x_mm, rpose.y_mm)
            # Raw centroid-only pivot blend jumps ~8.7 mm at 1 m and ~17.5 mm at 2 m.
            raw_tf = _pivot_blend_se2(tf_a, tf_rot5, 10.0, 10.0, 0.1, MARKER_MAX_STEP_MM, 0.5)
            rx_raw, ry_raw = raw_tf.apply_point(rpose.x_mm, rpose.y_mm)
            self.assertGreater(math.hypot(rx_raw - rx0, ry_raw - ry0), MARKER_MAX_STEP_MM)

            # Bounded at robot pose caps direct _pivot_blend_se2 as well as _blend_marker_transform.
            cap_tf = _pivot_blend_se2(tf_a, tf_rot5, 10.0, 10.0, 0.1, MARKER_MAX_STEP_MM, 0.5, bound_pose=rpose)
            rx_cap, ry_cap = cap_tf.apply_point(rpose.x_mm, rpose.y_mm)
            self.assertLessEqual(math.hypot(rx_cap - rx0, ry_cap - ry0), MARKER_MAX_STEP_MM + 1e-6)

            bounded_tf, accepted, _rej = _blend_marker_transform(tf_a, tf_inlier, rpose, sub, True, 1)
            self.assertTrue(accepted)
            rx_b, ry_b = bounded_tf.apply_point(rpose.x_mm, rpose.y_mm)
            self.assertLessEqual(math.hypot(rx_b - rx0, ry_b - ry0), MARKER_MAX_STEP_MM + 1e-6)

    def test_finding_j_peek_refreshes_curr_when_fusion_shifts_cell(self) -> None:
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("j:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0))
        calls = {"n": 0}

        def shifting_fuse(*_a, **_k):
            calls["n"] += 1
            # On the post-turn fusion during first peek, canonical pose shifts from (2,2) to (2,3).
            x_val = 0.0 if calls["n"] == 1 else 180.0
            return {
                "submap_id": "j:origin_0",
                "origin_id": 0,
                "rpose": Pose2D(x_val, 0.0, 0.0, 0),
                "cpose": Pose2D(x_val, 0.0, 0.0, 0),
                "tf_can": Transform2D(0.0, 0.0, 0.0),
                "relocalized_via": None,
                "localized": True,
                "marker_rejected": False,
                "marker_relocked": False,
                "safety_ok": True,
                "safety_reason": "ok",
            }

        with mock.patch("vector_slam_maze.maze_closed_loop.fuse_and_localize_step", side_effect=shifting_fuse):
            with mock.patch("vector_slam_maze.maze_closed_loop.check_safety", return_value=(True, "ok", {"prox_valid": True, "prox_found": True, "prox_lift_in_fov": False, "prox_mm": 100.0})):
                err, evs, _loc = peek_unknown_edges_tof(robot, atlas, grid, (2, 2), EAST, "j:origin_0", 180.0)
        self.assertIsNone(err)
        self.assertGreaterEqual(len(evs), 1)
        self.assertEqual(evs[0]["cell"], [2, 3])
        self.assertTrue(bool(grid.walls[2][3] & EAST))
        self.assertFalse(bool(grid.walls[2][2] & EAST))


if __name__ == "__main__":
    unittest.main()
