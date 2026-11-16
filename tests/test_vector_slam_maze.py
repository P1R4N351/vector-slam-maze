"""Unit and integration tests for vector_slam_maze."""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vector_slam_maze.landmarks import MazeMarkerAnchor, anchor_name_from_object, localize_from_marker_anchors
from vector_slam_maze.maze_solver import EAST, NORTH, SOUTH, WEST, FloodFillPlanner, MazeGrid, RightWallFollower
from vector_slam_maze.navmap_atlas import (
    LeafPatch,
    NavMapAtlas,
    SubmapLayer,
    compress_occupied_to_boxes,
    extract_quadtree_leaves,
)
from vector_slam_maze.sdk_session import VectorSDKSession, check_safety, safe_disconnect
from vector_slam_maze.types_and_math import NavContent, Pose2D, Transform2D, fit_se2_from_anchors, normalize_angle


def _mock_leaf(cx: float, cy: float, size: float, content: int) -> SimpleNamespace:
    return SimpleNamespace(center=SimpleNamespace(x=cx, y=cy), size=size, content=content, children=None)


def _mock_nav_map() -> SimpleNamespace:
    children = [
        _mock_leaf(20.0, 20.0, 20.0, NavContent.CLEAR_OF_OBSTACLE),
        _mock_leaf(20.0, -20.0, 20.0, NavContent.OBSTACLE_PROXIMITY),
        _mock_leaf(-20.0, 20.0, 20.0, NavContent.CLIFF),
        SimpleNamespace(children=[_mock_leaf(-30.0, -30.0, 10.0, NavContent.CLEAR_OF_CLIFF)], size=20.0),
    ]
    return SimpleNamespace(root_node=SimpleNamespace(children=children, size=80.0))


class TestGeometryAndAnchors(unittest.TestCase):
    """Verify SE(2) math, inversion, composition, and multi-anchor Procrustes fitting."""

    def test_transform_inverse_and_compose(self) -> None:
        tf = Transform2D(tx_mm=240.0, ty_mm=-110.0, dtheta_rad=math.radians(37.5))
        inv = tf.invert()
        px, py = tf.apply_point(125.0, -45.0)
        rx, ry = inv.apply_point(px, py)
        self.assertAlmostEqual(rx, 125.0, places=4)
        self.assertAlmostEqual(ry, -45.0, places=4)
        comp = tf.compose(inv)
        self.assertAlmostEqual(comp.tx_mm, 0.0, places=4)
        self.assertAlmostEqual(comp.ty_mm, 0.0, places=4)
        self.assertAlmostEqual(normalize_angle(comp.dtheta_rad), 0.0, places=4)

    def test_single_oriented_anchor_registration(self) -> None:
        src = Pose2D(100.0, 50.0, math.radians(15.0), origin_id=7)
        dst = Pose2D(400.0, -120.0, math.radians(105.0), origin_id=1)
        tf = fit_se2_from_anchors([(src, dst)])
        mapped = tf.apply_pose(src, target_origin_id=1)
        self.assertAlmostEqual(mapped.x_mm, dst.x_mm, places=4)
        self.assertAlmostEqual(mapped.y_mm, dst.y_mm, places=4)
        self.assertAlmostEqual(mapped.theta_rad, dst.theta_rad, places=5)

    def test_multi_anchor_procrustes_fit(self) -> None:
        true_tf = Transform2D(150.0, -80.0, math.radians(-62.0))
        src_pts = [Pose2D(0.0, 0.0, 0.0), Pose2D(200.0, 0.0, 0.0), Pose2D(0.0, 300.0, 0.5)]
        pairs = [(p, true_tf.apply_pose(p)) for p in src_pts]
        est = fit_se2_from_anchors(pairs)
        self.assertAlmostEqual(est.tx_mm, true_tf.tx_mm, places=3)
        self.assertAlmostEqual(est.ty_mm, true_tf.ty_mm, places=3)
        self.assertAlmostEqual(est.dtheta_rad, true_tf.dtheta_rad, places=4)


class TestNavMapAtlasAndPersistence(unittest.TestCase):
    """Verify quad-tree extraction, ToF carving, multi-session stitching, and ROS export."""

    def test_quadtree_leaf_extraction(self) -> None:
        leaves = extract_quadtree_leaves(_mock_nav_map())
        self.assertEqual(len(leaves), 4)
        contents = sorted(lf.content for lf in leaves)
        self.assertEqual(contents, [1, 2, 4, 7])

    def test_tof_raycast_carves_free_and_marks_hit(self) -> None:
        sub = SubmapLayer(cell_mm=20.0, origin_id=1)
        hits = sub.integrate_tof_ray(Pose2D(0.0, 0.0, 0.0, 1), prox_mm=210.0, found_object=True, lift_in_fov=False)
        self.assertGreater(hits, 5)
        self.assertLess(sub.cells[(0, 0)].log_odds, 0.0)
        self.assertGreater(sub.cells[(10, 0)].log_odds, 0.0)

    def test_multisession_anchor_stitching_and_lazy_merge(self) -> None:
        atlas = NavMapAtlas(cell_mm=20.0)
        s1 = atlas.ensure_submap("s1:origin_1", origin_id=1)
        s1.integrate_tof_ray(Pose2D(0.0, 0.0, 0.0, 1), 240.0, True, False)
        atlas.record_anchor("s1:origin_1", "charger", Pose2D(0.0, 0.0, 0.0, 1))
        tf_s2_to_s1 = Transform2D(300.0, -100.0, math.radians(90.0))
        tf_s1_to_s2 = tf_s2_to_s1.invert()
        charger_in_s2 = tf_s1_to_s2.apply_pose(Pose2D(0.0, 0.0, 0.0, 1), target_origin_id=7)
        s2 = atlas.ensure_submap("s2:origin_7", origin_id=7)
        s2.integrate_tof_ray(charger_in_s2, 240.0, True, False)
        atlas.record_anchor("s2:origin_7", "charger", charger_in_s2)
        self.assertIn("s2:origin_7", atlas.transforms_to_canonical)
        atlas.ensure_submap("s3:origin_9", origin_id=9)
        atlas.record_anchor("s3:origin_9", "marker:Circles2", Pose2D(10.0, 20.0, 0.0, 9))
        self.assertNotIn("s3:origin_9", atlas.transforms_to_canonical)
        atlas.record_anchor("s2:origin_7", "marker:Circles2", Pose2D(40.0, 50.0, 0.2, 7))
        self.assertIn("s3:origin_9", atlas.transforms_to_canonical)

    def test_save_load_export_and_frontier_compat(self) -> None:
        atlas = NavMapAtlas(cell_mm=20.0)
        s1 = atlas.ensure_submap("s1:origin_1", origin_id=1)
        s1.integrate_leaves(extract_quadtree_leaves(_mock_nav_map()))
        s1.integrate_tof_ray(Pose2D(0.0, 0.0, 0.0, 1), 320.0, True, False)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            atlas.save(p / "atlas.json")
            reloaded = NavMapAtlas.load(p / "atlas.json")
            stats = reloaded.export_ros_pgm_yaml(p / "grid.pgm", p / "grid.yaml")
            self.assertGreater(stats["free_cells"], 0)
            self.assertGreater(stats["cliff_cells"], 0)
            boxes = compress_occupied_to_boxes(reloaded, max_boxes=16)
            self.assertGreater(len(boxes), 0)
            ws_frontiers = Path(os.environ.get("CARTOGRAPHER_FRONTIERS_SCRIPT") or "/nonexistent/_cartographer_frontiers.py")
            if ws_frontiers.is_file():
                proc = subprocess.run([sys.executable, str(ws_frontiers), str(p / "grid.pgm"), str(p / "grid.yaml")], capture_output=True, text=True, check=True)
                self.assertIsInstance(json.loads(proc.stdout), list)


class TestMazeSolversAndSafety(unittest.TestCase):
    """Verify flood-fill planning, re-flood on wall discovery, right-wall follow, and consent gate."""

    def test_floodfill_detours_around_walls_and_blocks(self) -> None:
        grid = MazeGrid(rows=5, cols=5, cell_size_mm=180.0)
        for r in range(4):
            grid.set_wall(r, 1, EAST, True)
        grid.blocked[3][3] = True
        planner = FloodFillPlanner(grid)
        path = planner.plan_path((0, 0), [(0, 4)], initial_heading=NORTH)
        self.assertGreater(len(path), 6)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (0, 4))
        self.assertIn((4, 1), path)
        self.assertNotIn((3, 3), path)

    def test_from_atlas_discretizes_obstacles_and_landmark_localizer(self) -> None:
        atlas = NavMapAtlas(cell_mm=20.0)
        s1 = atlas.ensure_submap("s1:origin_1", 1)
        s1.integrate_tof_ray(Pose2D(0.0, 0.0, 0.0, 1), 160.0, True, False)
        grid = MazeGrid.from_atlas(atlas, cell_size_mm=180.0)
        self.assertGreaterEqual(grid.rows, 5)
        known = {"Circles2": MazeMarkerAnchor("Circles2", row=1, col=2, x_mm=360.0, y_mm=180.0, normal_deg=90.0)}
        est = localize_from_marker_anchors({"marker:Circles2": Pose2D(100.0, 0.0, 0.0, 5)}, known, Pose2D(0.0, 0.0, 0.0, 5))
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est.y_mm, 80.0, places=2)

    def test_right_wall_follower_decisions(self) -> None:
        wf = RightWallFollower(cell_step_mm=140.0, wall_thresh_mm=130.0)
        self.assertEqual(wf.decide_step(200.0, 200.0, False).action, "TURN_RIGHT_ADVANCE")
        self.assertEqual(wf.decide_step(200.0, 80.0, False).action, "ADVANCE")
        self.assertEqual(wf.decide_step(80.0, 80.0, False).action, "TURN_LEFT")
        self.assertEqual(wf.decide_step(200.0, 200.0, True).action, "ABORT_CLIFF")

    def test_consent_gate_and_py311_cancelled_error_teardown(self) -> None:
        with self.assertRaises(PermissionError):
            VectorSDKSession(serial="0dd1cafe", active=True, via_vector=False)
        called = {"released": False, "stopped": False, "disconnected": False}

        class DummyFuture:
            def result(self, timeout: float = 2.0) -> None:
                raise asyncio.CancelledError()

        class DummyTask:
            def cancel(self) -> None:
                pass

        bot = SimpleNamespace(
            motors=SimpleNamespace(stop_all_motors=lambda: called.__setitem__("stopped", True)),
            conn=SimpleNamespace(release_control=lambda: called.__setitem__("released", True), run_coroutine=lambda _t: DummyFuture()),
            nav_map=SimpleNamespace(_nav_map_feed_task=DummyTask()),
            disconnect=lambda: called.__setitem__("disconnected", True),
        )
        safe_disconnect(bot, active=True)
        self.assertTrue(all(called.values()))
        self.assertIsNone(bot.nav_map._nav_map_feed_task)

    def test_anchor_name_extraction(self) -> None:
        m = SimpleNamespace(archetype=SimpleNamespace(marker_front=SimpleNamespace(name="Hexagons3")))
        self.assertEqual(anchor_name_from_object(m), "marker:Hexagons3")

    def test_deep_quadtree_and_tof_max_range_no_hit(self) -> None:
        leaf = _mock_leaf(8.0, 8.0, 8.0, NavContent.CLEAR_OF_OBSTACLE)
        node = leaf
        for depth in range(6):
            node = SimpleNamespace(children=[node], size=float(8 * (2 ** (depth + 1))))
        leaves = extract_quadtree_leaves(SimpleNamespace(root_node=node))
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].size_mm, 8.0)
        sub = SubmapLayer(cell_mm=20.0, origin_id=2)
        sub.integrate_tof_ray(Pose2D(0.0, 0.0, 0.0, 2), prox_mm=1500.0, found_object=True, lift_in_fov=False)
        self.assertEqual(sum(1 for st in sub.cells.values() if st.log_odds > 0.0), 0)

    def test_floodfill_known_only_vs_exploration_and_online_reflood(self) -> None:
        grid = MazeGrid(rows=4, cols=4, cell_size_mm=180.0)
        planner = FloodFillPlanner(grid)
        self.assertEqual(planner.plan_path((0, 0), [(0, 3)], known_only=True), [])
        explore_path = planner.plan_path((0, 0), [(0, 3)], known_only=False)
        self.assertEqual(explore_path, [(0, 0), (0, 1), (0, 2), (0, 3)])
        grid.set_wall(0, 1, EAST, True)
        rerouted = planner.plan_path((0, 1), [(0, 3)], known_only=False)
        self.assertIn((1, 1), rerouted)
        self.assertEqual(rerouted[-1], (0, 3))

    def test_box_compression_respects_cap_and_covers_cells(self) -> None:
        atlas = NavMapAtlas(cell_mm=20.0)
        sub = atlas.ensure_submap("s:0", 0)
        for ix in range(4):
            for iy in range(2):
                st = sub.cells.setdefault((ix, iy), SubmapLayer(20.0).cells.get((0, 0)) or __import__("vector_slam_maze.navmap_atlas", fromlist=["CellStats"]).CellStats())
                st.apply_content(NavContent.OBSTACLE_UNRECOGNIZED)
        boxes = compress_occupied_to_boxes(atlas, max_boxes=4)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][2], 80.0, places=3)
        self.assertAlmostEqual(boxes[0][3], 40.0, places=3)

    def test_cliff_avoidance_in_maze_grid_and_negative_y_pgm(self) -> None:
        atlas = NavMapAtlas(cell_mm=20.0)
        sub = atlas.ensure_submap("s:0", 0)
        sub.integrate_leaves([LeafPatch(180.0, 180.0, 40.0, NavContent.CLIFF), LeafPatch(10.0, -40.0, 20.0, NavContent.CLEAR_OF_OBSTACLE)])
        grid = MazeGrid.from_atlas(atlas, cell_size_mm=180.0)
        cr, cc = grid.world_to_cell(180.0, 180.0)
        self.assertTrue(grid.blocked[cr][cc])
        with tempfile.TemporaryDirectory() as tmp:
            stats = atlas.export_ros_pgm_yaml(Path(tmp) / "g.pgm", Path(tmp) / "g.yaml")
            self.assertGreaterEqual(stats["cliff_cells"], 1)

    def test_assume_docked_canonical_and_unknown_edge_prior(self) -> None:
        atlas = NavMapAtlas(cell_mm=20.0)
        s1 = atlas.ensure_submap("s1:1", 1)
        s1.integrate_leaves([LeafPatch(0.0, 0.0, 20.0, NavContent.INTERESTING_EDGE)])
        s2 = atlas.ensure_submap("s2:2", 2)
        atlas.assume_same_frame_as_canonical("s2:2")
        self.assertIn("s2:2", atlas.transforms_to_canonical)
        self.assertLess(atlas.fuse_canonical_cells()[(0, 0)].log_odds, 0.4)

    def test_cli_plan_maze_offline_json(self) -> None:
        atlas = NavMapAtlas(cell_mm=20.0)
        s1 = atlas.ensure_submap("cli:1", 1)
        s1.integrate_tof_ray(Pose2D(0.0, 0.0, 0.0, 1), 300.0, True, False)
        with tempfile.TemporaryDirectory() as tmp:
            ap = Path(tmp) / "atlas.json"
            atlas.save(ap)
            from vector_slam_maze.cli import main as cli_main
            rc = cli_main(["--atlas", str(ap), "plan-maze"])
            self.assertEqual(rc, 0)

    def test_status_is_on_charger_property(self) -> None:
        robot = SimpleNamespace(
            status=SimpleNamespace(is_on_charger=True, is_in_calm_power_mode=True, is_charging=True),
            get_battery_state=lambda: SimpleNamespace(battery_volts=4.05, battery_level=2),
            proximity=SimpleNamespace(last_sensor_reading=None),
            pose=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0), rotation=SimpleNamespace(angle_z=SimpleNamespace(degrees=0.0)), origin_id=1),
        )
        ok, reason, tel = check_safety(robot, allow_on_charger=False)
        self.assertFalse(ok)
        self.assertEqual(reason, "on_charger")
        self.assertTrue(tel["is_on_charger"])
        self.assertTrue(tel["is_in_calm_power_mode"])

    def test_sdk_session_inits_nav_map_post_connect(self) -> None:
        captured: dict = {}

        class FakeBot:
            def __init__(self, **kw: object) -> None:
                captured["kwargs"] = kw
                self.nav_map = SimpleNamespace(init_nav_map_feed=lambda frequency: captured.__setitem__("freq", frequency))

            def connect(self) -> None:
                captured["connected"] = True

            def disconnect(self) -> None:
                captured["disconnected"] = True

        fake_av = SimpleNamespace(Robot=FakeBot, connection=SimpleNamespace(ControlPriorityLevel=SimpleNamespace(OVERRIDE_BEHAVIORS_PRIORITY=1)))
        with mock.patch("vector_slam_maze.sdk_session._import_anki_vector", return_value=fake_av):
            with VectorSDKSession(serial="0dd1cafe", active=False, enable_nav_map=True, nav_freq_hz=2.0):
                pass
        self.assertFalse(captured["kwargs"]["enable_nav_map_feed"])
        self.assertEqual(captured["freq"], 2.0)

    def test_entrypoint_os_exit_on_exception(self) -> None:
        import os as _os
        from vector_slam_maze.cli import entrypoint

        exited: list[int] = []
        with mock.patch("vector_slam_maze.cli.main", side_effect=RuntimeError("boom")), mock.patch.object(
            _os, "_exit", side_effect=lambda code: exited.append(code)
        ):
            entrypoint()
        self.assertEqual(exited, [1])

    def test_node_to_leaf_handles_none_content(self) -> None:
        nav = SimpleNamespace(
            root_node=SimpleNamespace(
                children=[
                    _mock_leaf(10.0, 10.0, 16.0, None),
                    SimpleNamespace(center=SimpleNamespace(x=20.0, y=20.0), size=16.0, content=SimpleNamespace(value=None), children=[]),
                ]
            )
        )
        leaves = extract_quadtree_leaves(nav)
        self.assertEqual([lf.content for lf in leaves], [0, 0])


if __name__ == "__main__":
    unittest.main()
