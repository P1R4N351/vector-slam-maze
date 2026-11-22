"""Tests for v0.5.0 basal MemoryMap persistence, SQLite export, rehydration clearance, and daemon loop."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from vector_slam_maze.basal_persist import (
    BasalConfig,
    BasalMapDaemon,
    check_pod_cryosleep,
    export_seated_navmap,
    export_seated_navmap_dict,
)
from vector_slam_maze.cli import main as cli_main
from vector_slam_maze.navmap_atlas import (
    LeafPatch,
    NavMapAtlas,
    canonical_charger_pose,
    compress_occupied_to_boxes,
    rehydrate_onboard_obstacles,
)
from vector_slam_maze.sdk_session import poll_fresh_nav_map
from vector_slam_maze.types_and_math import NavContent, Pose2D, Transform2D


def _make_pose(x: float, y: float, yaw_rad: float, oid: int) -> SimpleNamespace:
    """Construct lightweight SDK-like Pose object."""
    assert isinstance(oid, int)
    return SimpleNamespace(
        position=SimpleNamespace(x=float(x), y=float(y), z=0.0),
        rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=float(yaw_rad), degrees=math.degrees(yaw_rad))),
        origin_id=oid,
    )


def _make_leaf(cx: float, cy: float, size: float, content: int) -> SimpleNamespace:
    """Construct leaf NavMapGrid node."""
    assert size > 0.0
    return SimpleNamespace(center=SimpleNamespace(x=float(cx), y=float(cy)), size=float(size), content=int(content), children=None)


def _make_grid(origin_id: int, leaves: list[SimpleNamespace]) -> SimpleNamespace:
    """Construct NavMapGrid-like root node container."""
    assert isinstance(leaves, list)
    return SimpleNamespace(origin_id=int(origin_id), size=2048.0, root_node=SimpleNamespace(children=leaves))


def _make_status(
    *,
    on_charger: bool = True,
    wheels_moving: bool = False,
    picked_up: bool = False,
    calm: bool = False,
) -> SimpleNamespace:
    """Construct lightweight robot.status stub."""
    assert isinstance(on_charger, bool)
    return SimpleNamespace(
        is_on_charger=on_charger,
        is_on_charger_platform=False,
        are_wheels_moving=wheels_moving,
        is_picked_up=picked_up,
        is_in_calm_power_mode=calm,
        is_cliff_detected=False,
        is_falling=False,
        is_charging=on_charger,
    )


class _FakeEvents:
    """Minimal event pub/sub stub matching anki_vector events interface."""

    def __init__(self) -> None:
        self.subs: list[tuple[object, object]] = []
        self.robot_state_event = "robot_state"

    def subscribe(self, fn: object, ev_type: object) -> None:
        assert callable(fn)
        self.subs.append((fn, ev_type))

    def unsubscribe(self, fn: object, ev_type: object) -> None:
        self.subs = [item for item in self.subs if item != (fn, ev_type)]


class _FakeWorld:
    """Minimal world stub tracking fixed custom objects created during rehydration."""

    def __init__(self) -> None:
        self.fixed_calls: list[dict[str, object]] = []
        self.deleted_fixed = 0
        self.charger = None
        self.connected_light_cube = None
        self.visible_objects: list[object] = []

    def delete_custom_objects(self, **kwargs: object) -> None:
        assert "delete_fixed_custom_objects" in kwargs
        if kwargs.get("delete_fixed_custom_objects"):
            self.deleted_fixed += 1

    def create_custom_fixed_object(self, pose: object, **kwargs: object) -> SimpleNamespace:
        assert pose is not None
        self.fixed_calls.append({"pose": pose, **kwargs})
        return SimpleNamespace(pose=pose, **kwargs)


class _FakeBattery:
    battery_volts = 4.06
    battery_level = 3


class _FakeRobot:
    """Configurable robot stub for BasalMapDaemon and poll_fresh_nav_map tests."""

    def __init__(self, origin_id: int = 15, leaves: list[SimpleNamespace] | None = None) -> None:
        self.pose = _make_pose(100.0, 200.0, math.pi / 2.0, origin_id)
        self.status = _make_status(on_charger=True)
        self.nav_map = SimpleNamespace(_latest_nav_map=_make_grid(origin_id, leaves or []) if leaves is not None else None)
        self.events = _FakeEvents()
        self.world = _FakeWorld()
        self.proximity = SimpleNamespace(last_sensor_reading=None)

    def get_battery_state(self) -> _FakeBattery:
        return _FakeBattery()


def _install_fake_anki_util(mock_patch_stack: list[object]) -> None:
    """Provide fake anki_vector.util types when SDK is not installed."""
    assert isinstance(mock_patch_stack, list)
    fake_util = SimpleNamespace(
        Pose=lambda x, y, z, angle_z=None, origin_id=0: SimpleNamespace(x=x, y=y, z=z, angle_z=angle_z, origin_id=origin_id),
        radians=lambda rad: SimpleNamespace(radians=float(rad)),
        degrees=lambda deg: SimpleNamespace(degrees=float(deg)),
    )
    fake_sdk = SimpleNamespace(util=fake_util)
    patcher = mock.patch.dict("sys.modules", {"anki_vector": fake_sdk, "anki_vector.util": fake_util})
    patcher.start()
    mock_patch_stack.append(patcher)
    assert len(mock_patch_stack) >= 1


def _const_session_factory(robot: object) -> type:
    """Build single-robot context manager factory for BasalMapDaemon tests."""
    assert robot is not None and hasattr(robot, "nav_map")

    class _FakeSess:
        def __init__(self, **_kw: object) -> None:
            pass

        def __enter__(self) -> object:
            return robot

        def __exit__(self, *_args: object) -> bool:
            return False

    assert callable(_FakeSess)
    return _FakeSess


class TestBasalPersist(unittest.TestCase):
    """Unit and fault-injection tests for basal_persist, clearance filter, and fresh NavMap poll."""

    def setUp(self) -> None:
        self._patchers: list[object] = []

    def tearDown(self) -> None:
        for p in reversed(self._patchers):
            p.stop()

    def test_poll_fresh_nav_map_waits_for_new_identity(self) -> None:
        grid0 = _make_grid(15, [_make_leaf(0.0, 0.0, 20.0, NavContent.CLEAR_OF_OBSTACLE)])
        grid1 = _make_grid(15, [_make_leaf(200.0, 0.0, 20.0, NavContent.OBSTACLE_CUBE)])
        robot = _FakeRobot(15)
        robot.nav_map._latest_nav_map = grid0

        stale_grid, fresh0 = poll_fresh_nav_map(robot, max_wait_s=0.05, step_s=0.02, after_map=grid0)
        self.assertIs(stale_grid, grid0)
        self.assertFalse(fresh0)

        calls = 0

        def _advance(_dt: float) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                robot.nav_map._latest_nav_map = grid1

        with mock.patch("vector_slam_maze.sdk_session.time.sleep", side_effect=_advance):
            new_grid, fresh1 = poll_fresh_nav_map(robot, max_wait_s=0.3, step_s=0.05, after_map=grid0)
        self.assertIs(new_grid, grid1)
        self.assertTrue(fresh1)

    def test_rehydrate_clearance_excludes_robot_and_charger_corridor(self) -> None:
        _install_fake_anki_util(self._patchers)
        atlas = NavMapAtlas(cell_mm=20.0)
        sub0 = atlas.ensure_submap("s0", origin_id=1)
        atlas.transforms_to_canonical["s0"] = Transform2D(0.0, 0.0, 0.0)
        # Charger at canonical (0, 0, yaw=0): corridor [-40..220, -90..90].
        atlas.anchors["s0"] = {"charger": Pose2D(0.0, 0.0, 0.0, origin_id=1)}
        # Box A inside charger corridor (150, 10) -> must be excluded.
        sub0.integrate_leaves([LeafPatch(150.0, 10.0, 20.0, int(NavContent.OBSTACLE_PROXIMITY))])
        # Box B near current robot pose in rotated bot frame -> must be excluded.
        # sub1 origin is rotated +90 deg relative to canonical: tf_to_can = (1000, 0, +pi/2). Robot at (0, 100) -> can (900, 0).
        sub0.integrate_leaves([LeafPatch(910.0, 10.0, 20.0, int(NavContent.OBSTACLE_PROXIMITY))])
        # Box C far away at canonical (-350, -350) -> must be kept and transformed to bot frame.
        sub0.integrate_leaves([LeafPatch(-350.0, -350.0, 20.0, int(NavContent.OBSTACLE_PROXIMITY))])

        atlas.ensure_submap("s1", origin_id=2)
        tf_s1 = Transform2D(1000.0, 0.0, math.pi / 2.0)
        atlas.transforms_to_canonical["s1"] = tf_s1
        chg = canonical_charger_pose(atlas)
        self.assertIsNotNone(chg)
        robot = _FakeRobot(origin_id=2)
        created = rehydrate_onboard_obstacles(
            robot,
            atlas,
            "s1",
            max_boxes=24,
            robot_pose_bot=Pose2D(0.0, 100.0, 0.0, origin_id=2),
            charger_pose_can=chg,
        )
        self.assertEqual(created, 1)
        self.assertEqual(robot.world.deleted_fixed, 1)
        pose_sent = robot.world.fixed_calls[0]["pose"]
        self.assertEqual(pose_sent.origin_id, 2)
        exp_bx, exp_by = tf_s1.invert().apply_point(-350.0, -350.0)
        self.assertAlmostEqual(pose_sent.x, exp_bx, delta=15.0)
        self.assertAlmostEqual(pose_sent.y, exp_by, delta=15.0)

    def test_stationary_docked_daemon_survives_over_120s_without_navmap_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            leaves = [
                _make_leaf(100.0, 200.0, 20.0, NavContent.CLEAR_OF_CLIFF),
                _make_leaf(100.0, 260.0, 20.0, NavContent.OBSTACLE_PROXIMITY),
            ]
            robot = _FakeRobot(origin_id=15, leaves=leaves)
            now = [1000.0]

            def _sleep(dt: float) -> None:
                now[0] += dt
                daemon.on_robot_state()

            cfg = BasalConfig(
                serial="00000000",
                atlas_path=root / "atlas.json",
                export_dir=root / "ros",
                sqlite_path=root / "native.sqlite",
                navmap_json_path=root / "shelf-navmap.json",
                max_ticks=250,
                poll_sleep_s=0.5,
                state_stale_s=45.0,
                export_interval_s=30.0,
            )
            daemon = BasalMapDaemon(
                cfg,
                sleep_fn=_sleep,
                clock_fn=lambda: now[0],
                wall_fn=lambda: 1700000000.0,
                session_factory=_const_session_factory(robot),
            )
            with mock.patch.object(daemon, "_handle_grid", wraps=daemon._handle_grid) as spy:
                rc = daemon.run()
            self.assertEqual(rc, 0)
            self.assertGreaterEqual(now[0] - 1000.0, 120.0)
            self.assertEqual(spy.call_count, 1)
            payload = json.loads((root / "shelf-navmap.json").read_text(encoding="utf-8"))
            self.assertEqual((payload["cell_mm"], payload["frame"], payload["revisions"]), (10, "seated_robot", 1))
            self.assertEqual(payload["unregistered_revisions"], 0)
            self.assertTrue((root / "ros" / "grid.pgm").is_file() and (root / "ros" / "grid.yaml").is_file())

    def test_interesting_edge_does_not_accumulate_into_obstacle_across_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            robot = _FakeRobot(origin_id=7, leaves=[])
            tick = [0]
            now = [100.0]

            class _FakeSess:
                def __init__(self, **_kw: object) -> None:
                    self.robot = robot

                def __enter__(self) -> _FakeRobot:
                    return self.robot

                def __exit__(self, *_args: object) -> bool:
                    return False

            def _sleep(dt: float) -> None:
                now[0] += dt
                tick[0] += 1
                daemon.on_robot_state()
                if tick[0] <= 5:
                    leaves = [
                        _make_leaf(400.0, 400.0, 20.0, NavContent.INTERESTING_EDGE),
                        _make_leaf(600.0 + tick[0] * 40.0, 0.0, 20.0, NavContent.CLEAR_OF_OBSTACLE),
                    ]
                    if tick[0] == 5:
                        leaves.append(_make_leaf(0.0, 400.0, 20.0, NavContent.OBSTACLE_PROXIMITY))
                    robot.nav_map._latest_nav_map = _make_grid(7, leaves)

            cfg = BasalConfig(
                serial="00000000",
                atlas_path=root / "atlas.json",
                sqlite_path=root / "native.sqlite",
                max_ticks=6,
                poll_sleep_s=0.5,
            )
            daemon = BasalMapDaemon(cfg, sleep_fn=_sleep, clock_fn=lambda: now[0], wall_fn=lambda: 1700000000.0, session_factory=_FakeSess)
            self.assertEqual(daemon.run(), 0)
            atlas = NavMapAtlas.load(root / "atlas.json")
            fused = atlas.fuse_canonical_cells()
            edge_stats = fused[(20, 20)]
            self.assertAlmostEqual(edge_stats.log_odds, 0.15, places=4)
            boxes = compress_occupied_to_boxes(atlas)
            self.assertEqual(len(boxes), 1)
            self.assertAlmostEqual(boxes[0][1], 400.0, delta=25.0)

    def test_robot_state_stall_exits_5_and_flushes_navmap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            leaves = [_make_leaf(100.0, 200.0, 20.0, NavContent.CLEAR_OF_CLIFF)]
            robot = _FakeRobot(origin_id=15, leaves=leaves)
            now = [0.0]

            class _FakeSess:
                def __init__(self, **_kw: object) -> None:
                    pass

                def __enter__(self) -> _FakeRobot:
                    return robot

                def __exit__(self, *_args: object) -> bool:
                    return False

            cfg = BasalConfig(
                serial="00000000",
                atlas_path=root / "atlas.json",
                sqlite_path=root / "native.sqlite",
                navmap_json_path=root / "shelf-navmap.json",
                max_ticks=20,
                poll_sleep_s=10.0,
                state_stale_s=45.0,
                export_interval_s=999.0,
            )
            daemon = BasalMapDaemon(cfg, sleep_fn=lambda dt: now.__setitem__(0, now[0] + dt), clock_fn=lambda: now[0], wall_fn=lambda: 1700000000.0, session_factory=_FakeSess)
            rc = daemon.run()
            self.assertEqual(rc, 5)
            self.assertTrue((root / "shelf-navmap.json").is_file())
            self.assertEqual(len(robot.events.subs), 0)

    def test_rehydrate_origin_switch_and_cryosleep_calm_gating(self) -> None:
        _install_fake_anki_util(self._patchers)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prior = NavMapAtlas(cell_mm=20.0)
            s_prev = prior.ensure_submap("prev:origin_15", origin_id=15, robot_id="00000000")
            prior.assume_same_frame_as_canonical("prev:origin_15")
            s_prev.integrate_leaves([
                LeafPatch(100.0, 200.0, 20.0, int(NavContent.CLEAR_OF_CLIFF)),
                LeafPatch(500.0, 500.0, 20.0, int(NavContent.OBSTACLE_PROXIMITY)),
            ])
            prior.record_anchor("prev:origin_15", "charger_seated", Pose2D(100.0, 200.0, math.pi / 2.0, origin_id=15), assume_canonical=True)
            prior.save(root / "atlas.json")

            for calm, muted, exp_calls in ((True, False, 0), (False, True, 0), (False, False, 1)):
                robot = _FakeRobot(origin_id=16, leaves=[_make_leaf(100.0, 200.0, 20.0, NavContent.CLEAR_OF_CLIFF)])
                robot.status = _make_status(on_charger=True, calm=calm)

                class _FakeSess:
                    def __init__(self, **_kw: object) -> None:
                        pass

                    def __enter__(self) -> _FakeRobot:
                        return robot

                    def __exit__(self, *_args: object) -> bool:
                        return False

                cfg = BasalConfig(
                    serial="00000000",
                    atlas_path=root / "atlas.json",
                    pod_url="http://127.0.0.1:8080",
                    max_ticks=2,
                    poll_sleep_s=0.1,
                    rehydrate_on_new_origin=True,
                )
                daemon = BasalMapDaemon(cfg, sleep_fn=lambda _dt: daemon.on_robot_state(), session_factory=_FakeSess)
                with mock.patch("vector_slam_maze.basal_persist.check_pod_cryosleep", return_value=muted):
                    self.assertEqual(daemon.run(), 0)
                self.assertEqual(len(robot.world.fixed_calls), exp_calls)

    def test_cryosleep_http_and_seated_export_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            robot = _FakeRobot(
                origin_id=3,
                leaves=[
                    _make_leaf(100.0, 200.0, 20.0, NavContent.CLEAR_OF_CLIFF),
                    _make_leaf(100.0, 220.0, 20.0, NavContent.CLIFF),
                    _make_leaf(100.0, 220.0, 20.0, NavContent.CLEAR_OF_CLIFF),
                ],
            )
            daemon = BasalMapDaemon(BasalConfig(serial="00000000", atlas_path=root / "a.json", sqlite_path=root / "d.sqlite"))
            from vector_slam_maze.basal_persist import init_basal_db

            daemon._db = init_basal_db(root / "d.sqlite")
            try:
                daemon._handle_grid(robot, robot.nav_map._latest_nav_map)
            finally:
                daemon._db.close()
                daemon._db = None
            data = export_seated_navmap(root / "d.sqlite", root / "out.json")
            self.assertEqual(data, export_seated_navmap_dict(root / "d.sqlite"))
            self.assertTrue(any(cell[3] == 1 for cell in data["cells"]))

        with mock.patch("vector_slam_maze.basal_persist._http_pod_json", return_value=(502, {"pod_gated": True})):
            self.assertTrue(check_pod_cryosleep("http://127.0.0.1:8080", "00000000"))
        with mock.patch("vector_slam_maze.basal_persist._http_pod_json", return_value=(404, None)):
            self.assertFalse(check_pod_cryosleep("http://127.0.0.1:8080", "00000000"))

    def test_cli_basal_daemon_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seen: dict[str, object] = {}

            class _FakeDaemon:
                def __init__(self, cfg: BasalConfig) -> None:
                    seen["cfg"] = cfg

                def run(self) -> int:
                    return 0

            with mock.patch("vector_slam_maze.cli.BasalMapDaemon", side_effect=_FakeDaemon):
                rc = cli_main([
                    "--serial", "00000000",
                    "--atlas", str(root / "a.json"),
                    "--export-dir", "",
                    "basal-daemon",
                    "--pod-url", "http://127.0.0.1:8080",
                    "--sqlite", str(root / "n.sqlite"),
                    "--export-navmap-json", str(root / "s.json"),
                    "--ensure-pod-timeouts",
                    "--rehydrate-on-new-origin",
                    "--max-ticks", "4",
                ])
            self.assertEqual(rc, 0)
            cfg = seen["cfg"]
            self.assertIsInstance(cfg, BasalConfig)
            self.assertIsNone(cfg.export_dir)
            self.assertTrue(cfg.ensure_pod_timeouts and cfg.rehydrate_on_new_origin)

    def test_cli_rehydrate_reports_content3_and_fresh(self) -> None:
        _install_fake_anki_util(self._patchers)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atlas = NavMapAtlas(cell_mm=20.0)
            s0 = atlas.ensure_submap("s0", origin_id=9)
            atlas.assume_same_frame_as_canonical("s0")
            s0.integrate_leaves([LeafPatch(-400.0, -400.0, 20.0, int(NavContent.OBSTACLE_PROXIMITY))])
            atlas.save(root / "rehyd.json")
            grid0 = _make_grid(9, [_make_leaf(-400.0, -400.0, 20.0, NavContent.OBSTACLE_PROXIMITY)])
            grid1 = _make_grid(9, [_make_leaf(-400.0, -400.0, 20.0, NavContent.OBSTACLE_CUBE)])
            robot = _FakeRobot(origin_id=9)
            robot.nav_map._latest_nav_map = grid0

            def _after(*_a: object, **_kw: object) -> tuple[object, bool]:
                return (grid1, True)

            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with (
                mock.patch("vector_slam_maze.cli.VectorSDKSession", _const_session_factory(robot)),
                mock.patch("vector_slam_maze.cli.poll_fresh_nav_map", side_effect=_after),
                redirect_stdout(buf),
            ):
                rc = cli_main(["--serial", "00000000", "--atlas", str(root / "rehyd.json"), "--assume-docked-canonical", "rehydrate", "--via-vector"])
            self.assertEqual(rc, 0)
            res = json.loads(buf.getvalue())
            self.assertTrue(res["fresh_broadcast"])
            self.assertEqual((res["obstacle_cube_leaves_before"], res["obstacle_cube_leaves_after"]), (0, 1))
