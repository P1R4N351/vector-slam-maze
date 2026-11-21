"""The cliff guard as the bounded movement module and the session see it.

Three things test_cliff_guard.py does not cover:
  * a move the guard stops or refuses comes back from movement_directives as a
    result with a reason and the distance measured, not as an exception;
  * the forward lockout is lifted by a real back-up, not by a 1 mm reverse;
  * a session whose connect fails does not leave its takeover lock behind.

Stand-in SDK objects only. Nothing here has run on a robot.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_movement_directives import _FakeRobot  # noqa: E402

from vector_slam_maze import cliff_guard as cg  # noqa: E402
from vector_slam_maze import movement_directives as md  # noqa: E402
from vector_slam_maze.sdk_session import VectorSDKSession  # noqa: E402

REFLEX_FRACTION = 0.4  # how much of the commanded chunk the stand-in robot covers before its reflex stops it


def guarded_robot(reflex_on_drive=None, lose_control_on_turn=False, cancel_on_drive=None):
    """A stand-in robot behind the real guard. Returns (raw robot, guarded robot)."""
    raw = _FakeRobot(prox_mm=900.0, prox_found=False)
    guard = cg.CliffGuard(raw, "DEFAULT_PRIORITY")
    drive, turn, count = raw.behavior.drive_straight, raw.behavior.turn_in_place, {"drive": 0}

    def drive_straight(dist, speed, *args, **kwargs):
        count["drive"] += 1
        mm = float(getattr(dist, "distance_mm", 0.0))
        if count["drive"] == cancel_on_drive:
            raise concurrent.futures.CancelledError()
        if count["drive"] == reflex_on_drive:
            raw.x += REFLEX_FRACTION * mm * math.cos(raw.theta)
            raw.y += REFLEX_FRACTION * mm * math.sin(raw.theta)
            guard.on_robot_state(raw, None, SimpleNamespace(status=cg.STATUS_CLIFF_DETECTED))
            guard.on_robot_state(raw, None, SimpleNamespace(status=0))  # the back-off clears the bit
            return SimpleNamespace(result=SimpleNamespace(code=3))
        drive(dist, speed, *args, **kwargs)
        return SimpleNamespace(result=SimpleNamespace(code=0))

    def turn_in_place(angle, *args, **kwargs):
        turn(angle)
        if lose_control_on_turn:
            raw.conn.control_granted_event.clear()
        return SimpleNamespace(result=SimpleNamespace(code=0))

    granted = SimpleNamespace(state=True)
    granted.is_set = lambda: granted.state
    granted.clear = lambda: setattr(granted, "state", False)
    raw.conn = SimpleNamespace(control_granted_event=granted, requests=[])
    raw.conn.request_control = lambda behavior_control_level=None, timeout=None: (
        raw.conn.requests.append(behavior_control_level), setattr(granted, "state", True))
    raw.behavior.drive_straight, raw.behavior.turn_in_place = drive_straight, turn_in_place
    return raw, cg.GuardedRobot(raw, guard)


class GuardedMovementTest(unittest.TestCase):
    def test_a_quiet_drive_is_unchanged(self) -> None:
        raw, robot = guarded_robot()
        res = md.bounded_drive(robot, 300.0)
        self.assertEqual((res.ok, res.reason, res.detail), (True, md.STOP_COMPLETED, "goal"))
        self.assertAlmostEqual(res.moved_mm, 300.0, places=3)

    def test_a_reflex_stop_is_a_result_with_the_distance_measured(self) -> None:
        raw, robot = guarded_robot(reflex_on_drive=2)
        res = md.bounded_drive(robot, 500.0)
        self.assertEqual((res.ok, res.reason, res.detail, res.chunks), (False, md.STOP_CLIFF, "cliff", 2))
        self.assertAlmostEqual(res.moved_mm, 180.0 + REFLEX_FRACTION * 180.0, places=3)
        self.assertAlmostEqual(res.moved_mm, raw.x, places=3)
        self.assertEqual(robot.cliff_guard.lockout, "cliff_detected")

    def test_a_cancelled_action_is_reported_as_lost_control(self) -> None:
        _raw, robot = guarded_robot(cancel_on_drive=1)
        res = md.bounded_drive(robot, 100.0)
        self.assertEqual((res.ok, res.reason, res.detail), (False, md.STOP_CLIFF, "control_lost"))
        self.assertAlmostEqual(res.moved_mm, 0.0, places=3)

    def test_forward_after_a_cliff_is_refused_without_a_motor_call(self) -> None:
        raw, robot = guarded_robot(reflex_on_drive=1)
        md.bounded_drive(robot, 100.0)
        before = list(raw.behavior.calls)
        res = md.bounded_drive(robot, 100.0)
        self.assertEqual((res.ok, res.reason, res.detail), (False, md.STOP_REFUSED, "cliff_lockout"))
        self.assertAlmostEqual(res.moved_mm, 0.0, places=3)
        self.assertEqual(raw.behavior.calls, before)
        act = md.execute_bounded_directive(robot, "move", {"dx_mm": 100.0})
        self.assertEqual((act["ok"], act["stopped_reason"], act["motion_result"]["reason"]), (False, "cliff_lockout", md.STOP_REFUSED))

    def test_only_a_real_back_up_lifts_the_lockout(self) -> None:
        _raw, robot = guarded_robot(reflex_on_drive=1)
        md.bounded_drive(robot, 100.0)
        for short in (-1.0, -(cg.CLEAR_BACKUP_MM - 1.0)):
            self.assertTrue(md.bounded_drive(robot, short).ok)
            self.assertEqual(robot.cliff_guard.lockout, "cliff_detected", short)
            self.assertEqual(md.bounded_drive(robot, 100.0).detail, "cliff_lockout")
        self.assertTrue(md.bounded_drive(robot, -cg.CLEAR_BACKUP_MM).ok)
        self.assertIsNone(robot.cliff_guard.lockout)
        self.assertTrue(md.bounded_drive(robot, 100.0).ok)

    def test_a_turn_that_loses_control_is_a_result_with_the_yaw_measured(self) -> None:
        _raw, robot = guarded_robot(lose_control_on_turn=True)
        res = md.bounded_turn(robot, math.radians(30.0))
        self.assertEqual((res.ok, res.reason, res.detail), (False, md.STOP_CLIFF, "control_lost"))
        self.assertAlmostEqual(res.turned_rad, math.radians(30.0), places=4)

    def test_hazard_names_map_to_stop_reasons(self) -> None:
        cases = [(cg.CliffHazard("x", hazard="cliff_detected"), (md.STOP_CLIFF, "cliff")),
                 (cg.CliffHazard("x", hazard="control_lost"), (md.STOP_CLIFF, "control_lost")),
                 (cg.CliffHazard("x", hazard="action_result_3"), (md.STOP_CLIFF, "control_lost")),
                 (cg.CliffHazard("x", hazard="falling"), (md.STOP_REFUSED, "falling")),
                 (cg.CliffHazard("x", hazard="picked_up"), (md.STOP_REFUSED, "picked_up")),
                 (cg.CliffHazard("x", hazard="cliff_detected", refused=True), (md.STOP_REFUSED, "cliff_lockout"))]
        for exc, expected in cases:
            self.assertEqual(md._hazard_stop(exc), expected, exc.hazard)

    def test_the_guard_says_what_happened_and_whether_it_started(self) -> None:
        _raw, robot = guarded_robot(reflex_on_drive=1)
        with self.assertRaises(cg.CliffHazard) as stopped:
            robot.behavior.drive_straight(SimpleNamespace(distance_mm=100.0), SimpleNamespace(speed_mmps=50.0))
        self.assertEqual((stopped.exception.hazard, stopped.exception.refused), ("cliff_detected", False))
        with self.assertRaises(cg.CliffHazard) as refused:
            robot.behavior.drive_straight(SimpleNamespace(distance_mm=100.0), SimpleNamespace(speed_mmps=50.0))
        self.assertEqual((refused.exception.hazard, refused.exception.refused), ("cliff_detected", True))


class SessionLockTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.lock = os.path.join(tmp.name, "takeover.lock")
        self.events: list = []

    def fake_av(self, fail_connect: bool = False, fail_eyes: bool = False, grant_control: bool = True,
                cliff: bool = False):
        events = self.events

        class FakeBot:
            def __init__(self, **kwargs):
                self.conn = SimpleNamespace(control_granted_event=SimpleNamespace(is_set=lambda: grant_control))
                self.status = SimpleNamespace(is_cliff_detected=cliff)
                self.events = SimpleNamespace(subscribe=lambda fn, event: None)
                self.behavior = SimpleNamespace(set_eye_color=self._eyes)

            def _eyes(self, **kwargs):
                if fail_eyes:
                    raise RuntimeError("eyes failed")

            def connect(self):
                events.append("connect")
                if fail_connect:
                    raise RuntimeError("robot unreachable")

            def disconnect(self):
                events.append("disconnect")

        levels = SimpleNamespace(DEFAULT_PRIORITY="DEFAULT_PRIORITY", OVERRIDE_BEHAVIORS_PRIORITY="OVERRIDE")
        return SimpleNamespace(Robot=FakeBot, connection=SimpleNamespace(ControlPriorityLevel=levels),
                               events=SimpleNamespace(Events=SimpleNamespace(robot_state="robot_state")))

    def open_session(self, **faults):
        with mock.patch("vector_slam_maze.sdk_session._import_anki_vector", return_value=self.fake_av(**faults)):
            with mock.patch("vector_slam_maze.sdk_session._release_active_control"):
                with VectorSDKSession(serial="00e1e2e3", active=True, via_vector=True, enable_nav_map=False,
                                      lock_path=self.lock):
                    self.events.append("inside:lock=%s" % os.path.exists(self.lock))

    def test_a_failed_connect_releases_the_lock(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "robot unreachable"):
            self.open_session(fail_connect=True)
        self.assertFalse(os.path.exists(self.lock))
        self.open_session()  # and the next session is not refused
        self.assertIn("inside:lock=True", self.events)

    def test_a_failure_after_connect_disconnects_and_releases_the_lock(self) -> None:
        with mock.patch.object(VectorSDKSession, "_set_takeover_eyes", side_effect=RuntimeError("eyes failed")):
            with self.assertRaisesRegex(RuntimeError, "eyes failed"):
                self.open_session()
        self.assertEqual(self.events, ["connect", "disconnect"])
        self.assertFalse(os.path.exists(self.lock))

    def test_a_session_the_robot_gives_no_control_to_is_refused_before_anything_can_block(self) -> None:
        with self.assertRaises(cg.CliffHazard) as caught:
            self.open_session(grant_control=False)
        self.assertEqual((caught.exception.hazard, caught.exception.refused), ("control_refused", True))
        self.assertEqual(self.events, ["connect", "disconnect"])
        self.assertFalse(os.path.exists(self.lock))

    def test_a_refusal_at_an_edge_names_the_cliff_and_says_to_move_the_robot(self) -> None:
        with self.assertRaisesRegex(cg.CliffHazard, "move it away from the edge by hand") as caught:
            self.open_session(grant_control=False, cliff=True)
        self.assertEqual((caught.exception.hazard, caught.exception.refused), ("cliff_detected", True))
        self.assertFalse(os.path.exists(self.lock))

    def test_a_passive_session_is_not_asked_for_control(self) -> None:
        with mock.patch("vector_slam_maze.sdk_session._import_anki_vector", return_value=self.fake_av(grant_control=False)):
            with VectorSDKSession(serial="00e1e2e3", active=False, enable_nav_map=False, lock_path=self.lock):
                self.events.append("inside")
        self.assertEqual(self.events, ["connect", "inside", "disconnect"])

    def test_a_good_session_holds_the_lock_until_it_exits(self) -> None:
        self.open_session()
        self.assertEqual(self.events, ["connect", "inside:lock=True", "disconnect"])
        self.assertFalse(os.path.exists(self.lock))


class MazeAndSDKHardeningTests(unittest.TestCase):
    """Tests for future-aware control request, battery SDK fault handling, and maze CliffHazard halts."""

    def test_ensure_control_waits_for_returned_future(self) -> None:
        fut: concurrent.futures.Future[None] = concurrent.futures.Future()
        fut.set_result(None)
        wait_seen = []
        orig_result = fut.result

        def _tracked_result(timeout=None):
            wait_seen.append(timeout)
            return orig_result(timeout=timeout)

        fut.result = _tracked_result  # type: ignore[method-assign]
        granted = SimpleNamespace(is_set=lambda: False)
        conn = SimpleNamespace(control_granted_event=granted, request_control=lambda **_kw: fut)
        guard = cg.CliffGuard(SimpleNamespace(conn=conn), "DEFAULT")
        guard._ensure_control()
        self.assertEqual(wait_seen, [cg.CONTROL_TIMEOUT_S])

    def test_safe_battery_state_catches_sdk_exceptions(self) -> None:
        from vector_slam_maze import sdk_session

        class FakeUnimplError(Exception):
            pass

        bot = SimpleNamespace(get_battery_state=lambda: (_ for _ in ()).throw(FakeUnimplError("unimplemented")))
        with mock.patch("vector_slam_maze.sdk_session._sdk_exception_types", return_value=(FakeUnimplError,)):
            self.assertIsNone(sdk_session._safe_battery_state(bot))

    def test_maze_drivers_and_cli_return_structured_halt_on_cliff(self) -> None:
        import io
        from contextlib import redirect_stdout
        from vector_slam_maze import cli, maze_closed_loop, maze_solver, navmap_atlas

        atlas = navmap_atlas.NavMapAtlas()
        grid = maze_solver.MazeGrid(rows=2, cols=2, cell_size_mm=180.0)
        planner = maze_solver.FloodFillPlanner(grid)
        bot = _FakeRobot()
        bot.behavior.turn_in_place = lambda *_a, **_k: (_ for _ in ()).throw(
            cg.CliffHazard("stopped", hazard="control_lost")
        )
        res_wf = maze_solver.execute_wall_follow_run(bot, atlas, "s1:origin_1", max_steps=2)
        self.assertEqual((res_wf["status"], res_wf["reason"], res_wf["hazard"]), ("halted", "cliff", "control_lost"))

        fake_loc = {"cpose": md.Pose2D(90.0, 90.0, 0.0), "origin_id": 1, "tf_can": None}
        with mock.patch("vector_slam_maze.maze_closed_loop.fuse_and_localize_step", return_value=fake_loc), \
             mock.patch("vector_slam_maze.maze_closed_loop._run_floodfill_step", side_effect=cg.CliffHazard("refused", hazard="cliff_detected", refused=True)):
            res_ff = maze_closed_loop.run_floodfill_loop(bot, atlas, grid, planner, "s1:origin_1", (1, 1), 2, 180.0, None, False)
        self.assertEqual((res_ff["status"], res_ff["reason"]), ("halted", "cliff_lockout"))

        with mock.patch("vector_slam_maze.cli.VectorSDKSession", side_effect=cg.CliffHazard("parked", hazard="cliff_detected", refused=True)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["act", "move", "--via-vector", "--args-json", "{\"dx_mm\":50}"])
        self.assertEqual(rc, 6)
        import json as _json
        self.assertEqual(_json.loads(buf.getvalue())["stopped_reason"], "cliff_lockout")

    def test_none_return_from_sdk_cancelled_action_triggers_cliff_and_reacquires_control(self) -> None:
        import threading

        raw, robot = guarded_robot()

        def cancelled_drive(dist, speed, *args, **kwargs):
            # Simulate wirepod_vector_sdk connection.py:424-426 returning None before clearing control_granted_event
            timer = threading.Timer(0.015, raw.conn.control_granted_event.clear)
            timer.start()
            return None

        raw.behavior.drive_straight = cancelled_drive
        res1 = md.bounded_drive(robot, 200.0)
        self.assertEqual((res1.ok, res1.reason, res1.detail), (False, md.STOP_CLIFF, "control_lost"))
        self.assertEqual(robot.cliff_guard.lockout, "control_lost")
        self.assertFalse(raw.conn.control_granted_event.is_set())

        res_locked = md.bounded_drive(robot, 50.0)
        self.assertEqual((res_locked.ok, res_locked.reason, res_locked.detail), (False, md.STOP_REFUSED, "cliff_lockout"))

        def good_reverse(dist, speed, *args, **kwargs):
            mm = float(getattr(dist, "distance_mm", 0.0))
            raw.x += mm * math.cos(raw.theta)
            raw.y += mm * math.sin(raw.theta)
            return SimpleNamespace(result=SimpleNamespace(code=0))

        raw.behavior.drive_straight = good_reverse
        res_rev = md.bounded_drive(robot, -60.0)
        self.assertEqual((res_rev.ok, res_rev.reason), (True, md.STOP_COMPLETED))
        self.assertEqual(raw.conn.requests, ["DEFAULT_PRIORITY"])
        self.assertIsNone(robot.cliff_guard.lockout)

    def test_safe_disconnect_catches_sdk_control_exception_and_closes_transport(self) -> None:
        from vector_slam_maze import sdk_session

        class FakeControlError(Exception):
            pass

        fut: concurrent.futures.Future[None] = concurrent.futures.Future()
        fut.set_result(None)
        waits, closed = [], []
        orig_res = fut.result
        fut.result = lambda timeout=None: (waits.append(timeout), orig_res(timeout=timeout))[1]  # type: ignore[method-assign]
        bot = SimpleNamespace(
            motors=SimpleNamespace(stop_all_motors=lambda: (_ for _ in ()).throw(FakeControlError("no control"))),
            events=SimpleNamespace(close=lambda: closed.append("events")),
            conn=SimpleNamespace(release_control=lambda: fut, close=lambda: closed.append("conn")),
            disconnect=lambda: (_ for _ in ()).throw(FakeControlError("disable_all_vision_modes requires behavior control")),
        )
        with mock.patch("vector_slam_maze.sdk_session._sdk_exception_types", return_value=(FakeControlError,)):
            sdk_session.safe_disconnect(bot, active=True)
        self.assertEqual(waits, [2.0])
        self.assertEqual(closed, ["events", "conn"])


if __name__ == "__main__":
    unittest.main()
