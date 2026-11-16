"""Tests for the active-session cliff guard. Authored-by: CPCS."""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from vector_slam_maze import cliff_guard as cg
from vector_slam_maze.sdk_session import VectorSDKSession

DEFAULT, OVERRIDE = "DEFAULT_PRIORITY", "OVERRIDE_BEHAVIORS_PRIORITY"


class FakeConn:
    """Like the SDK: has_control is never cleared; control_granted_event is."""

    def __init__(self) -> None:
        self.has_control = True
        self.control_granted_event = threading.Event()
        self.control_granted_event.set()
        self.requests: list = []

    def lose(self) -> None:
        self.control_granted_event.clear()

    def request_control(self, behavior_control_level=None, timeout=None) -> None:
        self.requests.append(behavior_control_level)
        self.control_granted_event.set()


class FakeBehavior:
    def __init__(self) -> None:
        self.calls: list = []
        self.during = None  # called while a motion "runs"
        self.raise_cancel = False
        self.result_code = None  # if set, return a reply carrying this action result

    def _run(self, kind: str, amount: float) -> str:
        self.calls.append((kind, amount))
        if self.during:
            self.during()
        if self.raise_cancel:
            raise concurrent.futures.CancelledError()
        if self.result_code is not None:
            return SimpleNamespace(result=SimpleNamespace(code=self.result_code))
        return "done"

    def drive_straight(self, distance, speed, **kw):
        return self._run("drive", cg._mm(distance))

    def turn_in_place(self, angle, **kw):
        return self._run("turn", cg._deg(angle))

    def set_eye_color(self, **kw):
        return None


class FakeEvents:
    def __init__(self) -> None:
        self.subs: list = []

    def subscribe(self, fn, event) -> None:
        self.subs.append((fn, event))


def make_robot():
    robot = SimpleNamespace(behavior=FakeBehavior(), conn=FakeConn(), events=FakeEvents(), status="status-passthrough")
    guard = cg.CliffGuard(robot, DEFAULT)
    guard.attach("robot_state")
    return robot, guard, cg.GuardedRobot(robot, guard)


def state(bits: int) -> SimpleNamespace:
    return SimpleNamespace(status=bits)


class StatusBitsTest(unittest.TestCase):
    def test_bits_match_the_installed_sdk(self) -> None:
        try:
            from anki_vector.messaging import protocol  # type: ignore
        except ImportError:
            self.skipTest("anki_vector not installed")
        rs = protocol.RobotStatus
        self.assertEqual(cg.STATUS_PICKED_UP, rs.Value("ROBOT_STATUS_IS_PICKED_UP"))
        self.assertEqual(cg.STATUS_FALLING, rs.Value("ROBOT_STATUS_IS_FALLING"))
        self.assertEqual(cg.STATUS_CLIFF_DETECTED, rs.Value("ROBOT_STATUS_CLIFF_DETECTED"))

    def test_hazard_of(self) -> None:
        self.assertIsNone(cg.hazard_of(0))
        self.assertEqual(cg.hazard_of(cg.STATUS_CLIFF_DETECTED | 1), "cliff_detected")
        self.assertEqual(cg.hazard_of(cg.STATUS_FALLING), "falling")
        self.assertEqual(cg.hazard_of(cg.STATUS_PICKED_UP), "picked_up")


class GuardTest(unittest.TestCase):
    def test_quiet_drive_runs(self) -> None:
        robot, guard, g = make_robot()
        self.assertEqual(g.behavior.drive_straight(120.0, 60.0), "done")
        self.assertEqual(robot.behavior.calls, [("drive", 120.0)])
        self.assertIsNone(guard.lockout)

    def test_cliff_during_drive_raises(self) -> None:
        robot, guard, g = make_robot()
        robot.behavior.during = lambda: guard.on_robot_state(robot, "robot_state", state(cg.STATUS_CLIFF_DETECTED))
        with self.assertRaisesRegex(cg.CliffHazard, "cliff_detected"):
            g.behavior.drive_straight(200.0, 60.0)
        self.assertEqual(guard.lockout, "cliff_detected")

    def test_reflex_cancel_becomes_cliff_hazard(self) -> None:
        robot, guard, g = make_robot()

        def reflex() -> None:
            guard.on_robot_state(robot, "robot_state", state(cg.STATUS_CLIFF_DETECTED))
            robot.conn.lose()

        robot.behavior.during = reflex
        robot.behavior.raise_cancel = True
        with self.assertRaisesRegex(cg.CliffHazard, "interrupted: cliff_detected"):
            g.behavior.drive_straight(200.0, 60.0)

    def test_forward_refused_until_turn_or_back_up(self) -> None:
        robot, guard, g = make_robot()
        guard.on_robot_state(robot, "robot_state", state(cg.STATUS_CLIFF_DETECTED))
        guard.on_robot_state(robot, "robot_state", state(0))
        with self.assertRaisesRegex(cg.CliffHazard, "locked"):
            g.behavior.drive_straight(100.0, 60.0)
        self.assertEqual(robot.behavior.calls, [])
        g.behavior.turn_in_place(30.0)
        with self.assertRaises(cg.CliffHazard):
            g.behavior.drive_straight(100.0, 60.0)
        g.behavior.turn_in_place(-90.0)
        self.assertEqual(g.behavior.drive_straight(100.0, 60.0), "done")

    def test_backing_up_lifts_lockout(self) -> None:
        robot, guard, g = make_robot()
        guard.on_robot_state(robot, "robot_state", state(cg.STATUS_CLIFF_DETECTED))
        guard.on_robot_state(robot, "robot_state", state(0))
        g.behavior.drive_straight(-60.0, 50.0)
        self.assertIsNone(guard.lockout)

    def test_forward_refused_while_cliff_reported(self) -> None:
        robot, guard, g = make_robot()
        guard.on_robot_state(robot, "robot_state", state(cg.STATUS_CLIFF_DETECTED))
        with self.assertRaises(cg.CliffHazard):
            g.behavior.drive_straight(50.0, 50.0)

    def test_cancelled_result_without_cliff_sample_raises(self) -> None:
        robot, guard, g = make_robot()
        robot.behavior.during = robot.conn.lose  # reflex too quick for any robot_state sample
        robot.behavior.result_code = 33554432  # ACTION_RESULT_CANCELLED_WHILE_RUNNING
        with self.assertRaisesRegex(cg.CliffHazard, "interrupted: action_result_33554432"):
            g.behavior.drive_straight(200.0, 60.0)
        self.assertIsNotNone(guard.lockout)

    def test_losing_control_during_a_move_raises(self) -> None:
        robot, guard, g = make_robot()
        robot.behavior.during = robot.conn.lose
        with self.assertRaisesRegex(cg.CliffHazard, "interrupted: control_lost"):
            g.behavior.drive_straight(200.0, 60.0)
        with self.assertRaisesRegex(cg.CliffHazard, "locked"):
            g.behavior.drive_straight(50.0, 50.0)

    def test_control_requested_again_at_default(self) -> None:
        robot, guard, g = make_robot()
        robot.conn.lose()
        g.behavior.turn_in_place(90.0)
        self.assertEqual(robot.conn.requests, [DEFAULT])

    def test_other_attributes_pass_through(self) -> None:
        robot, guard, g = make_robot()
        self.assertEqual(g.status, "status-passthrough")
        self.assertIsNone(g.behavior.set_eye_color(hue=0.5))


class SessionTest(unittest.TestCase):
    def _fake_av(self, captured: dict):
        class FakeBot:
            def __init__(self, **kw):
                captured["kwargs"] = kw
                self.behavior = FakeBehavior()
                self.conn = FakeConn()
                self.events = FakeEvents()
                captured["bot"] = self

            def connect(self):
                pass

            def disconnect(self):
                pass

        levels = SimpleNamespace(DEFAULT_PRIORITY=DEFAULT, OVERRIDE_BEHAVIORS_PRIORITY=OVERRIDE)
        return SimpleNamespace(Robot=FakeBot, connection=SimpleNamespace(ControlPriorityLevel=levels),
                               events=SimpleNamespace(Events=SimpleNamespace(robot_state="robot_state")))

    def test_active_session_uses_default_priority_and_guard(self) -> None:
        captured: dict = {}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lock = os.path.join(tmp.name, "takeover.lock")
        with mock.patch("vector_slam_maze.sdk_session._import_anki_vector", return_value=self._fake_av(captured)):
            with mock.patch("vector_slam_maze.sdk_session.safe_disconnect"):
                with VectorSDKSession(serial="00e1e2e3", active=True, via_vector=True, enable_nav_map=False,
                                      lock_path=lock) as robot:
                    pass
        self.assertEqual(captured["kwargs"]["behavior_control_level"], DEFAULT)
        self.assertIsInstance(robot, cg.GuardedRobot)
        self.assertEqual([e for _, e in captured["bot"].events.subs], ["robot_state"])


if __name__ == "__main__":
    unittest.main()
