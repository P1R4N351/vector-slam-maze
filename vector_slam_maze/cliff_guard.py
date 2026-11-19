"""Cliff guard for active SDK sessions. Authored-by: CPCS.

The session takes behavior control at DEFAULT_PRIORITY, which runs below
Vector's mandatory physical reactions: when a cliff sensor fires, the robot
stops and backs away by itself, takes control from the SDK and cancels the
running action. (OVERRIDE_BEHAVIORS_PRIORITY suppresses that reaction; the
SDK documents it as "will drive off table".)

This module handles what the SDK leaves to the caller after such a reaction:

* a drive the reflex cancelled raises CliffHazard, whether the SDK reports
  it as a CancelledError or as a reply whose action result is not success;
* control is requested again before the next motion command;
* forward drives are refused after a cliff until the robot has turned at
  least CLEAR_TURN_DEG or backed up at least CLEAR_BACKUP_MM, so a loop cannot
  drive at the same edge again;
* a drive during which a cliff, fall or pick-up was reported raises
  CliffHazard even if the SDK call itself returned.

Hazards are read from the robot_state event stream, and losing control
while a motion runs counts as one: at DEFAULT priority only a mandatory
physical reaction takes control away. The cliff bit alone is not enough,
because the reflex backs away within a few milliseconds and the bit can fall
between two robot_state samples. Control is judged by the connection's
control_granted_event; Connection.has_control is not cleared when control is
lost (wirepod_vector_sdk 0.8.1).
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any, Optional

CLEAR_TURN_DEG = 45.0
CLEAR_BACKUP_MM = 30.0  # a shorter reverse does not lift the forward lockout
CONTROL_TIMEOUT_S = 10.0
CONTROL_LOSS_SETTLE_POLLS = 20
CONTROL_LOSS_SETTLE_S = 0.005

# anki_vector RobotStatus bits, read from wirepod_vector_sdk 0.8.1
# (protocol.RobotStatus) and pinned by a test against the installed SDK.
STATUS_PICKED_UP = 0x8
STATUS_FALLING = 0x20
STATUS_CLIFF_DETECTED = 0x4000


class CliffHazard(RuntimeError):
    """A motion was stopped or refused because of a cliff, fall or pick-up.

    hazard names what happened (cliff_detected, falling, picked_up,
    control_lost or action_result_<code>). refused is True when the motion
    was never started.
    """

    def __init__(self, message: str, hazard: str = "unknown", refused: bool = False) -> None:
        super().__init__(message)
        assert isinstance(hazard, str) and hazard, "hazard name required"
        self.hazard = hazard
        self.refused = bool(refused)


def hazard_of(status: int) -> Optional[str]:
    """Name the hazard in a robot_state status word, or None."""
    if status & STATUS_CLIFF_DETECTED:
        return "cliff_detected"
    if status & STATUS_FALLING:
        return "falling"
    if status & STATUS_PICKED_UP:
        return "picked_up"
    return None


def _mm(distance: Any) -> float:
    return float(getattr(distance, "distance_mm", distance))


def _deg(angle: Any) -> float:
    return float(getattr(angle, "degrees", angle))


class CliffGuard:
    """Tracks hazards from robot_state and gates motion commands."""

    def __init__(self, robot: Any, default_priority: Any) -> None:
        assert robot is not None
        self._robot = robot
        self._priority = default_priority
        self._lock = threading.Lock()
        self._current: Optional[str] = None  # hazard in the latest state
        self._seen: Optional[str] = None  # first hazard since the last motion began
        self._lockout: Optional[str] = None  # why forward motion is refused

    def on_robot_state(self, _robot: Any, _event_type: Any, msg: Any) -> None:
        """robot_state event handler."""
        h = hazard_of(int(getattr(msg, "status", 0)))
        with self._lock:
            self._current = h
            if h is not None:
                if self._seen is None:
                    self._seen = h
                self._lockout = self._lockout or h

    def attach(self, robot_state_event: Any) -> None:
        """Subscribe to robot_state (anki_vector.events.Events.robot_state)."""
        self._robot.events.subscribe(self.on_robot_state, robot_state_event)

    @property
    def lockout(self) -> Optional[str]:
        with self._lock:
            return self._lockout

    def _begin(self, forward: bool, name: str) -> None:
        with self._lock:
            if forward and self._lockout is not None:
                raise CliffHazard(
                    f"{name} refused: forward motion locked after {self._lockout}; "
                    f"turn at least {CLEAR_TURN_DEG:.0f} degrees or back up "
                    f"at least {CLEAR_BACKUP_MM:.0f} mm first", hazard=self._lockout, refused=True)
            if forward and self._current is not None:
                raise CliffHazard(f"{name} refused: robot reports {self._current}", hazard=self._current, refused=True)
            self._seen = None
        self._ensure_control()

    def _has_control(self) -> bool:
        conn = getattr(self._robot, "conn", None)
        granted = getattr(conn, "control_granted_event", None)
        if granted is None:
            return True
        return bool(granted.is_set())

    def _ensure_control(self) -> None:
        if self._has_control():
            return
        res = self._robot.conn.request_control(behavior_control_level=self._priority, timeout=CONTROL_TIMEOUT_S)
        if hasattr(res, "result"):
            try:
                res.result(timeout=CONTROL_TIMEOUT_S)
            except (concurrent.futures.TimeoutError, TimeoutError, concurrent.futures.CancelledError):
                pass

    def _interrupted(self, name: str, why: str) -> CliffHazard:
        with self._lock:
            seen = self._seen or why
            self._lockout = self._lockout or seen
        return CliffHazard(f"{name} interrupted: {seen}", hazard=seen)

    def _await_control_clear(self) -> None:
        """Wait up to 100 ms for connection.py to clear control_granted_event after _cancel_active."""
        assert CONTROL_LOSS_SETTLE_POLLS == 20 and CONTROL_LOSS_SETTLE_S > 0.0
        for _ in range(CONTROL_LOSS_SETTLE_POLLS):
            if not self._has_control():
                return
            time.sleep(CONTROL_LOSS_SETTLE_S)

    def _end(self, name: str, clears: bool, result: Any) -> None:
        if result is None:
            self._await_control_clear()
            raise self._interrupted(name, "control_lost")
        code = int(getattr(getattr(result, "result", None), "code", 0) or 0)
        if code != 0:
            raise self._interrupted(name, f"action_result_{code}")
        if not self._has_control():
            raise self._interrupted(name, "control_lost")
        with self._lock:
            seen = self._seen
            if seen is None and clears:
                self._lockout = None
        if seen is not None:
            raise CliffHazard(f"{name} stopped: {seen}", hazard=seen)

    def run(self, name: str, forward: bool, clears: bool, call: Any) -> Any:
        """Run one motion command under the guard."""
        self._begin(forward, name)
        try:
            result = call()
        except concurrent.futures.CancelledError as exc:
            raise self._interrupted(name, "control_lost") from exc
        self._end(name, clears, result)
        return result


class GuardedBehavior:
    """robot.behavior with drive_straight and turn_in_place under the guard."""

    def __init__(self, behavior: Any, guard: CliffGuard) -> None:
        self._behavior = behavior
        self._guard = guard

    def __getattr__(self, name: str) -> Any:
        return getattr(self._behavior, name)

    def drive_straight(self, distance: Any, speed: Any, *args: Any, **kwargs: Any) -> Any:
        mm = _mm(distance)
        return self._guard.run("drive_straight", mm > 0, mm <= -CLEAR_BACKUP_MM,
                               lambda: self._behavior.drive_straight(distance, speed, *args, **kwargs))

    def turn_in_place(self, angle: Any, *args: Any, **kwargs: Any) -> Any:
        clears = abs(_deg(angle)) >= CLEAR_TURN_DEG
        return self._guard.run("turn_in_place", False, clears,
                               lambda: self._behavior.turn_in_place(angle, *args, **kwargs))


class GuardedRobot:
    """The robot as callers see it in an active session: behavior is guarded,
    everything else is the SDK robot."""

    def __init__(self, robot: Any, guard: CliffGuard) -> None:
        self._robot = robot
        self.cliff_guard = guard
        self._behavior = GuardedBehavior(robot.behavior, guard)

    @property
    def behavior(self) -> GuardedBehavior:
        return self._behavior

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)
