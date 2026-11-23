"""Tests for bounded Vector translation, movement directives, lazy imports, and CLI dry-run."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, List, Tuple
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vector_slam_maze.cli import main as cli_main
from vector_slam_maze.movement_directives import (
    STOP_BLOCKED,
    STOP_CLIFF,
    STOP_COMPLETED,
    STOP_REFUSED,
    MotionResult,
    bounded_drive,
    bounded_turn,
    execute_bounded_directive,
    execute_directive_batch,
    plan_directive,
    plan_directive_batch,
)


@dataclass
class _Pos:
    x: float
    y: float


@dataclass
class _Ang:
    radians: float

    @property
    def degrees(self) -> float:
        """Return degrees matching anki_vector.util.Angle."""
        assert math.isfinite(self.radians) and abs(self.radians) <= 100.0
        return math.degrees(self.radians)


@dataclass
class _Rot:
    angle_z: _Ang


@dataclass
class _Pose:
    position: _Pos
    rotation: _Rot
    origin_id: int = 0


class _FakeBehavior:
    def __init__(self, robot: "_FakeRobot") -> None:
        self._r = robot
        self.calls: List[Tuple[str, float]] = []

    def turn_in_place(self, angle: Any) -> None:
        """Rotate fake robot yaw in place."""
        deg = float(getattr(angle, "degrees", 0.0))
        assert math.isfinite(deg) and abs(deg) <= 720.0
        self.calls.append(("turn", deg))
        self._r.theta = math.atan2(math.sin(self._r.theta + math.radians(deg)), math.cos(self._r.theta + math.radians(deg)))
        if self._r.prox_on_turn is not None:
            self._r.prox_mm, self._r.prox_found = self._r.prox_on_turn

    def drive_straight(self, dist: Any, speed: Any) -> None:
        """Advance fake robot along heading, simulating optional cliff, slip, or origin jump."""
        mm = float(getattr(dist, "distance_mm", 0.0))
        spd = float(getattr(speed, "speed_mmps", 0.0))
        assert math.isfinite(mm) and spd > 0.0
        self.calls.append(("drive", round(mm, 3)))
        idx = self._r.drive_idx
        self._r.drive_idx += 1
        actual = mm * self._r.slip_factor
        self._r.x += actual * math.cos(self._r.theta)
        self._r.y += actual * math.sin(self._r.theta)
        if self._r.wall_x_mm is not None and self._r.prox_found:
            self._r.prox_mm = max(15.0, self._r.wall_x_mm - self._r.x)
        if idx in self._r.cliff_on_step:
            self._r.status.is_cliff_detected = True
        if idx in self._r.origin_jump_on_step:
            self._r.origin_id = self._r.origin_jump_on_step[idx]
        if idx in self._r.lift_fov_on_step:
            self._r.lift_fov = True
        if idx in self._r.pose_none_on_step:
            self._r.pose_none = True

    def set_head_angle(self, angle: Any) -> None:
        """Record head angle command."""
        deg = float(getattr(angle, "degrees", 0.0))
        assert math.isfinite(deg) and abs(deg) <= 90.0
        self.calls.append(("head", deg))

    def set_lift_height(self, height: float) -> None:
        """Record normalized lift height command."""
        assert math.isfinite(height) and 0.0 <= height <= 1.0
        self.calls.append(("lift", float(height)))


class _FakeRobot:
    def __init__(self, *, x: float = 0.0, y: float = 0.0, theta_deg: float = 0.0, prox_mm: float = 900.0, prox_found: bool = True, lift_fov: bool = False, on_charger: bool = False) -> None:
        assert math.isfinite(x) and math.isfinite(prox_mm)
        self.x, self.y, self.theta = float(x), float(y), math.radians(theta_deg)
        self.origin_id = 0
        self.prox_mm, self.prox_found, self.lift_fov = float(prox_mm), bool(prox_found), bool(lift_fov)
        self.wall_x_mm: Any = None
        self.prox_on_turn: Any = None
        self.prox_none: bool = False
        self.slip_factor = 1.0
        self.drive_idx = 0
        self.cliff_on_step: set = set()
        self.origin_jump_on_step: dict = {}
        self.lift_fov_on_step: set = set()
        self.pose_none: bool = False
        self.pose_none_on_step: set = set()
        self.status = SimpleNamespace(is_cliff_detected=False, is_picked_up=False, is_falling=False, is_on_charger=on_charger, is_on_charger_platform=on_charger, is_in_calm_power_mode=False, is_charging=on_charger)
        self.behavior = _FakeBehavior(self)
        self.motors = SimpleNamespace(stop_all_motors=lambda: self.behavior.calls.append(("stop", 0.0)))
        self.world = SimpleNamespace(charger=None)

    @property
    def pose(self) -> Any:
        """Return SDK-shaped current pose, or None when unavailable."""
        if self.pose_none:
            return None
        assert math.isfinite(self.x) and math.isfinite(self.y)
        return _Pose(_Pos(self.x, self.y), _Rot(_Ang(self.theta)), self.origin_id)

    @property
    def proximity(self) -> Any:
        """Return SDK-shaped ToF reading or None when sensor is absent."""
        if self.prox_none:
            return None
        assert self.prox_mm >= 0.0 and isinstance(self.prox_found, bool)
        return SimpleNamespace(last_sensor_reading=SimpleNamespace(distance=SimpleNamespace(distance_mm=self.prox_mm), found_object=self.prox_found, is_lift_in_fov=self.lift_fov))

    def get_battery_state(self) -> Any:
        """Return healthy battery state."""
        assert isinstance(self.x, float) and isinstance(self.y, float)
        return SimpleNamespace(battery_volts=4.05, battery_level=3)


class MovementDirectivesTests(unittest.TestCase):
    """Unit and CLI tests for bounded movement primitives, 2D translate, and safety."""

    def test_forward_chunking_and_tof_obstacle_standoff(self) -> None:
        r = _FakeRobot(prox_mm=950.0, prox_found=True)
        res = bounded_drive(r, 450.0)
        self.assertTrue(res.ok)
        self.assertEqual((res.reason, res.chunks), (STOP_COMPLETED, 3))
        self.assertEqual([c for c in r.behavior.calls], [("drive", 180.0), ("drive", 180.0), ("drive", 90.0)])
        r2 = _FakeRobot(prox_mm=230.0, prox_found=True)
        r2.wall_x_mm = 230.0
        res2 = bounded_drive(r2, 500.0)
        self.assertEqual((res2.ok, res2.reason, res2.detail, res2.chunks), (False, STOP_BLOCKED, "proximity_standoff", 1))
        self.assertAlmostEqual(res2.moved_mm, 150.0, places=3)
        res3 = bounded_drive(r2, 100.0)
        self.assertEqual((res3.ok, res3.reason, len(r2.behavior.calls)), (False, STOP_BLOCKED, 1))

    def test_lift_in_fov_vs_raised_lift_obstacle(self) -> None:
        r_fov = _FakeRobot(prox_mm=500.0, prox_found=True, lift_fov=True)
        res_fov = bounded_drive(r_fov, 100.0)
        self.assertEqual((res_fov.ok, res_fov.reason, res_fov.detail, r_fov.behavior.calls), (False, STOP_REFUSED, "prox_occluded_lift_in_fov", []))
        res_opt = bounded_drive(r_fov, 100.0, allow_occluded_tof=True)
        self.assertTrue(res_opt.ok and abs(res_opt.moved_mm - 100.0) < 1e-3)
        r_obs = _FakeRobot(prox_mm=50.0, prox_found=True, lift_fov=False)
        res_obs = bounded_drive(r_obs, 100.0)
        self.assertEqual((res_obs.ok, res_obs.reason, res_obs.detail, r_obs.behavior.calls), (False, STOP_BLOCKED, "proximity_standoff", []))

    def test_audit_findings_standoff_unusable_tof_and_turn_safety(self) -> None:
        r_end = _FakeRobot(prox_mm=220.0, prox_found=True)
        res_end = bounded_drive(r_end, 180.0, standoff_mm=120.0)
        self.assertEqual((res_end.ok, res_end.reason, res_end.detail), (False, STOP_BLOCKED, "proximity_standoff"))
        self.assertAlmostEqual(res_end.moved_mm, 100.0, places=3)
        r_miss = _FakeRobot()
        r_miss.prox_none = True
        self.assertEqual((bounded_drive(r_miss, 100.0).reason, bounded_drive(r_miss, 100.0).detail, r_miss.behavior.calls), (STOP_REFUSED, "prox_unavailable", []))
        self.assertTrue(bounded_drive(r_miss, -50.0).ok)
        r_mid = _FakeRobot(prox_mm=900.0, prox_found=False)
        r_mid.lift_fov_on_step.add(0)
        res_mid = bounded_drive(r_mid, 300.0)
        self.assertEqual((res_mid.ok, res_mid.reason, res_mid.detail, res_mid.chunks), (False, STOP_REFUSED, "prox_occluded_lift_in_fov", 1))
        r_bud = _FakeRobot(prox_mm=950.0, prox_found=False)
        res_bud = bounded_drive(r_bud, 300.0, max_chunk_mm=10.0)
        self.assertEqual((res_bud.ok, res_bud.reason, res_bud.detail, res_bud.chunks), (False, STOP_REFUSED, "chunk_budget_exhausted", 16))
        r_hdg = _FakeRobot(prox_mm=900.0, prox_found=False)
        r_hdg.prox_on_turn = (50.0, True)
        res_hdg = execute_bounded_directive(r_hdg, "translate", {"dx_mm": 0.0, "dy_mm": 200.0})
        self.assertEqual((res_hdg["ok"], res_hdg["stopped_reason"], r_hdg.behavior.calls), (False, "proximity_standoff", [("turn", 90.0)]))
        r_chg = _FakeRobot(on_charger=True)
        self.assertEqual((bounded_turn(r_chg, 0.5).detail, r_chg.behavior.calls), ("on_charger", []))

    def test_reverse_cap_and_mid_reverse_cliff_halt(self) -> None:
        r_clamp = _FakeRobot()
        res_c = bounded_drive(r_clamp, -300.0, strict=False)
        self.assertTrue(res_c.ok and res_c.clamped)
        self.assertAlmostEqual(res_c.moved_mm, -150.0, places=3)
        self.assertEqual(r_clamp.behavior.calls, [("drive", -50.0), ("drive", -50.0), ("drive", -50.0)])
        r_strict = _FakeRobot()
        res_s = bounded_drive(r_strict, -200.0, strict=True)
        self.assertEqual((res_s.ok, res_s.reason, res_s.detail, r_strict.behavior.calls), (False, STOP_REFUSED, "bounds_exceeded", []))
        r_cliff = _FakeRobot()
        r_cliff.cliff_on_step.add(0)
        res_cl = bounded_drive(r_cliff, -120.0)
        self.assertEqual((res_cl.ok, res_cl.reason, res_cl.chunks), (False, STOP_CLIFF, 1))
        self.assertAlmostEqual(res_cl.moved_mm, -50.0, places=3)

    def test_2d_translate_decomposition_and_pure_reverse(self) -> None:
        r = _FakeRobot(prox_mm=950.0, prox_found=False)
        out = execute_bounded_directive(r, "translate", {"dx_mm": 200.0, "dy_mm": 200.0, "preserve_heading": True})
        self.assertTrue(out["ok"] and out["stopped_reason"] == "goal")
        self.assertAlmostEqual(out["actual_delta"]["x"], 0.2, places=3)
        self.assertAlmostEqual(out["actual_delta"]["y"], 0.2, places=3)
        self.assertAlmostEqual(out["actual_delta"]["theta"], 0.0, places=3)
        self.assertEqual([c[0] for c in r.behavior.calls], ["turn", "drive", "drive", "turn"])
        r_rev = _FakeRobot()
        out_rev = execute_bounded_directive(r_rev, "translate", {"dx_mm": -100.0, "dy_mm": 2.0})
        self.assertTrue(out_rev["ok"])
        self.assertEqual(r_rev.behavior.calls, [("drive", -50.0), ("drive", -50.0)])
        clamped = plan_directive("translate", {"dx_mm": 500.0, "dy_mm": 500.0}, strict=False)
        self.assertTrue(clamped["ok"] and clamped["clamped"])
        self.assertAlmostEqual(clamped["normalized"]["distance_mm"], 600.0, places=3)
        rejected = plan_directive("translate", {"dx_mm": 500.0, "dy_mm": 500.0}, strict=True)
        self.assertEqual((rejected["ok"], rejected["stopped_reason"]), (False, "bounds_exceeded"))

    def test_nan_inf_and_invalid_verbs_fail_closed(self) -> None:
        r = _FakeRobot()
        for bad_val in (float("nan"), float("inf"), float("-inf"), True, "nope"):
            res = bounded_drive(r, bad_val)  # type: ignore[arg-type]
            self.assertEqual((res.ok, res.reason, res.detail), (False, STOP_REFUSED, "invalid_numeric"))
            t_res = bounded_turn(r, bad_val)  # type: ignore[arg-type]
            self.assertEqual((t_res.ok, t_res.reason, t_res.detail), (False, STOP_REFUSED, "invalid_numeric"))
        unk = execute_bounded_directive(r, "spin_wildly", {})
        self.assertFalse(unk["ok"])
        self.assertEqual(r.behavior.calls, [])

    def test_rotation_slip_origin_jump_and_head_lift(self) -> None:
        r_turn = _FakeRobot()
        dead = bounded_turn(r_turn, math.radians(0.2))
        self.assertTrue(dead.ok and dead.detail == "deadband" and r_turn.behavior.calls == [])
        full = bounded_turn(r_turn, math.radians(450.0), strict=False)
        self.assertTrue(full.ok and full.clamped)
        r_slip = _FakeRobot(prox_mm=900.0, prox_found=False)
        r_slip.slip_factor = 0.1
        slipped = bounded_drive(r_slip, 100.0)
        self.assertEqual((slipped.ok, slipped.reason, slipped.detail), (False, STOP_BLOCKED, "wheel_slip_or_blocked"))
        r_oid = _FakeRobot(prox_mm=900.0, prox_found=False)
        r_oid.origin_jump_on_step[0] = 99
        oid_res = bounded_drive(r_oid, 120.0)
        self.assertEqual((oid_res.ok, oid_res.reason, oid_res.detail), (False, STOP_REFUSED, "origin_id_changed"))
        head_out = execute_bounded_directive(_FakeRobot(), "head", {"head_deg": 99.0})
        self.assertTrue(head_out["ok"] and head_out["clamped"])
        lift_out = execute_bounded_directive(_FakeRobot(), "lift", {"lift_norm": 2.0})
        self.assertTrue(lift_out["ok"] and lift_out["clamped"])
        move_out = execute_bounded_directive(_FakeRobot(prox_mm=950.0, prox_found=False), "move", {"dx_mm": 900.0})
        self.assertTrue(move_out["ok"] and move_out["clamped"])

    def test_batch_budget_and_early_stop(self) -> None:
        too_many = [{"verb": "move", "dx_mm": 50.0} for _ in range(17)]
        self.assertEqual(plan_directive_batch(too_many)["stopped_reason"], "batch_steps_exceeded")
        over_dist = [{"verb": "move", "dx_mm": 500.0}, {"verb": "move", "dx_mm": 500.0}, {"verb": "move", "dx_mm": 300.0}]
        self.assertEqual(plan_directive_batch(over_dist)["stopped_reason"], "batch_budget_exceeded")
        r = _FakeRobot(prox_mm=900.0, prox_found=False)
        r.cliff_on_step.add(1)
        seq = [{"verb": "move", "dx_mm": 100.0}, {"verb": "move", "dx_mm": 100.0}, {"verb": "turn", "dtheta_deg": 90.0}]
        out = execute_directive_batch(r, seq)
        self.assertEqual((out["ok"], out["stopped_reason"], out["steps_completed"]), (False, "cliff", 1))
        self.assertEqual([c[0] for c in r.behavior.calls], ["drive", "drive"])

    def test_strict_floor_and_boolean_flags(self) -> None:
        p1 = plan_directive("move", {"dx_mm": 9999.0, "strict": False}, strict=True)
        self.assertFalse(p1["ok"])
        self.assertEqual(p1["stopped_reason"], "bounds_exceeded")
        r = _FakeRobot(prox_mm=900.0)
        res = execute_bounded_directive(r, "move", {"dx_mm": 9999.0, "strict": False}, strict=True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stopped_reason"], "bounds_exceeded")
        self.assertEqual(r.behavior.calls, [])
        p2 = plan_directive("move", {"dx_mm": 100.0, "strict": "false"})
        self.assertEqual(p2["stopped_reason"], "invalid_boolean:strict")
        p3 = plan_directive("move", {"dx_mm": 100.0, "dry_run": 1})
        self.assertEqual(p3["stopped_reason"], "invalid_boolean:dry_run")
        p4 = plan_directive("translate", {"dx_mm": 100.0, "preserve_heading": "no"})
        self.assertEqual(p4["stopped_reason"], "invalid_boolean:preserve_heading")

    def test_missing_and_unknown_payload_keys(self) -> None:
        self.assertEqual(plan_directive("move", {})["stopped_reason"], "missing_distance")
        self.assertEqual(plan_directive("move", {"dist_mm": 100.0})["stopped_reason"], "unknown_keys:dist_mm")
        self.assertEqual(plan_directive("move", {"dx_mm": 100.0, "dx_m": 0.1})["stopped_reason"], "conflicting_distance_keys")
        cart = plan_directive("move", {"dx_m": 0.10, "dtheta_deg": 0.0, "speed_mps": 0.05})
        self.assertTrue(cart["ok"])
        self.assertAlmostEqual(cart["normalized"]["distance_mm"], 100.0)
        self.assertEqual(plan_directive("move", {"dx_m": 0.10, "dtheta_deg": 15.0})["stopped_reason"], "nonzero_dtheta_on_move")
        self.assertEqual(plan_directive("translate", {})["stopped_reason"], "missing_translation")
        self.assertEqual(plan_directive("head", {})["stopped_reason"], "missing_head_angle")
        self.assertEqual(plan_directive("lift", {})["stopped_reason"], "missing_lift_height")
        self.assertEqual(plan_directive("rotate", {"dtheta_deg": 10.0, "angle_deg": 10.0})["stopped_reason"], "conflicting_angle_keys")

    def test_standoff_mm_floor_and_override(self) -> None:
        p1 = plan_directive("move", {"dx_mm": 150.0, "standoff_mm": 0.0}, standoff_mm=80.0, strict=False)
        self.assertTrue(p1["ok"] and p1["clamped"])
        self.assertAlmostEqual(p1["normalized"]["standoff_mm"], 80.0)
        p2 = plan_directive("move", {"dx_mm": 150.0, "standoff_mm": 20.0}, standoff_mm=80.0, strict=True)
        self.assertEqual(p2["stopped_reason"], "bounds_exceeded")
        p3 = plan_directive("move", {"dx_mm": 150.0, "standoff_mm": 120.0}, standoff_mm=80.0, strict=True)
        self.assertTrue(p3["ok"])
        self.assertAlmostEqual(p3["normalized"]["standoff_mm"], 120.0)
        r = _FakeRobot(prox_mm=90.0)
        res = execute_bounded_directive(r, "move", {"dx_mm": 150.0, "standoff_mm": 0.0}, standoff_mm=80.0, strict=False)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stopped_reason"], "proximity_standoff")
        self.assertEqual(res["motion_result"]["reason"], "blocked")
        self.assertLessEqual(res["motion_result"]["moved_mm"], 15.0)
        p4 = plan_directive("translate", {"dx_mm": 150.0, "standoff_mm": 0.0}, standoff_mm=80.0, strict=True)
        self.assertEqual(p4["stopped_reason"], "bounds_exceeded")

    def test_pose_unavailable_and_turn_non_fabrication(self) -> None:
        r1 = _FakeRobot(prox_mm=900.0)
        r1.pose_none = True
        dr1 = bounded_drive(r1, 100.0)
        self.assertFalse(dr1.ok)
        self.assertEqual(dr1.detail, "pose_unavailable")
        self.assertEqual(r1.behavior.calls, [])
        tr1 = bounded_turn(r1, math.radians(45.0))
        self.assertFalse(tr1.ok)
        self.assertEqual(tr1.detail, "pose_unavailable")
        self.assertEqual(r1.behavior.calls, [])
        r2 = _FakeRobot(prox_mm=900.0)
        r2.pose_none_on_step.add(0)
        dr2 = bounded_drive(r2, 100.0)
        self.assertFalse(dr2.ok)
        self.assertEqual(dr2.detail, "pose_unavailable")
        self.assertAlmostEqual(dr2.moved_mm, 0.0)
        r3 = _FakeRobot(prox_mm=900.0)
        r3.behavior.turn_in_place = lambda *a, **k: r3.behavior.calls.append(("turn_noop", 0.0))
        tr3 = bounded_turn(r3, math.radians(45.0))
        self.assertTrue(tr3.ok)
        self.assertAlmostEqual(tr3.turned_rad, 0.0, places=9)
        r4 = _FakeRobot(prox_mm=900.0)
        r4.behavior.turn_in_place = lambda *a, **k: setattr(r4, "pose_none", True)
        tr4 = bounded_turn(r4, math.radians(45.0))
        self.assertFalse(tr4.ok)
        self.assertEqual(tr4.detail, "pose_unavailable")
        self.assertAlmostEqual(tr4.turned_rad, 0.0, places=9)

    def test_lazy_imports_and_cli_dry_run(self) -> None:
        repo_root = str(Path(__file__).resolve().parents[1])
        code = (
            f"import sys; sys.path.insert(0, {repo_root!r}); "
            "sys.modules['numpy']=None; sys.modules['PIL']=None; "
            "import vector_slam_maze, vector_slam_maze.movement_directives, vector_slam_maze.navmap_atlas, vector_slam_maze.mechanism_characterize; "
            "assert vector_slam_maze.__version__ == '0.5.1'; "
            "mr = vector_slam_maze.MotionResult(True, 'completed', 50.0, 0.0); "
            "assert mr.moved_mm == 50.0"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        buf1, buf2, buf3 = io.StringIO(), io.StringIO(), io.StringIO()
        with redirect_stdout(buf1):
            rc1 = cli_main(["translate", "--dx-mm", "250", "--dy-mm", "100", "--preserve-heading", "--dry-run"])
        self.assertEqual(rc1, 0)
        self.assertEqual(json.loads(buf1.getvalue())["verb"], "translate")
        with redirect_stdout(buf2):
            rc2 = cli_main(["translate", "--dx-mm", "700", "--strict", "--dry-run"])
        self.assertEqual(rc2, 6)
        with redirect_stdout(buf3):
            rc3 = cli_main(["act", "move", "--args-json", "{\"dx_m\":0.2}", "--dry-run"])
        self.assertEqual(rc3, 0)
        self.assertEqual(json.loads(buf3.getvalue())["primitives"][0]["distance_mm"], 180.0)


if __name__ == "__main__":
    unittest.main()
