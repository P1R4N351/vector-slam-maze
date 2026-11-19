"""Offline unit and Delta #27 ablation tests for vector_slam_maze.mechanism_characterize."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vector_slam_maze.cliff_guard import CliffHazard
from vector_slam_maze.cli import main as cli_main
from vector_slam_maze.mechanism_characterize import (
    HEAD_SETTLE_TOL_DEG,
    LIFT_SETTLE_TOL_MM,
    WHEEL_SPEED_TOL_MMPS,
    RobotStateRecorder,
    check_wheel_bench_safety,
    evaluate_physical_realism,
    fit_braking_response,
    fit_step_response,
    run_mechanism_characterization,
)


def _synth_first_order(t0: float, y0: float, y1: float, lat_s: float, tau_s: float, n: int = 32, dt_s: float = 0.025, field: str = "head_deg") -> list:
    """Build synthetic delayed first-order step trajectory samples."""
    import math

    rows = []
    for i in range(n):
        t = t0 + i * dt_s
        eff = max(0.0, (t - t0) - lat_s)
        val = y0 + (y1 - y0) * (1.0 - math.exp(-eff / tau_s)) if eff > 0.0 else y0
        rows.append({"t_s": t, field: val, "x_mm": 0.0, "y_mm": 0.0})
    return rows


def _realistic_report() -> dict:
    """Return a clean baseline report satisfying all non-ideal realism bounds."""
    step_ok = {
        "ok": True,
        "latency_ms": 36.0,
        "rise_10_90_ms": 140.0,
        "interior_samples": 6,
        "overshoot_pct": 2.4,
        "steady_state_error": 0.5,
    }
    return {
        "mechanisms": {
            "head": {"step_up": dict(step_ok), "hold_drift_deg": 0.1},
            "lift": {"step_up": dict(step_ok), "hold_drift_mm": 0.2},
            "left_wheel": {"step_fwd": dict(step_ok), "brake_high": {"ok": True, "braking_dist_mm": 7.8}},
            "right_wheel": {"step_fwd": dict(step_ok), "brake_high": {"ok": True, "braking_dist_mm": 7.6}},
        },
        "chassis": {"drive_err_mm": -0.4, "turn_err_deg": -0.9},
    }


def _fake_robot(on_charger: bool = False, prox_mm: float = 400.0, cliff: bool = False, lift_fov: bool = False) -> SimpleNamespace:
    """Build minimal fake Vector object for safety/characterization tests."""
    status_bits = (0x1000 if on_charger else 0) | (0x4000 if cliff else 0)
    status = SimpleNamespace(
        _status=status_bits, is_cliff_detected=cliff, is_picked_up=False, is_falling=False,
        is_on_charger=on_charger, is_on_charger_platform=on_charger, is_charging=on_charger, is_in_calm_power_mode=False,
    )
    prox = SimpleNamespace(last_sensor_reading=SimpleNamespace(distance=SimpleNamespace(distance_mm=prox_mm), found_object=prox_mm < 800.0, is_lift_in_fov=lift_fov))
    pose = SimpleNamespace(position=SimpleNamespace(x=200.0, y=200.0), rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=0.0)), origin_id=1)
    subs: list = []
    events = SimpleNamespace(subscribe=lambda cb, _ev: subs.append(cb), unsubscribe=lambda cb, _ev: subs.remove(cb) if cb in subs else None)
    motors = SimpleNamespace(set_wheel_motors=lambda *_a: None, set_head_motor=lambda *_a: None, set_lift_motor=lambda *_a: None, stop_all_motors=lambda: None)
    behavior = SimpleNamespace(set_head_angle=lambda *_a: None, set_lift_height=lambda *_a: None, drive_straight=lambda *_a: None, turn_in_place=lambda *_a: None)
    return SimpleNamespace(status=status, proximity=prox, pose=pose, events=events, motors=motors, behavior=behavior, get_battery_state=lambda: SimpleNamespace(battery_volts=4.1, battery_level=2), head_angle_rad=0.0, lift_height_mm=32.0, _subs=subs)


class TestMechanismCharacterize(unittest.TestCase):
    """Tests for curve fitting, safety gating, and Delta #27 physics realism checks."""

    def test_fit_step_response_non_ideal_metrics(self) -> None:
        rows = _synth_first_order(10.0, -20.0, 35.0, lat_s=0.035, tau_s=0.12, n=40, dt_s=0.025, field="head_deg")
        fit = fit_step_response(rows, "head_deg", 10.0, -20.0, 35.0, 0.5, HEAD_SETTLE_TOL_DEG)
        self.assertTrue(fit["ok"])
        self.assertGreaterEqual(fit["latency_ms"], 30.0)
        self.assertGreaterEqual(fit["rise_10_90_ms"], 200.0)
        self.assertGreaterEqual(fit["interior_samples"], 5)
        self.assertLessEqual(fit["steady_state_error"], 1.5)

    def test_fit_braking_response_pose_and_velocity(self) -> None:
        rows = [
            {"t_s": 0.99, "lw_mmps": 160.0, "x_mm": 0.0, "y_mm": 0.0},
            {"t_s": 1.02, "lw_mmps": 160.0, "x_mm": 2.4, "y_mm": 0.0},
            {"t_s": 1.05, "lw_mmps": 40.0, "x_mm": 4.0, "y_mm": 0.0},
            {"t_s": 1.08, "lw_mmps": 0.0, "x_mm": 4.4, "y_mm": 0.0},
        ]
        brk = fit_braking_response(rows, "lw_mmps", 1.00, 160.0, single_wheel=True)
        self.assertTrue(brk["ok"])
        self.assertGreater(brk["braking_dist_mm"], 5.0)
        self.assertAlmostEqual(brk["stop_latency_ms"], 50.0, delta=1.0)

    def test_wheel_bench_safety_fail_closed_gates(self) -> None:
        bot = _fake_robot(on_charger=False, prox_mm=400.0)
        ok, reason, _ = check_wheel_bench_safety(bot, confirm_safe_floor=False)
        self.assertFalse(ok)
        self.assertEqual(reason, "confirm_safe_floor_required")
        bot_chg = _fake_robot(on_charger=True, prox_mm=400.0)
        ok, reason, _ = check_wheel_bench_safety(bot_chg, confirm_safe_floor=True)
        self.assertFalse(ok)
        self.assertEqual(reason, "on_charger")
        bot_fov = _fake_robot(on_charger=False, prox_mm=400.0, lift_fov=True)
        ok, reason, _ = check_wheel_bench_safety(bot_fov, confirm_safe_floor=True, allow_occluded_tof=False)
        self.assertFalse(ok)
        self.assertEqual(reason, "prox_lift_in_fov")
        ok2, reason2, _ = check_wheel_bench_safety(bot_fov, confirm_safe_floor=True, allow_occluded_tof=True)
        self.assertTrue(ok2)
        self.assertEqual(reason2, "ok")

    def test_delta27_ablation_rejects_every_missing_effect(self) -> None:
        base = _realistic_report()
        self.assertTrue(evaluate_physical_realism(base, include_wheels=True)["realistic"])
        ablations = []
        r_lat = _realistic_report()
        r_lat["mechanisms"]["head"]["step_up"]["latency_ms"] = 1.0
        ablations.append(("no_latency", r_lat, "zero_or_low_latency"))
        r_snap = _realistic_report()
        r_snap["mechanisms"]["lift"]["step_up"]["interior_samples"] = 0
        ablations.append(("ideal_snap", r_snap, "ideal_snap_no_dynamics"))
        r_pd = _realistic_report()
        r_pd["mechanisms"]["head"]["step_up"]["steady_state_error"] = HEAD_SETTLE_TOL_DEG + 3.0
        ablations.append(("no_pd", r_pd, "pd_steady_error"))
        r_arm = _realistic_report()
        r_arm["mechanisms"]["left_wheel"]["brake_high"]["braking_dist_mm"] = 0.2
        ablations.append(("no_armature_brake", r_arm, "unrealistic_brake_dist"))
        r_fric = _realistic_report()
        r_fric["mechanisms"]["head"]["hold_drift_deg"] = 12.0
        ablations.append(("no_friction_sag", r_fric, "gravity_sag_no_stiction"))
        r_damp = _realistic_report()
        r_damp["mechanisms"]["lift"]["step_up"]["overshoot_pct"] = 40.0
        ablations.append(("no_damping_overshoot", r_damp, "excess_overshoot"))
        for name, rep, token in ablations:
            res = evaluate_physical_realism(rep, include_wheels=True)
            self.assertFalse(res["realistic"], f"{name} unexpectedly passed realism")
            self.assertIn(token, ",".join(res["failures"]), f"{name} missed {token}: {res['failures']}")

    def test_cliff_mid_wheel_aborts_with_cliff_hazard(self) -> None:
        bot = _fake_robot(on_charger=False, cliff=True)
        with self.assertRaises(CliffHazard):
            from vector_slam_maze.mechanism_characterize import _watch_sleep

            _watch_sleep(bot, 0.04, watch_hazards=True)

    def test_docked_characterize_runs_and_records_events(self) -> None:
        bot = _fake_robot(on_charger=True)
        with patch("vector_slam_maze.mechanism_characterize.time.sleep", lambda _s: None):
            rep = run_mechanism_characterization(bot, include_wheels=False)
        self.assertFalse(rep["ok"])  # empty fake event stream produces missing_step_up
        self.assertIn("head:missing_step_up", rep["failures"])
        rec = RobotStateRecorder()
        rec.attach(bot)
        self.assertEqual(len(bot._subs), 1)
        rec.detach(bot)
        self.assertEqual(len(bot._subs), 0)

    def test_cli_characterize_mechanisms_requires_via_vector(self) -> None:
        rc = cli_main(["characterize-mechanisms"])
        self.assertEqual(rc, 6)


if __name__ == "__main__":
    unittest.main()
