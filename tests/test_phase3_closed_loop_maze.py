"""Phase 3 tests: closed-loop FloodFill, marker kidnap recovery, flock, cert_onboard, rosbags."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple
import unittest
from unittest import mock

from vector_slam_maze.atlas_lock import AtlasLockTimeoutError, atlas_file_lock
from vector_slam_maze.cert_onboard import onboard_vector_cert
from vector_slam_maze.cli import _commit_session_atlas, _load_atlas_snapshot, main as cli_main

_HAS_ROSBAGS = importlib.util.find_spec("rosbags") is not None
from vector_slam_maze.landmarks import MazeMarkerAnchor
from vector_slam_maze.maze_closed_loop import (
    coalesce_collinear_runs,
    fuse_and_localize_step,
    heading_from_theta,
    run_floodfill_loop,
    turn_to_canonical_theta,
)
from vector_slam_maze.maze_solver import (
    DIR_DELTA,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    FloodFillPlanner,
    MazeGrid,
    execute_floodfill_run,
    execute_speed_run,
)
from vector_slam_maze.navmap_atlas import NavMapAtlas
from vector_slam_maze.types_and_math import Pose2D, Transform2D, normalize_angle


@dataclass
class _Pos:
    x: float
    y: float


@dataclass
class _Ang:
    radians: float

    @property
    def degrees(self) -> float:
        """Return angle in degrees."""
        assert math.isfinite(self.radians)
        return math.degrees(self.radians)


@dataclass
class _Rot:
    angle_z: _Ang


@dataclass
class _Pose:
    position: _Pos
    rotation: _Rot
    origin_id: int = 0


@dataclass
class _Dist:
    distance_mm: float


@dataclass
class _ProxReading:
    distance: _Dist
    found_object: bool
    is_lift_in_fov: bool = False


class _ProxComp:
    def __init__(self, robot: "_MockMazeRobot") -> None:
        self._r = robot

    @property
    def last_sensor_reading(self) -> _ProxReading:
        """Raycast true maze walls from mock physical world pose."""
        d_mm, found = self._r._raycast_tof_mm()
        assert d_mm >= 0.0
        return _ProxReading(_Dist(d_mm), found, False)


class _Status:
    is_cliff_detected = False
    is_picked_up = False
    is_falling = False
    is_on_charger = False
    is_on_charger_platform = False
    is_in_calm_power_mode = False
    is_charging = False


class _Batt:
    battery_volts = 4.05
    battery_level = 2


@dataclass
class _MarkerEnum:
    name: str


@dataclass
class _Archetype:
    marker_front: _MarkerEnum


@dataclass
class _MarkerObj:
    archetype: _Archetype
    pose: _Pose


class _World:
    def __init__(self, robot: "_MockMazeRobot") -> None:
        self._r = robot

    @property
    def visible_objects(self) -> list:
        """Expose visible CustomObjectMarkers in the active robot origin."""
        assert self._r.pose is not None
        return list(self._r.markers_by_origin.get(self._r.pose.origin_id, []))

    @property
    def charger(self) -> None:
        """Return None (off charger)."""
        return None


class _NavMapComp:
    latest_nav_map = None


class _Behavior:
    def __init__(self, robot: "_MockMazeRobot") -> None:
        self._r = robot

    def turn_in_place(self, angle: object) -> None:
        """Rotate both true and reported robot headings identically."""
        deg = float(getattr(angle, "degrees", 0.0))
        assert self._r is not None and math.isfinite(deg)
        d_rad = math.radians(deg)
        self._r.turn_history_deg.append(deg)
        self._r.theta = normalize_angle(self._r.theta + d_rad)
        self._r.true_theta = normalize_angle(self._r.true_theta + d_rad)

    def drive_straight(self, dist: object, _speed: object) -> None:
        """Advance physical pose, applying configured slip, heading bias, and optional kidnap."""
        mm = float(getattr(dist, "distance_mm", 0.0))
        assert self._r is not None and math.isfinite(mm)
        step = self._r.move_count
        self._r.move_count += 1
        actual = mm * (0.2 if step in self._r.slip_steps else 1.0)
        if self._r.heading_bias_rad != 0.0:
            self._r.theta = normalize_angle(self._r.theta + self._r.heading_bias_rad)
            self._r.true_theta = normalize_angle(self._r.true_theta + self._r.heading_bias_rad)
        self._r.x += actual * math.cos(self._r.theta)
        self._r.y += actual * math.sin(self._r.theta)
        self._r.true_x += actual * math.cos(self._r.true_theta)
        self._r.true_y += actual * math.sin(self._r.true_theta)
        self._r.pose_history.append((self._r.true_x, self._r.true_y))
        if step in self._r.kidnap_map:
            oid, rx, ry, rth = self._r.kidnap_map[step]
            self._r.origin_id, self._r.x, self._r.y, self._r.theta = oid, rx, ry, rth


class _MockMazeRobot:
    def __init__(self, wall_grid: MazeGrid, init_true: Pose2D, init_robot: Optional[Pose2D] = None) -> None:
        assert wall_grid is not None and init_true is not None
        self.wall_grid = wall_grid
        self.true_x, self.true_y, self.true_theta = init_true.x_mm, init_true.y_mm, init_true.theta_rad
        rp = init_robot or init_true
        self.x, self.y, self.theta, self.origin_id = rp.x_mm, rp.y_mm, rp.theta_rad, rp.origin_id
        self.move_count = 0
        self.slip_steps: set[int] = set()
        self.kidnap_map: Dict[int, Tuple[int, float, float, float]] = {}
        self.markers_by_origin: Dict[int, List[_MarkerObj]] = {}
        self.turn_history_deg: List[float] = []
        self.heading_bias_rad: float = 0.0
        self.pose_history: List[Tuple[float, float]] = []
        self.dynamic_obstacle_x_mm: Optional[float] = None
        self.dynamic_obstacle_row: Optional[int] = None
        self.status, self.world, self.nav_map = _Status(), _World(self), _NavMapComp()
        self.proximity, self.behavior = _ProxComp(self), _Behavior(self)

    @property
    def pose(self) -> _Pose:
        """Return current reported robot SDK pose."""
        return _Pose(_Pos(self.x, self.y), _Rot(_Ang(self.theta)), self.origin_id)

    def get_battery_state(self) -> _Batt:
        """Return healthy battery reading."""
        return _Batt()

    def _raycast_tof_mm(self) -> Tuple[float, bool]:
        """Compute realistic ToF range (mm) along cardinal direction from true pose."""
        d = heading_from_theta(self.true_theta)
        r0, c0 = self.wall_grid.world_to_cell(self.true_x, self.true_y)
        cs = self.wall_grid.cell_size_mm
        assert cs > 0.0 and d in DIR_DELTA
        dr, dc = DIR_DELTA[d]
        wall_dist, rr, cc = 9999.0, r0, c0
        for hop in range(16):
            if self.wall_grid.has_wall(rr, cc, d):
                wall_dist = (hop + 0.5) * cs
                break
            rr, cc = rr + dr, cc + dc
        if self.dynamic_obstacle_x_mm is not None and d == EAST and (self.dynamic_obstacle_row is None or r0 == self.dynamic_obstacle_row):
            d_obs = self.dynamic_obstacle_x_mm - self.true_x
            if 0.0 < d_obs < wall_dist:
                return (d_obs, d_obs <= 850.0)
        return (wall_dist, wall_dist <= 900.0)


def _make_pose(x: float, y: float, th: float, oid: int = 0) -> _Pose:
    """Construct lightweight mock SDK Pose."""
    assert math.isfinite(x) and math.isfinite(y)
    return _Pose(_Pos(x, y), _Rot(_Ang(th)), oid)


def _gen_self_signed_tls_files(tmp_dir: Path, cn: str = "Vector-1DF0") -> Tuple[Path, Path]:
    """Generate throwaway self-signed cert/key PEM pair with Vector CN using openssl CLI."""
    cert_p, key_p = tmp_dir / f"srv_{cn}.crt", tmp_dir / f"srv_{cn}.key"
    cmd = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key_p), "-out", str(cert_p), "-days", "1", "-subj", f"/C=US/O=Anki/CN={cn}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0 and cert_p.is_file()
    return (cert_p, key_p)


def _start_once_tls_server(cert_p: Path, key_p: Path) -> Tuple[socket.socket, threading.Thread]:
    """Start local ephemeral TLS server responding once on loopback."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert_p), str(key_p))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)

    def _serve() -> None:
        conn, _ = srv.accept()
        try:
            with ctx.wrap_socket(conn, server_side=True) as tc:
                tc.recv(16)
        except OSError:
            pass
        finally:
            conn.close()
            srv.close()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return (srv, t)


class Phase3ClosedLoopMazeTests(unittest.TestCase):
    """Tests for Phase 3A closed-loop navigation, locking, onboarding, and rosbags."""

    def test_heading_rotation_with_nonzero_tf_can(self) -> None:
        """Non-zero submap rotation tf_can.dtheta_rad must rotate robot commands accurately."""
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.ensure_submap("s:origin_5", 5)
        tf = Transform2D(0.0, 0.0, math.pi * 0.5)
        atlas.transforms_to_canonical["s:origin_5"] = tf
        robot = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0), Pose2D(0.0, 0.0, -math.pi * 0.5, 5))
        turn_to_canonical_theta(robot, 0.0, tf)
        assert abs(normalize_angle(robot.theta - (-math.pi * 0.5))) < 1e-6
        res = execute_floodfill_run(robot, atlas, "s:origin_5", goal_cell=(2, 3), max_steps=6, cell_size_mm=180.0)
        self.assertEqual(res["status"], "reached")
        self.assertEqual(res["curr"], [2, 3])

    def test_wheel_slip_recovery_recomputes_true_cell(self) -> None:
        """Wheel slip on move step must NOT falsely advance curr cell."""
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("slip:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0))
        robot.slip_steps.add(0)
        res = execute_floodfill_run(robot, atlas, "slip:origin_0", goal_cell=(2, 3), max_steps=6, cell_size_mm=180.0)
        moved = [e for e in res["trace"] if e["event"] == "moved"]
        self.assertEqual(res["status"], "reached")
        self.assertTrue(moved[0]["slipped"])
        self.assertGreaterEqual(len(moved), 2)

    def test_kidnap_origin_jump_recovered_via_marker_map(self) -> None:
        """Mid-run origin_id jump must relocalize via CustomObjectMarker and reach goal."""
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("kid:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0))
        robot.kidnap_map[0] = (77, 500.0, -200.0, math.pi * 0.5)
        robot.markers_by_origin[77] = [_MarkerObj(_Archetype(_MarkerEnum("Circles2")), _make_pose(500.0, -20.0, math.pi * 0.5, 77))]
        markers = {"Circles2": MazeMarkerAnchor("Circles2", 2, 4, 360.0, 0.0, 0.0)}
        res = execute_floodfill_run(robot, atlas, "kid:origin_0", goal_cell=(2, 4), max_steps=8, cell_size_mm=180.0, marker_map=markers)
        events = [e["event"] for e in res["trace"]]
        self.assertEqual(res["status"], "reached")
        self.assertIn("relocalized_marker", events)

    def test_blind_origin_jump_halts_cleanly(self) -> None:
        """Origin jump without visible bridge anchor or marker must halt safely."""
        grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("blind:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(0.0, 0.0, 0.0, 0))
        robot.kidnap_map[0] = (88, 300.0, 100.0, 1.0)
        res = execute_floodfill_run(robot, atlas, "blind:origin_0", goal_cell=(2, 4), max_steps=6, cell_size_mm=180.0)
        self.assertEqual(res["status"], "halted")
        self.assertEqual(res["reason"], "unlocalized_origin_change")

    def test_tof_peek_and_two_pass_collinear_speed_run(self) -> None:
        """Pass 1 with 3-way ToF peek discovers walls; Pass 2 runs coalesced speed segments."""
        wall_grid = MazeGrid(5, 5, 180.0, -450.0, -450.0)
        wall_grid.set_wall(2, 2, SOUTH, True)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("twopass:origin_0")
        robot = _MockMazeRobot(wall_grid, Pose2D(0.0, 0.0, 0.0, 0))
        res = execute_floodfill_run(robot, atlas, "twopass:origin_0", goal_cell=(2, 4), max_steps=10, cell_size_mm=180.0, peek_unknown=True, two_pass=True)
        segs = coalesce_collinear_runs([(2, 2), (2, 3), (2, 4), (3, 4)])
        self.assertEqual(res["status"], "reached")
        self.assertTrue(res["two_pass"])
        self.assertEqual(res["pass2_speed_run"]["status"], "reached")
        self.assertEqual([(s["dir"], s["cells"]) for s in segs], [(EAST, 2), (NORTH, 1)])

    def test_atlas_file_lock_multiprocess_serialization(self) -> None:
        """Two OS processes overlapping RMW on atlas.json must preserve both submaps via flock."""
        repo_root = str(Path(__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": repo_root}
        worker = "\n".join([
            "import sys, time",
            "from pathlib import Path",
            "from vector_slam_maze.atlas_lock import atlas_file_lock",
            "from vector_slam_maze.navmap_atlas import NavMapAtlas",
            "ap, tag, oid, delay = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])",
            "time.sleep(delay)",
            "with atlas_file_lock(ap, timeout_s=3.0):",
            "    a = NavMapAtlas.load(ap)",
            "    time.sleep(0.25)",
            "    a.ensure_submap(tag, oid)",
            "    a.save(ap)",
        ])
        with tempfile.TemporaryDirectory() as td:
            ap = Path(td) / "atlas.json"
            NavMapAtlas().save(ap)
            p1 = subprocess.Popen([sys.executable, "-c", worker, str(ap), "p1:origin_1", "1", "0.0"], env=env)
            p2 = subprocess.Popen([sys.executable, "-c", worker, str(ap), "p2:origin_2", "2", "0.05"], env=env)
            self.assertEqual(p1.wait(timeout=5), 0)
            self.assertEqual(p2.wait(timeout=5), 0)
            final = NavMapAtlas.load(ap)
            self.assertIn("p1:origin_1", final.submaps)
            self.assertIn("p2:origin_2", final.submaps)

    def test_atlas_file_lock_multiprocess_timeout(self) -> None:
        """Bounded LOCK_NB retry must raise AtlasLockTimeoutError when another PID holds lock."""
        repo_root = str(Path(__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": repo_root}
        holder = "\n".join([
            "import sys, time",
            "from pathlib import Path",
            "from vector_slam_maze.atlas_lock import atlas_file_lock",
            "ap, ready = Path(sys.argv[1]), Path(sys.argv[2])",
            "with atlas_file_lock(ap, timeout_s=2.0):",
            "    ready.write_text('held')",
            "    time.sleep(0.55)",
        ])
        with tempfile.TemporaryDirectory() as td:
            ap, ready = Path(td) / "atlas.json", Path(td) / "ready"
            NavMapAtlas().save(ap)
            p = subprocess.Popen([sys.executable, "-c", holder, str(ap), str(ready)], env=env)
            t0 = time.monotonic()
            while not ready.is_file() and (time.monotonic() - t0) < 2.0:
                time.sleep(0.01)
            self.assertTrue(ready.is_file())
            t_wait = time.monotonic()
            with self.assertRaises(AtlasLockTimeoutError):
                with atlas_file_lock(ap, timeout_s=0.12):
                    pass
            self.assertLess(time.monotonic() - t_wait, 0.45)
            self.assertEqual(p.wait(timeout=5), 0)

    def test_lock_free_session_commit_merges_concurrent_updates(self) -> None:
        """Concurrent snapshot/working sessions merge cleanly without losing canonical guard or cells."""
        with tempfile.TemporaryDirectory() as td:
            ap, exp = Path(td) / "atlas.json", Path(td) / "exp"
            b1, w1 = _load_atlas_snapshot(ap)
            b2, w2 = _load_atlas_snapshot(ap)
            w1.ensure_submap("demo:origin_1", 1)
            w1.submaps["demo:origin_1"].cells[(1, 1)] = w1.submaps["demo:origin_1"].cells.get((1, 1), __import__("vector_slam_maze.navmap_atlas", fromlist=["CellStats"]).CellStats(log_odds=1.25, obs_count=1))
            w2.ensure_submap("v1df0:origin_2", 2)
            _commit_session_atlas(ap, exp, b1, w1, set(), {"tag": "demo"})
            _commit_session_atlas(ap, exp, b2, w2, set(), {"tag": "v1df0"})
            final = NavMapAtlas.load(ap)
            self.assertEqual(final.canonical_submap_id, "demo:origin_1")
            self.assertIn("v1df0:origin_2", final.submaps)
            self.assertNotIn("v1df0:origin_2", final.transforms_to_canonical)
            self.assertEqual(len(final.sessions), 2)

    def test_cert_onboard_unreachable_preserves_ini(self) -> None:
        """Unreachable Vector :443 without session cert returns rc=3 without mutating sdk_config.ini."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            bot_info, sdk_dir = d / "botSdkInfo.json", d / "sdk"
            bot_info.write_text(json.dumps({"global_guid": "glob==", "robots": [{"esn": "0dd1beef", "ip_address": "127.0.0.1", "guid": "g1==", "activated": True}]}))
            ini_path = sdk_dir / "sdk_config.ini"
            orig_ini = "[0dd1cafe]\ncert = /tmp/c.cert\nip = 192.0.2.130\nname = Vector-DEMO\nguid = x==\n"
            sdk_dir.mkdir(); ini_path.write_text(orig_ini, encoding="utf-8")
            with mock.patch("vector_slam_maze.cert_onboard.socket.create_connection", side_effect=ConnectionRefusedError("closed")):
                rc, payload = onboard_vector_cert("0dd1beef", bot_info, d / "sess", sdk_dir, timeout_s=0.4)
            self.assertEqual(rc, 3)
            self.assertEqual(payload["error"], "tls_unreachable")
            self.assertFalse(payload["sdk_config_mutated"])
            self.assertEqual(ini_path.read_text(encoding="utf-8"), orig_ini)

    def test_cert_onboard_prefers_cache_extracts_cn_and_enforces_0600(self) -> None:
        """Cached wire-pod session cert supplies CN=Vector-1DF0 and writes sdk_config.ini mode 0600."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            cert_p, _key_p = _gen_self_signed_tls_files(d, cn="Vector-1DF0")
            sess_dir, sdk_dir = d / "sess", d / "sdk"
            sess_dir.mkdir(); sdk_dir.mkdir()
            (sess_dir / "0dd1beef").write_text(cert_p.read_text(), encoding="utf-8")
            ini_p = sdk_dir / "sdk_config.ini"
            ini_p.write_text("[0dd1cafe]\ncert = /tmp/c\nip = 192.0.2.1\nname = Vector-DEMO\nguid = c==\n")
            os.chmod(ini_p, 0o644)
            info = d / "botSdkInfo.json"; info.write_text(json.dumps({"0dd1beef": {"esn": "0dd1beef", "ip_address": "192.0.2.113", "guid": "new==", "activated": True}}))
            with mock.patch("vector_slam_maze.cert_onboard.socket.create_connection", side_effect=ConnectionRefusedError("offline")):
                rc, out = onboard_vector_cert("0dd1beef", info, sess_dir, sdk_dir)
            cfg = configparser.ConfigParser(interpolation=None); cfg.read(ini_p)
            self.assertEqual(rc, 0)
            self.assertEqual(out["name"], "Vector-1DF0")
            self.assertEqual(out["source"], "session_certs_cache")
            self.assertEqual(ini_p.stat().st_mode & 0o777, 0o600)
            self.assertIn("0dd1cafe", cfg.sections()); self.assertIn("0dd1beef", cfg.sections())

    def test_cert_onboard_refuses_cache_vs_live_mismatch_and_gates_live_fallback(self) -> None:
        """Refuse when live :443 cert mismatches wire-pod cache; require opt-in for uncached live fallback."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            cert_a, _ = _gen_self_signed_tls_files(d / "a", cn="Vector-1DF0") if (d / "a").mkdir() or True else (d, d)
            cert_b, key_b = _gen_self_signed_tls_files(d / "b", cn="Vector-1DF0") if (d / "b").mkdir() or True else (d, d)
            sess_dir, sdk_dir = d / "sess", d / "sdk"
            sess_dir.mkdir(); sdk_dir.mkdir()
            (sess_dir / "0dd1beef").write_text(cert_a.read_text(), encoding="utf-8")
            info = d / "botSdkInfo.json"; info.write_text(json.dumps({"0dd1beef": {"esn": "0dd1beef", "ip_address": "192.0.2.113", "guid": "new==", "activated": True}}))
            srv, t = _start_once_tls_server(cert_b, key_b)
            real_conn = socket.create_connection
            with mock.patch("vector_slam_maze.cert_onboard.socket.create_connection", side_effect=lambda _a, timeout=2.0: real_conn(("127.0.0.1", srv.getsockname()[1]), timeout=timeout)):
                rc_mis, out_mis = onboard_vector_cert("0dd1beef", info, sess_dir, sdk_dir)
            t.join(timeout=2.0)
            self.assertEqual(rc_mis, 3); self.assertEqual(out_mis["error"], "cert_mismatch_vs_wire_pod_cache")
            (sess_dir / "0dd1beef").unlink()
            srv2, t2 = _start_once_tls_server(cert_b, key_b)
            with mock.patch("vector_slam_maze.cert_onboard.socket.create_connection", side_effect=lambda _a, timeout=2.0: real_conn(("127.0.0.1", srv2.getsockname()[1]), timeout=timeout)):
                rc_unv, out_unv = onboard_vector_cert("0dd1beef", info, sess_dir, sdk_dir)
            t2.join(timeout=2.0)
            self.assertEqual(rc_unv, 3); self.assertEqual(out_unv["error"], "live_cert_unverified")
            srv3, t3 = _start_once_tls_server(cert_b, key_b)
            with mock.patch("vector_slam_maze.cert_onboard.socket.create_connection", side_effect=lambda _a, timeout=2.0: real_conn(("127.0.0.1", srv3.getsockname()[1]), timeout=timeout)):
                rc_ok, out_ok = onboard_vector_cert("0dd1beef", info, sess_dir, sdk_dir, allow_live_fallback=True)
            t3.join(timeout=2.0)
            self.assertEqual(rc_ok, 0); self.assertEqual(out_ok["source"], "tls_443_fallback")

    @unittest.skipUnless(_HAS_ROSBAGS, "optional rosbags extra not installed")
    def test_rosbag_export_sqlite3_and_mcap_roundtrip(self) -> None:
        """export-rosbag writes readable SQLite3 & MCAP ROS 2 bags with ToF-only /vector/scan."""
        from rosbags.rosbag2 import Reader  # type: ignore
        from rosbags.typesys import Stores, get_typestore  # type: ignore

        ts = get_typestore(Stores.ROS2_HUMBLE)
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "odom.csv").write_text("timestamp_s,x_m,y_m,theta_rad,left_wheel_mps,right_wheel_mps,pitch_rad,prox_m,prox_found\n1700000000.0,0.0,0.0,0.0,0.05,0.05,0.0,0.25,1\n1700000001.0,0.1,0.0,0.1,0.0,0.0,0.0,9.99,0\n")
            (d / "imu.csv").write_text("1700000000000000000,0.0,0.0,0.05,0.0,0.0,9.81\n1700000001000000000,0.0,0.0,0.0,0.0,0.0,9.81\n")
            for fmt in ("sqlite3", "mcap"):
                bag = d / f"bag_{fmt}"
                rc = cli_main(["export-rosbag", "--dataset", str(d), "--out-bag", str(bag), "--storage", fmt])
                self.assertEqual(rc, 0)
                scans = []
                with Reader(bag) as reader:
                    topics = {c.topic for c in reader.connections}
                    self.assertEqual(topics, {"/vector/odom", "/vector/imu", "/vector/scan"})
                    for conn, _t, raw in reader.messages():
                        if conn.topic == "/vector/scan":
                            scans.append(ts.deserialize_cdr(raw, conn.msgtype))
                self.assertEqual(len(scans), 2)
                self.assertEqual(len(scans[0].ranges), 1)
                self.assertAlmostEqual(float(scans[0].ranges[0]), 0.25, places=4)
                self.assertTrue(math.isinf(float(scans[1].ranges[0])))

    def test_fix1_speed_run_halts_before_beyond_range_dynamic_obstacle(self) -> None:
        """Speed run must chunk <= MAX_TOF_MM, re-check ToF, and halt >=120 mm short."""
        grid = MazeGrid(3, 10, 180.0, -90.0, -90.0)
        for c in range(9):
            grid.set_wall(1, c, EAST, False)
        for c in range(10):
            grid.set_wall(1, c, NORTH, True)
            grid.set_wall(1, c, SOUTH, True)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("spd:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(0.0, 180.0, 0.0, 0))
        robot.dynamic_obstacle_x_mm, robot.dynamic_obstacle_row = 1190.0, 1
        res = execute_speed_run(robot, atlas, "spd:origin_0", goal_cell=(1, 9), cell_size_mm=180.0, grid=grid)
        self.assertEqual(res["status"], "halted")
        self.assertEqual(res["reason"], "unexpected_obstacle")
        self.assertLessEqual(robot.true_x, 1190.0 - 120.0)
        self.assertGreaterEqual(robot.move_count, 2)

    def test_fix2_persistent_heading_bias_stays_within_20mm_of_cell_centers(self) -> None:
        """Cell-center steering + 0.5 deg deadband hold <=20 mm error over 12 biased cells."""
        grid = MazeGrid(3, 14, 180.0, -90.0, -90.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("bias:origin_0")
        robot = _MockMazeRobot(grid, Pose2D(0.0, 180.0, math.radians(3.0), 0))
        robot.heading_bias_rad = math.radians(3.0)
        res = run_floodfill_loop(robot, atlas, grid, FloodFillPlanner(grid), "bias:origin_0", (1, 12), 16, 180.0, None, False)
        self.assertEqual(res["status"], "reached")
        self.assertEqual(len(robot.pose_history), 12)
        worst = max(math.hypot(px - (k + 1) * 180.0, py - 180.0) for k, (px, py) in enumerate(robot.pose_history))
        self.assertLessEqual(worst, 20.0)

    def test_fix3_noisy_marker_sightings_keep_integrated_cells_within_20mm(self) -> None:
        """Noisy markers + gross outliers must not shift mapped cells beyond 20 mm."""
        atlas = NavMapAtlas(cell_mm=20.0)
        markers = {"Circles2": MazeMarkerAnchor("Circles2", 2, 4, 360.0, 0.0, 0.0)}
        robot = _MockMazeRobot(MazeGrid(5, 5, 180.0, -450.0, -450.0), Pose2D(0.0, 0.0, 0.0, 7))
        robot.markers_by_origin[7] = [_MarkerObj(_Archetype(_MarkerEnum("Circles2")), _make_pose(360.0, 0.0, 0.0, 7))]
        fuse_and_localize_step(robot, atlas, "m:origin_0", marker_map=markers)
        sub = atlas.submaps["m:origin_7"]
        for px, py, th in ((0.0, 0.0, 0.0), (0.0, 0.0, math.pi * 0.5), (-200.0, 100.0, 0.0), (200.0, -150.0, -math.pi * 0.5)):
            sub.integrate_tof_ray(Pose2D(px, py, th, 7), 300.0, True, False)
        self.assertGreaterEqual(len(sub.cells), 35)
        base_pts = {k: atlas.transforms_to_canonical["m:origin_7"].apply_point((k[0] + 0.5) * 20.0, (k[1] + 0.5) * 20.0) for k in sub.cells}
        sightings = ((35, -20, 5.0), (-32, 18, -4.8), (28, 24, 4.2), (-30, -22, -4.5), (180, -90, 22.0), (36, -15, 5.2), (-38, 16, -5.5), (25, 20, 3.8), (-26, -19, -4.0), (-165, 110, -19.0), (30, -12, 4.5), (-28, 14, -4.3), (22, 15, 3.2), (-20, -14, -3.5), (18, -10, 2.5), (-16, 12, -2.5))
        worst_disp_mm, rejected = 0.0, 0
        for dx, dy, ddeg in sightings:
            robot.markers_by_origin[7] = [_MarkerObj(_Archetype(_MarkerEnum("Circles2")), _make_pose(360.0 + dx, float(dy), math.radians(ddeg), 7))]
            loc = fuse_and_localize_step(robot, atlas, "m:origin_0", marker_map=markers)
            rejected += int(bool(loc.get("marker_rejected")))
            tf_cur = atlas.transforms_to_canonical["m:origin_7"]
            for k, (bx, by) in base_pts.items():
                nx, ny = tf_cur.apply_point((k[0] + 0.5) * 20.0, (k[1] + 0.5) * 20.0)
                worst_disp_mm = max(worst_disp_mm, math.hypot(nx - bx, ny - by))
        self.assertEqual(rejected, 2)
        self.assertLessEqual(worst_disp_mm, 20.0)

    def test_fix4_post_peek_heading_avoids_redundant_tie_break_turn(self) -> None:
        """Planner heading must reflect post-peek pose so NORTH exit takes 0 extra turns."""
        wall_grid = MazeGrid(4, 4, 180.0, -90.0, -90.0)
        atlas = NavMapAtlas()
        atlas.assume_same_frame_as_canonical("pk:origin_0")
        robot = _MockMazeRobot(wall_grid, Pose2D(0.0, 0.0, 0.0, 0))
        res = run_floodfill_loop(robot, atlas, wall_grid, FloodFillPlanner(wall_grid), "pk:origin_0", (1, 1), 1, 180.0, None, True)
        self.assertEqual(len(robot.turn_history_deg), 1)
        self.assertLess(abs(robot.turn_history_deg[0] - 90.0), 1e-3)
        self.assertEqual(res["trace"][-1]["dir"], NORTH)

    def test_fix5_marker_filter_tracks_continuous_drift_without_lockout(self) -> None:
        """Gain floor must keep tracking continuous intra-origin drift over 400 sightings."""
        atlas, markers = NavMapAtlas(cell_mm=20.0), {"Circles2": MazeMarkerAnchor("Circles2", 2, 4, 360.0, 0.0, 0.0)}
        robot = _MockMazeRobot(MazeGrid(5, 5, 180.0, -450.0, -450.0), Pose2D(0.0, 0.0, 0.0, 7))
        robot.markers_by_origin[7] = [_MarkerObj(_Archetype(_MarkerEnum("Circles2")), _make_pose(360.0, 0.0, 0.0, 7))]
        fuse_and_localize_step(robot, atlas, "m:origin_0", marker_map=markers)
        rng, rejected, max_lag = random.Random(42), 0, 0.0
        for i in range(1, 401):
            true_x = 2.0 * i
            mx, my = 360.0 - (true_x + rng.gauss(0.0, 3.0)), -rng.gauss(0.0, 3.0)
            robot.markers_by_origin[7] = [_MarkerObj(_Archetype(_MarkerEnum("Circles2")), _make_pose(mx, my, 0.0, 7))]
            loc = fuse_and_localize_step(robot, atlas, "m:origin_0", marker_map=markers)
            rejected += int(bool(loc.get("marker_rejected")))
            max_lag = max(max_lag, abs(loc["cpose"].x_mm - true_x))
        self.assertEqual(rejected, 0)
        self.assertLessEqual(max_lag, 22.0)
        self.assertLessEqual(abs(loc["cpose"].x_mm - 800.0), 20.0)

    def test_fix6_step_displacement_relocks_after_consensus_while_rejecting_wild_glitches(self) -> None:
        """Consistent displaced sightings must relock at quorum 3 while wild glitches stay rejected."""
        atlas, markers = NavMapAtlas(cell_mm=20.0), {"Circles2": MazeMarkerAnchor("Circles2", 2, 4, 360.0, 0.0, 0.0)}
        robot = _MockMazeRobot(MazeGrid(5, 5, 180.0, -450.0, -450.0), Pose2D(0.0, 0.0, 0.0, 7))
        robot.markers_by_origin[7] = [_MarkerObj(_Archetype(_MarkerEnum("Circles2")), _make_pose(360.0, 0.0, 0.0, 7))]
        fuse_and_localize_step(robot, atlas, "m:origin_0", marker_map=markers)
        atlas.marker_obs_counts["m:origin_7"] = 10
        for gx, gy, gdeg in ((180.0, -90.0, 22.0), (-165.0, 110.0, -19.0), (195.0, 140.0, -24.0)):
            robot.markers_by_origin[7] = [_MarkerObj(_Archetype(_MarkerEnum("Circles2")), _make_pose(360.0 + gx, gy, math.radians(gdeg), 7))]
            loc = fuse_and_localize_step(robot, atlas, "m:origin_0", marker_map=markers)
            self.assertTrue(loc["marker_rejected"]); self.assertFalse(loc["marker_relocked"])
            self.assertLessEqual(math.hypot(loc["cpose"].x_mm, loc["cpose"].y_mm), 1.0)
        tf_bump, rng = Transform2D(260.0, -80.0, math.radians(22.0)), random.Random(7)
        m_inv = tf_bump.invert().apply_pose(Pose2D(360.0, 0.0, 0.0, 0), target_origin_id=7)
        for k in range(1, 6):
            robot.x, robot.y, robot.theta = 90.0 * k, 20.0 * k, math.radians(5.0 * k)
            mx, my, mth = m_inv.x_mm + rng.uniform(-8.0, 8.0), m_inv.y_mm + rng.uniform(-8.0, 8.0), m_inv.theta_rad + math.radians(rng.uniform(-1.2, 1.2))
            robot.markers_by_origin[7] = [_MarkerObj(_Archetype(_MarkerEnum("Circles2")), _make_pose(mx, my, mth, 7))]
            loc, c_true = fuse_and_localize_step(robot, atlas, "m:origin_0", marker_map=markers), tf_bump.apply_pose(Pose2D(robot.x, robot.y, robot.theta, 7), target_origin_id=0)
            err = math.hypot(loc["cpose"].x_mm - c_true.x_mm, loc["cpose"].y_mm - c_true.y_mm)
            if k <= 2:
                self.assertTrue(loc["marker_rejected"]); self.assertFalse(loc["marker_relocked"])
            elif k == 3:
                self.assertFalse(loc["marker_rejected"]); self.assertTrue(loc["marker_relocked"]); self.assertLessEqual(err, 15.0)
            else:
                self.assertFalse(loc["marker_rejected"]); self.assertLessEqual(err, 10.0)


if __name__ == "__main__":
    unittest.main()
