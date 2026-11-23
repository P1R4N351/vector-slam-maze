"""Passive multi-session basal MemoryMap daemon, SQLite exporter, and origin rehydrator."""

from __future__ import annotations

import concurrent.futures
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib

from .atlas_lock import atlas_file_lock
from .navmap_atlas import (
    CellStats,
    NavMapAtlas,
    SubmapLayer,
    canonical_charger_pose,
    extract_quadtree_leaves,
    observe_world_anchors,
    rehydrate_onboard_obstacles,
)
from .sdk_session import VectorSDKSession, _sdk_exception_types
from .types_and_math import Pose2D, pose2d_from_sdk_pose

MAX_DAEMON_TICKS = 172_800
MAX_LEAF_CELL_SPAN = 200
HTTP_POD_TIMEOUT_S = 5.0
SEATED_CELL_MM = 10.0


@dataclass(frozen=True)
class BasalConfig:
    """Runtime configuration for BasalMapDaemon (zero hardcoded household endpoints)."""

    serial: str
    atlas_path: Path
    export_dir: Optional[Path] = None
    sqlite_path: Optional[Path] = None
    navmap_json_path: Optional[Path] = None
    pod_url: Optional[str] = None
    ip: Optional[str] = None
    session_id: Optional[str] = None
    nav_freq_hz: float = 1.0
    poll_sleep_s: float = 0.5
    state_stale_s: float = 45.0
    export_interval_s: float = 30.0
    max_ticks: int = MAX_DAEMON_TICKS
    ensure_pod_timeouts: bool = False
    rehydrate_on_new_origin: bool = False
    max_rehydrate_boxes: int = 24


def init_basal_db(db_path: Path) -> sqlite3.Connection:
    """Open WAL SQLite store matching p2-shelf revisions + seated schemas."""
    assert isinstance(db_path, Path) and db_path.name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path), timeout=10.0)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS revisions (session TEXT, origin INTEGER, utc REAL, leaves BLOB)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS seated "
        "(session TEXT, origin INTEGER, utc REAL, x REAL, y REAL, yaw REAL, PRIMARY KEY(session,origin))"
    )
    return db


def encode_leaves_payload(leaves: List[Any]) -> Tuple[bytes, bytes]:
    """Serialize non-unknown quadtree leaves and return (raw_json_bytes, sha256_digest)."""
    assert isinstance(leaves, list)
    rows = [[p.cx_mm, p.cy_mm, p.size_mm, p.content] for p in leaves if int(p.content) != 0]
    raw = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return raw, hashlib.sha256(raw).digest()


def _rasterize_seated_leaf(
    leaf_row: Tuple[float, float, float, int],
    ox: float,
    oy: float,
    c: float,
    s: float,
    terrain: Dict[Tuple[int, int], int],
    conflicts: Set[Tuple[int, int]],
    semantic: List[Any],
) -> None:
    """Transform one compressed leaf into docked seated_robot frame and update terrain or semantic list."""
    x, y, width, kind = float(leaf_row[0]), float(leaf_row[1]), float(leaf_row[2]), int(leaf_row[3])
    dx, dy = x - ox, y - oy
    cx, cy = c * dx + s * dy, -s * dx + c * dy
    if kind not in (2, 7):
        semantic.append([round(cx, 2), round(cy, 2), leaf_row[2], leaf_row[3], round(-math.atan2(s, c), 6)])
        return
    radius = width * (abs(c) + abs(s)) * 0.5
    i0 = math.floor((cx - radius) / SEATED_CELL_MM)
    i1 = min(i0 + MAX_LEAF_CELL_SPAN, math.ceil((cx + radius) / SEATED_CELL_MM))
    j0 = math.floor((cy - radius) / SEATED_CELL_MM)
    j1 = min(j0 + MAX_LEAF_CELL_SPAN, math.ceil((cy + radius) / SEATED_CELL_MM))
    for i in range(i0, i1):
        for j in range(j0, j1):
            a, b = (i + 0.5) * SEATED_CELL_MM - cx, (j + 0.5) * SEATED_CELL_MM - cy
            if abs(c * a - s * b) > width * 0.5 or abs(s * a + c * b) > width * 0.5:
                continue
            if (i, j) in terrain and terrain[(i, j)] != kind:
                conflicts.add((i, j))
            terrain[(i, j)] = kind


def export_seated_navmap_dict(db_path: Path) -> Dict[str, Any]:
    """Read SQLite revisions joined with seated poses and return 10 mm simulator map dict."""
    assert isinstance(db_path, Path) and db_path.is_file(), f"missing sqlite db: {db_path}"
    terrain: Dict[Tuple[int, int], int] = {}
    conflicts: Set[Tuple[int, int]] = set()
    latest: Optional[float] = None
    revisions = 0
    semantic: List[Any] = []
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        rows = db.execute(
            "SELECT r.utc,r.leaves,s.x,s.y,s.yaw FROM revisions r JOIN seated s USING(session,origin) ORDER BY r.utc,r.rowid"
        )
        for latest, compressed, ox, oy, yaw in rows:
            c, s = math.cos(float(yaw)), math.sin(float(yaw))
            semantic = []
            for leaf_row in json.loads(zlib.decompress(compressed)):
                _rasterize_seated_leaf(tuple(leaf_row), float(ox), float(oy), c, s, terrain, conflicts, semantic)
            revisions += 1
        total = int(db.execute("SELECT count(*) FROM revisions").fetchone()[0])
    assert revisions <= total
    return {
        "cell_mm": int(SEATED_CELL_MM),
        "frame": "seated_robot",
        "latest_utc": latest,
        "revisions": revisions,
        "unregistered_revisions": total - revisions,
        "latest_objects_and_visibility": semantic,
        "cells": [[i, j, k, int((i, j) in conflicts)] for (i, j), k in sorted(terrain.items())],
    }


def export_seated_navmap(db_path: Path, dest_path: Path) -> Dict[str, Any]:
    """Atomically write simulator-compatible seated NavMap JSON from native SQLite."""
    assert isinstance(dest_path, Path) and dest_path.name
    data = export_seated_navmap_dict(db_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(f"{dest_path.suffix}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, dest_path)
    return data


def _http_pod_json(url: str, method: str = "GET", timeout_s: float = HTTP_POD_TIMEOUT_S) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Execute bounded stdlib HTTP request against wire-pod and decode JSON if valid."""
    assert url.startswith(("http://", "https://")) and method in ("GET", "POST")
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read(1_048_576)
            return int(resp.status), json.loads(raw.decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raw = exc.read(65_536) if hasattr(exc, "read") else b""
        try:
            return int(exc.code), json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            return int(exc.code), None
    except (OSError, ValueError):
        return 0, None


def resolve_robot_ip(pod_url: Optional[str], serial: str, explicit_ip: Optional[str] = None) -> Optional[str]:
    """Resolve current robot IP via explicit override or wire-pod /api-sdk/get_sdk_info."""
    assert serial, "serial required"
    if explicit_ip:
        return explicit_ip.strip() or None
    if not pod_url:
        return None
    status, data = _http_pod_json(f"{pod_url.rstrip('/')}/api-sdk/get_sdk_info", method="GET")
    if status != 200 or not isinstance(data, dict):
        return None
    for row in list(data.get("robots", []))[:64]:
        if isinstance(row, dict) and str(row.get("esn", "")).lower() == serial.lower():
            ip = str(row.get("ip_address", "")).strip()
            return ip or None
    return None


def check_pod_cryosleep(pod_url: Optional[str], serial: str) -> bool:
    """Return True if wire-pod reports Cryosleep active or pod-gated split state."""
    if not pod_url:
        return False
    q = urllib.parse.urlencode({"serial": serial})
    status, data = _http_pod_json(f"{pod_url.rstrip('/')}/api-sdk/cryo_mute?{q}", method="GET")
    if status == 404 or status == 0 or not isinstance(data, dict):
        return False
    if status == 502:
        return bool(data.get("pod_gated"))
    return bool(data.get("muted") or data.get("pod_gated") or data.get("robot_muted"))


def ensure_pod_basal_timeouts(pod_url: Optional[str], serial: str) -> Optional[Dict[str, Any]]:
    """Best-effort POST /api-sdk/basal_map?enable=true on wire-pod at daemon start."""
    if not pod_url:
        return None
    q = urllib.parse.urlencode({"serial": serial, "enable": "true"})
    status, data = _http_pod_json(f"{pod_url.rstrip('/')}/api-sdk/basal_map?{q}", method="POST")
    if status in (200, 202) and isinstance(data, dict):
        return data
    return None


def _robot_is_seated(pose: Any, status: Any, origin_id: int) -> bool:
    """Return True when robot is stationary on charger in the active map origin."""
    if pose is None or status is None:
        return False
    return (
        int(getattr(pose, "origin_id", -1)) == int(origin_id)
        and bool(getattr(status, "is_on_charger", False))
        and not bool(getattr(status, "are_wheels_moving", False))
        and not bool(getattr(status, "is_picked_up", False))
    )


def _has_prior_canonical_obstacles(atlas: NavMapAtlas, current_sid: str) -> bool:
    """Return True if any registered canonical submap other than current_sid holds occupied/cliff cells."""
    assert isinstance(atlas, NavMapAtlas)
    for sid, submap in atlas.submaps.items():
        if sid == current_sid or sid not in atlas.transforms_to_canonical:
            continue
        for st in submap.cells.values():
            if st.log_odds >= 0.4 or st.cliff_hits > 0:
                return True
    return False


def _snapshot_submap_baseline(dst: NavMapAtlas, src: NavMapAtlas, sid: str) -> None:
    """Clone one submap's current cells, anchors, and transform from src into dst as baseline."""
    assert isinstance(dst, NavMapAtlas) and isinstance(src, NavMapAtlas) and sid in src.submaps
    src_sub = src.submaps[sid]
    dst_sub = SubmapLayer(cell_mm=src_sub.cell_mm, origin_id=src_sub.origin_id, robot_id=src_sub.robot_id)
    for k, st in src_sub.cells.items():
        dst_sub.cells[k] = CellStats(st.log_odds, st.cliff_hits, st.obs_count, st.last_content, st.floor_obs)
    dst.submaps[sid] = dst_sub
    dst.anchors[sid] = dict(src.anchors.get(sid, {}))
    if dst.canonical_submap_id is None and src.canonical_submap_id is not None:
        dst.canonical_submap_id = src.canonical_submap_id
    if sid in src.transforms_to_canonical:
        dst.transforms_to_canonical[sid] = src.transforms_to_canonical[sid]


class BasalMapDaemon:
    """Passive NavMap + seated SQLite + NavMapAtlas recorder and origin rehydrator."""

    def __init__(
        self,
        cfg: BasalConfig,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
        wall_fn: Callable[[], float] = time.time,
        session_factory: Callable[..., Any] = VectorSDKSession,
    ) -> None:
        assert isinstance(cfg, BasalConfig) and cfg.serial and cfg.max_ticks > 0
        self.cfg = cfg
        self._sleep_fn = sleep_fn
        self._clock_fn = clock_fn
        self._wall_fn = wall_fn
        self._session_factory = session_factory
        self._session_id = cfg.session_id or uuid.uuid4().hex
        self._state_lock = threading.Lock()
        self._last_state_mono = clock_fn()
        self._last_map: Any = None
        self._last_digest: Optional[Tuple[int, bytes]] = None
        self._seated_origins: Set[int] = set()
        self._rehydrated_origins: Set[int] = set()
        self._working_atlas = NavMapAtlas()
        self._baseline_atlas = NavMapAtlas()
        self._last_export_mono = 0.0
        self._last_commit_mono = 0.0
        self._db: Optional[sqlite3.Connection] = None

    def on_robot_state(self, _robot: Any = None, _event: Any = None, _msg: Any = None) -> None:
        """Record monotonic receipt timestamp whenever RobotState arrives."""
        now = self._clock_fn()
        with self._state_lock:
            self._last_state_mono = now

    def _subscribe_state(self, robot: Any) -> Any:
        """Attach RobotState heartbeat callback and return unsubscribe token/callable."""
        events = getattr(robot, "events", None)
        if events is None or not hasattr(events, "subscribe"):
            return None
        ev_type = getattr(events, "robot_state_event", None)
        if ev_type is None:
            try:
                from anki_vector.events import Events  # type: ignore

                ev_type = Events.robot_state
            except ImportError:
                return None
        events.subscribe(self.on_robot_state, ev_type)
        return (events, ev_type)

    def _unsubscribe_state(self, token: Any) -> None:
        """Detach RobotState callback cleanly during teardown."""
        if token is None:
            return
        events, ev_type = token
        catch_types = (OSError, RuntimeError, AttributeError, TypeError) + _sdk_exception_types()
        try:
            if hasattr(events, "unsubscribe"):
                events.unsubscribe(self.on_robot_state, ev_type)
        except catch_types:
            pass

    def _flush_navmap_export(self, force: bool = False) -> None:
        """Write atomic seated NavMap JSON when configured and due."""
        if self.cfg.sqlite_path is None or self.cfg.navmap_json_path is None:
            return
        now = self._clock_fn()
        if not force and (now - self._last_export_mono) < self.cfg.export_interval_s:
            return
        if self.cfg.sqlite_path.is_file():
            export_seated_navmap(self.cfg.sqlite_path, self.cfg.navmap_json_path)
            self._last_export_mono = now

    def _commit_revision_to_atlas(
        self,
        origin: int,
        leaves: List[Any],
        pose: Any,
        is_seated: bool,
        anchors: Dict[str, Pose2D],
        digest_changed: bool,
    ) -> Tuple[NavMapAtlas, str]:
        """Fold current revision + anchors into shared NavMapAtlas under short lock."""
        sid = f"{self._session_id}:origin_{origin}"
        submap = self._working_atlas.ensure_submap(sid, origin_id=origin, robot_id=self.cfg.serial)
        if digest_changed:
            submap.integrate_leaves(leaves)
        assume_can = False
        if is_seated and pose is not None:
            self._working_atlas.record_anchor(sid, "charger_seated", pose2d_from_sdk_pose(pose), assume_canonical=True)
            assume_can = True
        for aid, apose in anchors.items():
            self._working_atlas.record_anchor(sid, aid, apose, assume_canonical=(assume_can and aid == "charger"))
        explicit = {sid} if assume_can else set()
        with atlas_file_lock(self.cfg.atlas_path):
            fresh = NavMapAtlas.load(self.cfg.atlas_path)
            fresh.merge_session_from(self._working_atlas, baseline=self._baseline_atlas, explicit_tf_sids=explicit)
            fresh.save(self.cfg.atlas_path)
            if self.cfg.export_dir is not None:
                fresh.export_ros_pgm_yaml(self.cfg.export_dir / "grid.pgm", self.cfg.export_dir / "grid.yaml")
        _snapshot_submap_baseline(self._baseline_atlas, self._working_atlas, sid)
        self._last_commit_mono = self._clock_fn()
        return fresh, sid

    def _maybe_rehydrate_origin(self, robot: Any, fresh: NavMapAtlas, sid: str, origin: int, pose: Any, status: Any) -> int:
        """Seed fresh MemoryMap origin once anchored if prior canonical obstacles exist and robot is awake."""
        if not self.cfg.rehydrate_on_new_origin or origin in self._rehydrated_origins:
            return 0
        if sid not in fresh.transforms_to_canonical or not _has_prior_canonical_obstacles(fresh, sid):
            return 0
        if bool(getattr(status, "is_in_calm_power_mode", False)) or check_pod_cryosleep(self.cfg.pod_url, self.cfg.serial):
            return 0
        self._rehydrated_origins.add(origin)
        bot_pose = pose2d_from_sdk_pose(pose) if pose is not None else None
        chg_pose = canonical_charger_pose(fresh)
        catch_types = (OSError, RuntimeError, TimeoutError, concurrent.futures.TimeoutError) + _sdk_exception_types()
        try:
            count = rehydrate_onboard_obstacles(
                robot,
                fresh,
                sid,
                max_boxes=self.cfg.max_rehydrate_boxes,
                robot_pose_bot=bot_pose,
                charger_pose_can=chg_pose,
            )
            print(f"basal_daemon: rehydrated {count} fixed obstacles into origin {origin}", flush=True)
            return count
        except catch_types as exc:
            print(f"basal_daemon: skipped origin {origin} rehydration ({exc})", file=sys.stderr, flush=True)
            return 0

    def _handle_grid(self, robot: Any, grid: Any) -> None:
        """Persist new NavMap revisions and seated origin anchors into SQLite and atlas."""
        origin = int(getattr(grid, "origin_id", 0))
        stamp = self._wall_fn()
        pose = getattr(robot, "pose", None)
        status = getattr(robot, "status", None)
        is_seated = _robot_is_seated(pose, status, origin)
        leaves = extract_quadtree_leaves(grid)
        payload, raw_digest = encode_leaves_payload(leaves)
        digest = (origin, raw_digest)
        digest_changed = digest != self._last_digest
        new_seated = is_seated and origin not in self._seated_origins
        if not digest_changed and not new_seated:
            self._last_map = grid
            return
        if self._db is not None:
            with self._db:
                if digest_changed:
                    self._db.execute("INSERT INTO revisions VALUES(?,?,?,?)", (self._session_id, origin, stamp, zlib.compress(payload)))
                    print(f"origin {origin}, {len(leaves)} native leaves", flush=True)
                if is_seated:
                    self._db.execute(
                        "INSERT OR IGNORE INTO seated VALUES(?,?,?,?,?,?)",
                        (self._session_id, origin, stamp, float(pose.position.x), float(pose.position.y), float(pose.rotation.angle_z.radians)),
                    )
                    self._seated_origins.add(origin)
        anchors = observe_world_anchors(robot)
        fresh, sid = self._commit_revision_to_atlas(origin, leaves, pose, is_seated, anchors, digest_changed)
        self._maybe_rehydrate_origin(robot, fresh, sid, origin, pose, status)
        if digest_changed:
            self._last_digest = digest
        self._last_map = grid
        self._flush_navmap_export(force=new_seated)

    def _poll_loop(self, robot: Any) -> int:
        """Poll cached _latest_nav_map and check RobotState heartbeat up to max_ticks."""
        assert robot is not None and self.cfg.max_ticks > 0
        nav = getattr(robot, "nav_map", None)
        for _ in range(self.cfg.max_ticks):
            grid = getattr(nav, "_latest_nav_map", None)
            if grid is not None:
                origin = int(getattr(grid, "origin_id", 0))
                need_seat = (origin not in self._seated_origins) and _robot_is_seated(
                    getattr(robot, "pose", None), getattr(robot, "status", None), origin
                )
                if grid is not self._last_map or need_seat:
                    self._handle_grid(robot, grid)
            self._flush_navmap_export(force=False)
            with self._state_lock:
                age = self._clock_fn() - self._last_state_mono
            if age > self.cfg.state_stale_s:
                print(f"basal_daemon: RobotState stream stale for {age:.1f}s; exiting for restart", file=sys.stderr, flush=True)
                return 5
            self._sleep_fn(self.cfg.poll_sleep_s)
        return 0

    def run(self) -> int:
        """Run bounded passive persistence loop until max_ticks rollover or RobotState stall."""
        assert self.cfg.state_stale_s > 0.0 and self.cfg.poll_sleep_s >= 0.0
        if self.cfg.ensure_pod_timeouts:
            ensure_pod_basal_timeouts(self.cfg.pod_url, self.cfg.serial)
        resolved_ip = resolve_robot_ip(self.cfg.pod_url, self.cfg.serial, self.cfg.ip)
        if self.cfg.sqlite_path is not None:
            self._db = init_basal_db(self.cfg.sqlite_path)
        self._last_state_mono = self._clock_fn()
        try:
            with self._session_factory(
                serial=self.cfg.serial,
                active=False,
                enable_nav_map=True,
                nav_freq_hz=self.cfg.nav_freq_hz,
                ip=resolved_ip,
            ) as robot:
                token = self._subscribe_state(robot)
                try:
                    return self._poll_loop(robot)
                finally:
                    self._unsubscribe_state(token)
        finally:
            self._flush_navmap_export(force=True)
            if self._db is not None:
                self._db.close()
                self._db = None
        return 0
