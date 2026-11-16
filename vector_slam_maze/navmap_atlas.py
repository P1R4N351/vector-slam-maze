"""Persistent multi-session NavMapGrid atlas, SE(2) anchor stitching, and ROS map export."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .atlas_lock import atlas_file_lock
from .landmarks import anchor_name_from_object
from .types_and_math import (
    NavContent,
    Pose2D,
    Transform2D,
    identity_transform,
    pose2d_from_sdk_pose,
)

MAX_QUADTREE_NODES = 16384
MAX_LEAF_SPAN = 64
MAX_RAY_STEPS = 256
MAX_BFS_HOPS = 1024
MAX_GRID_DIM = 2048
MAX_REHYDRATE_BOXES = 64
MAX_REHYDRATE_CELLS = 65536
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0
OCC_THRESHOLD = 0.4
FREE_THRESHOLD = -0.3
MAX_TOF_MM = 1000.0
TOF_HIT_CUTOFF_MM = 850.0


@dataclass(frozen=True)
class LeafPatch:
    """Flattened quad-tree leaf in its local origin_id coordinate frame."""

    cx_mm: float
    cy_mm: float
    size_mm: float
    content: int


@dataclass
class CellStats:
    """Occupancy log-odds and cliff counter for one quantized cell."""

    log_odds: float = 0.0
    cliff_hits: int = 0
    obs_count: int = 0
    last_content: int = int(NavContent.UNKNOWN)

    def apply_content(self, content: int) -> None:
        """Fold a NavNodeContentTypes reading into this cell."""
        c = int(content)
        self.last_content = c
        if c in (int(NavContent.CLEAR_OF_OBSTACLE), int(NavContent.CLEAR_OF_CLIFF)):
            self.log_odds = max(LOG_ODDS_MIN, self.log_odds - 0.85)
            self.obs_count += 1
        elif c in (
            int(NavContent.OBSTACLE_CUBE),
            int(NavContent.OBSTACLE_PROXIMITY),
            int(NavContent.OBSTACLE_PROXIMITY_EXPLORED),
            int(NavContent.OBSTACLE_UNRECOGNIZED),
        ):
            self.log_odds = min(LOG_ODDS_MAX, self.log_odds + 1.25)
            self.obs_count += 1
        elif c == int(NavContent.CLIFF):
            self.log_odds = min(LOG_ODDS_MAX, self.log_odds + 2.0)
            self.cliff_hits += 1
            self.obs_count += 1
        elif c in (int(NavContent.INTERESTING_EDGE), int(NavContent.NON_INTERESTING_EDGE)):
            if self.obs_count == 0:
                self.log_odds = min(LOG_ODDS_MAX, self.log_odds + 0.15)
                self.obs_count = 1

    def to_list(self) -> List[Any]:
        """Encode compactly for JSON storage."""
        return [round(self.log_odds, 4), self.cliff_hits, self.obs_count, self.last_content]

    @classmethod
    def from_list(cls, raw: List[Any]) -> "CellStats":
        """Restore from compact JSON list."""
        assert isinstance(raw, list) and len(raw) >= 4
        return cls(float(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))


def extract_quadtree_leaves(nav_map_obj: Any) -> List[LeafPatch]:
    """Iteratively extract leaf patches from anki_vector NavMapGrid (zero recursion)."""
    if nav_map_obj is None:
        return []
    root = getattr(nav_map_obj, "root_node", None)
    if root is None:
        return []
    leaves: List[LeafPatch] = []
    stack: List[Any] = [root]
    visited = 0
    while stack and visited < MAX_QUADTREE_NODES:
        visited += 1
        node = stack.pop()
        children = getattr(node, "children", None)
        if not children:
            leaves.append(_node_to_leaf(node))
            continue
        for child in children:
            if child is not None:
                stack.append(child)
    assert visited < MAX_QUADTREE_NODES, "quadtree traversal exceeded node cap"
    return leaves


def _node_to_leaf(node: Any) -> LeafPatch:
    """Convert SDK NavMapGridNode leaf to LeafPatch."""
    center = getattr(node, "center", None)
    cx = float(getattr(center, "x", 0.0)) if center is not None else 0.0
    cy = float(getattr(center, "y", 0.0)) if center is not None else 0.0
    size = float(getattr(node, "size", 8.0))
    raw_content = getattr(node, "content", None)
    val = getattr(raw_content, "value", raw_content)
    content = 0 if val is None else int(val)
    return LeafPatch(cx, cy, size, content)


class SubmapLayer:
    """Sparse planar occupancy grid inside a single session/origin coordinate frame."""

    def __init__(self, cell_mm: float = 20.0, origin_id: int = 0) -> None:
        assert cell_mm > 0.0, "cell_mm must be positive"
        self.cell_mm = float(cell_mm)
        self.origin_id = int(origin_id)
        self.cells: Dict[Tuple[int, int], CellStats] = {}

    def world_to_cell(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Quantize millimeter coordinates into integer grid indices."""
        return (int(math.floor(x_mm / self.cell_mm)), int(math.floor(y_mm / self.cell_mm)))

    def integrate_leaves(self, leaves: List[LeafPatch]) -> int:
        """Rasterize NavMapGrid leaves into this sparse submap."""
        updates = 0
        for leaf in leaves:
            if leaf.content == int(NavContent.UNKNOWN):
                continue
            updates += self._rasterize_leaf(leaf)
        return updates

    def _rasterize_leaf(self, leaf: LeafPatch) -> int:
        """Rasterize a single square quadtree leaf into intersecting cells."""
        half = max(leaf.size_mm * 0.5, 1e-3)
        ix0, iy0 = self.world_to_cell(leaf.cx_mm - half + 1e-4, leaf.cy_mm - half + 1e-4)
        ix1, iy1 = self.world_to_cell(leaf.cx_mm + half - 1e-4, leaf.cy_mm + half - 1e-4)
        ix1 = min(ix1, ix0 + MAX_LEAF_SPAN - 1)
        iy1 = min(iy1, iy0 + MAX_LEAF_SPAN - 1)
        hits = 0
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                stats = self.cells.setdefault((ix, iy), CellStats())
                stats.apply_content(leaf.content)
                hits += 1
        return hits

    def integrate_tof_ray(
        self, pose: Pose2D, prox_mm: float, found_object: bool, lift_in_fov: bool
    ) -> int:
        """Carve free cells along front ToF laser and mark obstacle endpoint if hit."""
        if lift_in_fov or prox_mm <= 0.0:
            return 0
        dist_mm = min(float(prox_mm), MAX_TOF_MM)
        hit = found_object and dist_mm < TOF_HIT_CUTOFF_MM
        steps = min(MAX_RAY_STEPS, max(1, int(dist_mm / (self.cell_mm * 0.5))))
        free_end = steps - 1 if hit else steps
        dx_step = math.cos(pose.theta_rad) * (dist_mm / steps)
        dy_step = math.sin(pose.theta_rad) * (dist_mm / steps)
        seen: set[Tuple[int, int]] = set()
        for step in range(free_end + 1):
            cell = self.world_to_cell(pose.x_mm + step * dx_step, pose.y_mm + step * dy_step)
            if cell not in seen:
                seen.add(cell)
                st = self.cells.setdefault(cell, CellStats())
                st.log_odds = max(LOG_ODDS_MIN, st.log_odds - 0.45)
                st.obs_count += 1
        if hit:
            end_cell = self.world_to_cell(pose.x_mm + steps * dx_step, pose.y_mm + steps * dy_step)
            st = self.cells.setdefault(end_cell, CellStats())
            st.log_odds = min(LOG_ODDS_MAX, st.log_odds + 1.15)
            st.last_content = int(NavContent.OBSTACLE_PROXIMITY)
            st.obs_count += 1
        return len(seen) + (1 if hit else 0)

    def to_dict(self) -> dict:
        """Serialize submap cells cleanly to JSON dict."""
        encoded = {f"{ix},{iy}": st.to_list() for (ix, iy), st in self.cells.items()}
        return {"cell_mm": self.cell_mm, "origin_id": self.origin_id, "cells": encoded}

    @classmethod
    def from_dict(cls, raw: dict) -> "SubmapLayer":
        """Load submap from serialized JSON dict."""
        layer = cls(cell_mm=float(raw["cell_mm"]), origin_id=int(raw.get("origin_id", 0)))
        for k, vals in raw.get("cells", {}).items():
            parts = k.split(",", 1)
            layer.cells[(int(parts[0]), int(parts[1]))] = CellStats.from_list(vals)
        return layer


def observe_world_anchors(robot: Any) -> Dict[str, Pose2D]:
    """Read Charger, LightCube, and CustomObjects matching robot.pose.origin_id."""
    assert robot is not None, "robot required"
    rpose = pose2d_from_sdk_pose(robot.pose)
    world = getattr(robot, "world", None)
    if world is None:
        return {}
    out: Dict[str, Pose2D] = {}
    candidates: List[Any] = [getattr(world, "charger", None), getattr(world, "connected_light_cube", None)]
    for obj in list(getattr(world, "visible_objects", []) or []):
        candidates.append(obj)
    for obj in candidates:
        name = anchor_name_from_object(obj)
        sdk_pose = getattr(obj, "pose", None) if obj is not None else None
        if not name or sdk_pose is None:
            continue
        ap = pose2d_from_sdk_pose(sdk_pose, fallback_origin_id=rpose.origin_id)
        if ap.origin_id == rpose.origin_id:
            out[name] = ap
    return out


class NavMapAtlas:
    """Multi-session pose-graph atlas fusing Vector submaps via fiducial anchors."""

    def __init__(self, cell_mm: float = 20.0) -> None:
        assert cell_mm > 0.0, "cell_mm must be positive"
        self.cell_mm = float(cell_mm)
        self.canonical_submap_id: Optional[str] = None
        self.submaps: Dict[str, SubmapLayer] = {}
        self.anchors: Dict[str, Dict[str, Pose2D]] = {}
        self.transforms_to_canonical: Dict[str, Transform2D] = {}
        self.marker_obs_counts: Dict[str, int] = {}
        self.marker_outlier_streaks: Dict[str, Tuple[Transform2D, int]] = {}
        self.sessions: List[Dict[str, Any]] = []

    def ensure_submap(self, submap_id: str, origin_id: int = 0) -> SubmapLayer:
        """Create submap if absent; first submap becomes canonical frame."""
        assert submap_id, "submap_id required"
        if submap_id not in self.submaps:
            self.submaps[submap_id] = SubmapLayer(cell_mm=self.cell_mm, origin_id=origin_id)
            self.anchors.setdefault(submap_id, {})
        if self.canonical_submap_id is None:
            self.canonical_submap_id = submap_id
            self.transforms_to_canonical[submap_id] = identity_transform()
        return self.submaps[submap_id]

    def record_anchor(
        self, submap_id: str, anchor_id: str, pose: Pose2D, assume_canonical: bool = False
    ) -> None:
        """Store anchor pose in submap frame and solve/re-solve registrations."""
        self.ensure_submap(submap_id, pose.origin_id)
        self.anchors[submap_id][anchor_id] = pose
        if assume_canonical and submap_id not in self.transforms_to_canonical:
            self.transforms_to_canonical[submap_id] = identity_transform()
        self.resolve_registrations()

    def assume_same_frame_as_canonical(self, submap_id: str) -> None:
        """Explicitly anchor a stationary docked session into canonical coordinates."""
        assert isinstance(submap_id, str) and bool(submap_id)
        self.ensure_submap(submap_id)
        self.transforms_to_canonical[submap_id] = identity_transform()
        self.marker_obs_counts[submap_id] = max(self.marker_obs_counts.get(submap_id, 0), 6)
        assert submap_id in self.transforms_to_canonical
        self.resolve_registrations()

    def resolve_registrations(self) -> int:
        """Run bounded BFS over shared-anchor graph to register submaps to canonical."""
        if self.canonical_submap_id is None:
            return 0
        self.transforms_to_canonical.setdefault(self.canonical_submap_id, identity_transform())
        queue: deque[str] = deque(self.transforms_to_canonical.keys())
        hops = 0
        while queue and hops < MAX_BFS_HOPS:
            hops += 1
            known_id = queue.popleft()
            t_known = self.transforms_to_canonical[known_id]
            for cand_id in list(self.submaps.keys()):
                if cand_id in self.transforms_to_canonical:
                    continue
                t_rel = _estimate_relative_transform(self.anchors.get(cand_id, {}), self.anchors.get(known_id, {}))
                if t_rel is not None:
                    self.transforms_to_canonical[cand_id] = t_known.compose(t_rel)
                    queue.append(cand_id)
        return len(self.transforms_to_canonical)

    def fuse_canonical_cells(self) -> Dict[Tuple[int, int], CellStats]:
        """Project all registered submaps into a single canonical sparse grid."""
        fused: Dict[Tuple[int, int], CellStats] = {}
        for sid, submap in self.submaps.items():
            tf = self.transforms_to_canonical.get(sid)
            if tf is None:
                continue
            _accumulate_transformed_submap(submap, tf, self.cell_mm, fused)
        return fused

    def export_ros_pgm_yaml(
        self, pgm_path: Path, yaml_path: Path, padding_cells: int = 8
    ) -> Dict[str, Any]:
        """Write binary P5 grid.pgm and ROS grid.yaml atomically and return map_stats."""
        fused = self.fuse_canonical_cells()
        min_ix, max_ix, min_iy, max_iy = _grid_bounds(fused, padding_cells)
        w, h = max_ix - min_ix + 1, max_iy - min_iy + 1
        assert 0 < w <= MAX_GRID_DIM and 0 < h <= MAX_GRID_DIM, f"grid dims out of bounds: {w}x{h}"
        pixels, counts = _render_pgm_bytes(fused, min_ix, min_iy, w, h)
        _atomic_write_pgm(pgm_path, w, h, pixels)
        res_m = self.cell_mm / 1000.0
        _atomic_write_yaml(yaml_path, pgm_path.name, res_m, min_ix * res_m, min_iy * res_m)
        counts.update(self._meta_stats(w, h, res_m))
        return counts

    def _meta_stats(self, w: int, h: int, res_m: float) -> Dict[str, Any]:
        """Summarize atlas registration counts for metrics JSON."""
        reg = len(self.transforms_to_canonical)
        total = len(self.submaps)
        return {
            "width_px": w,
            "height_px": h,
            "resolution_m": round(res_m, 4),
            "registered_submaps": reg,
            "unregistered_submaps": total - reg,
            "anchor_count": sum(len(v) for v in self.anchors.values()),
            "session_count": len(self.sessions),
        }

    def save(self, path: Path) -> None:
        """Atomically persist atlas JSON via temporary sibling + os.replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "cell_mm": self.cell_mm,
            "canonical_submap_id": self.canonical_submap_id,
            "submaps": {k: v.to_dict() for k, v in self.submaps.items()},
            "anchors": {s: {a: p.to_dict() for a, p in m.items()} for s, m in self.anchors.items()},
            "transforms_to_canonical": {k: v.to_dict() for k, v in self.transforms_to_canonical.items()},
            "sessions": self.sessions,
        }
        tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
        with atlas_file_lock(path):
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "NavMapAtlas":
        """Load persisted atlas JSON if present; otherwise create empty atlas."""
        assert isinstance(path, Path)
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        atlas = cls(cell_mm=float(raw.get("cell_mm", 20.0)))
        atlas.canonical_submap_id = raw.get("canonical_submap_id")
        for sid, sdata in raw.get("submaps", {}).items():
            atlas.submaps[sid] = SubmapLayer.from_dict(sdata)
        for sid, amap in raw.get("anchors", {}).items():
            atlas.anchors[sid] = {aid: Pose2D.from_dict(pd) for aid, pd in amap.items()}
        for sid, tdata in raw.get("transforms_to_canonical", {}).items():
            atlas.transforms_to_canonical[sid] = Transform2D.from_dict(tdata)
            if sid in atlas.submaps and atlas.submaps[sid].cells:
                atlas.marker_obs_counts[sid] = 6
        atlas.sessions = list(raw.get("sessions", []))
        atlas.resolve_registrations()
        return atlas

    def merge_session_from(
        self,
        working: "NavMapAtlas",
        baseline: Optional["NavMapAtlas"] = None,
        explicit_tf_sids: Optional[Set[str]] = None,
        new_sessions: Sequence[Dict[str, Any]] = (),
    ) -> None:
        """Merge session deltas from working into self without holding locks during robot IO."""
        assert isinstance(working, NavMapAtlas), "working NavMapAtlas required"
        for sid, w_layer in working.submaps.items():
            b_layer = baseline.submaps.get(sid) if baseline is not None else None
            if _submap_changed(sid, w_layer, b_layer, sid in self.submaps):
                dst = self.ensure_submap(sid, w_layer.origin_id)
                _merge_submap_cells(dst, w_layer, b_layer)
            self.anchors.setdefault(sid, {}).update(working.anchors.get(sid, {}))
            if sid in working.transforms_to_canonical and _should_copy_tf(self, working, baseline, sid, explicit_tf_sids):
                self.transforms_to_canonical[sid] = working.transforms_to_canonical[sid]
        self.sessions.extend(new_sessions)
        self.resolve_registrations()


def _submap_changed(
    sid: str, w_layer: SubmapLayer, b_layer: Optional[SubmapLayer], already_present: bool
) -> bool:
    """Return True when working submap contains new cells or cell deltas relative to baseline."""
    if not already_present or b_layer is None or len(w_layer.cells) != len(b_layer.cells):
        return True
    for k, w_st in w_layer.cells.items():
        b_st = b_layer.cells.get(k)
        if b_st is None or w_st.obs_count != b_st.obs_count or w_st.cliff_hits != b_st.cliff_hits or abs(w_st.log_odds - b_st.log_odds) > 1e-9:
            return True
    return False


def _should_copy_tf(
    dst: NavMapAtlas, working: NavMapAtlas, baseline: Optional[NavMapAtlas], sid: str, explicit_tf_sids: Optional[Set[str]]
) -> bool:
    """Guard canonical transform transfer against dual-empty-atlas bootstrap collisions."""
    if explicit_tf_sids and sid in explicit_tf_sids:
        return True
    if dst.canonical_submap_id == sid:
        return True
    return sid != working.canonical_submap_id and (baseline is None or sid not in baseline.transforms_to_canonical)


def _merge_submap_cells(dst: SubmapLayer, w_layer: SubmapLayer, b_layer: Optional[SubmapLayer]) -> None:
    """Apply cell log-odds/obs_count/cliff_hits deltas from working submap onto dst."""
    assert dst is not None and w_layer is not None
    for k, w_st in w_layer.cells.items():
        b_st = b_layer.cells.get(k) if b_layer is not None else None
        d_obs = w_st.obs_count - (b_st.obs_count if b_st is not None else 0)
        d_cliff = w_st.cliff_hits - (b_st.cliff_hits if b_st is not None else 0)
        d_lo = w_st.log_odds - (b_st.log_odds if b_st is not None else 0.0)
        if d_obs == 0 and d_cliff == 0 and abs(d_lo) <= 1e-9:
            continue
        c_dst = dst.cells.setdefault(k, CellStats())
        c_dst.log_odds = max(LOG_ODDS_MIN, min(LOG_ODDS_MAX, c_dst.log_odds + d_lo))
        c_dst.cliff_hits = max(0, c_dst.cliff_hits + d_cliff)
        c_dst.obs_count = max(0, c_dst.obs_count + d_obs)
        c_dst.last_content = w_st.last_content


def _estimate_relative_transform(
    src_anchors: Dict[str, Pose2D], dst_anchors: Dict[str, Pose2D]
) -> Optional[Transform2D]:
    """Compute SE(2) transform src -> dst from overlapping named anchors."""
    from .types_and_math import fit_se2_from_anchors

    shared = sorted(set(src_anchors.keys()) & set(dst_anchors.keys()))
    if not shared:
        return None
    pairs = [(src_anchors[name], dst_anchors[name]) for name in shared]
    return fit_se2_from_anchors(pairs)


def _accumulate_transformed_submap(
    submap: SubmapLayer, tf: Transform2D, target_cell_mm: float, out: Dict[Tuple[int, int], CellStats]
) -> None:
    """Transform occupied/free cells from a source submap into target grid."""
    step = submap.cell_mm
    for (ix, iy), st in submap.cells.items():
        if st.obs_count == 0:
            continue
        wx, wy = tf.apply_point((ix + 0.5) * step, (iy + 0.5) * step)
        tix = int(math.floor(wx / target_cell_mm))
        tiy = int(math.floor(wy / target_cell_mm))
        dst = out.setdefault((tix, tiy), CellStats())
        dst.log_odds = max(LOG_ODDS_MIN, min(LOG_ODDS_MAX, dst.log_odds + st.log_odds))
        dst.cliff_hits += st.cliff_hits
        dst.obs_count += st.obs_count
        dst.last_content = st.last_content


def _grid_bounds(fused: Dict[Tuple[int, int], CellStats], pad: int) -> Tuple[int, int, int, int]:
    """Compute padded integer bounds covering all fused cells (minimum 1x1 around 0)."""
    if not fused:
        return (-pad, pad, -pad, pad)
    xs = [k[0] for k in fused]
    ys = [k[1] for k in fused]
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def _render_pgm_bytes(
    fused: Dict[Tuple[int, int], CellStats], min_ix: int, min_iy: int, w: int, h: int
) -> Tuple[bytes, Dict[str, int]]:
    """Render binary ROS map_saver PGM bytes (row 0 = top max_y) and counts."""
    buf = bytearray([205]) * (w * h)
    free_n = occ_n = cliff_n = 0
    for (ix, iy), st in fused.items():
        col = ix - min_ix
        row = h - 1 - (iy - min_iy)
        if not (0 <= col < w and 0 <= row < h):
            continue
        if st.cliff_hits > 0:
            buf[row * w + col] = 0
            cliff_n += 1
        elif st.log_odds >= OCC_THRESHOLD:
            buf[row * w + col] = 0
            occ_n += 1
        elif st.log_odds <= FREE_THRESHOLD:
            buf[row * w + col] = 254
            free_n += 1
    unk_n = max(0, (w * h) - free_n - occ_n - cliff_n)
    return bytes(buf), {"free_cells": free_n, "occ_cells": occ_n, "cliff_cells": cliff_n, "unknown_cells": unk_n, "known_cells": free_n + occ_n + cliff_n}


def _atomic_write_pgm(path: Path, w: int, h: int, pixels: bytes) -> None:
    """Write P5 PGM atomically via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(pixels)
    os.replace(tmp, path)


def _atomic_write_yaml(
    path: Path, image_name: str, res_m: float, ox_m: float, oy_m: float
) -> None:
    """Write ROS map_server YAML metadata atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    body = (
        f"image: {image_name}\n"
        f"resolution: {res_m:.4f}\n"
        f"origin: [{ox_m:.4f}, {oy_m:.4f}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def compress_occupied_to_boxes(
    atlas: NavMapAtlas, max_boxes: int = 48
) -> List[Tuple[float, float, float, float]]:
    """Merge occupied canonical cells into bounded axis-aligned (cx_mm, cy_mm, sx_mm, sy_mm) boxes."""
    fused = atlas.fuse_canonical_cells()
    occ = {(ix, iy) for (ix, iy), st in fused.items() if (st.log_odds >= OCC_THRESHOLD or st.cliff_hits > 0)}
    if not occ or len(occ) > MAX_REHYDRATE_CELLS:
        return []
    boxes: List[Tuple[float, float, float, float]] = []
    remaining = set(occ)
    while remaining and len(boxes) < min(max_boxes, MAX_REHYDRATE_BOXES):
        ix0, iy0 = min(remaining)
        ix1, iy1 = _grow_box(remaining, ix0, iy0)
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                remaining.discard((ix, iy))
        cs = atlas.cell_mm
        boxes.append(((ix0 + ix1 + 1) * 0.5 * cs, (iy0 + iy1 + 1) * 0.5 * cs, (ix1 - ix0 + 1) * cs, (iy1 - iy0 + 1) * cs))
    return boxes


def _grow_box(occ: set[Tuple[int, int]], ix0: int, iy0: int) -> Tuple[int, int]:
    """Greedy maximal horizontal run then vertical merge over sparse occupied cells."""
    ix1 = ix0
    for step in range(1, MAX_LEAF_SPAN):
        if (ix0 + step, iy0) in occ:
            ix1 = ix0 + step
        else:
            break
    iy1 = iy0
    for row_step in range(1, MAX_LEAF_SPAN):
        if all((ix, iy0 + row_step) in occ for ix in range(ix0, ix1 + 1)):
            iy1 = iy0 + row_step
        else:
            break
    return (ix1, iy1)


def rehydrate_onboard_obstacles(
    robot: Any,
    atlas: NavMapAtlas,
    submap_id: str,
    max_boxes: int = 48,
    wall_height_mm: float = 80.0,
) -> int:
    """Inject canonical obstacle boxes back into Vector's active origin via FixedCustomObject."""
    from anki_vector.util import Pose, radians  # type: ignore

    tf_to_can = atlas.transforms_to_canonical.get(submap_id)
    if tf_to_can is None:
        return 0
    tf_to_bot = tf_to_can.invert()
    boxes = compress_occupied_to_boxes(atlas, max_boxes=max_boxes)
    if not boxes:
        return 0
    robot.world.delete_custom_objects(
        delete_custom_marker_objects=False,
        delete_fixed_custom_objects=True,
        delete_custom_object_archetypes=False,
    )
    created = 0
    oid = int(getattr(robot.pose, "origin_id", 0))
    for cx, cy, sx, sy in boxes:
        bx, by = tf_to_bot.apply_point(cx, cy)
        pose = Pose(bx, by, wall_height_mm * 0.5, angle_z=radians(tf_to_bot.dtheta_rad), origin_id=oid)
        robot.world.create_custom_fixed_object(pose, x_size_mm=sx, y_size_mm=sy, z_size_mm=wall_height_mm, use_robot_origin=True)
        created += 1
    return created
