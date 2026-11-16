"""Visual-inertial pose-graph SLAM, ZNCC yaw estimation, loop closure, and relocalization."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera_characterize import assess_illumination
from .navmap_atlas import NavMapAtlas
from .sensor_bridge import CameraIntrinsics, InertialOdomSample
from .types_and_math import Pose2D, normalize_angle

SIG_GRID_W = 12
SIG_GRID_H = 8
SIG_DIM = SIG_GRID_W * SIG_GRID_H * 3  # 288-D zero-mean + gradient signature
GRID_COLS = 8
GRID_ROWS = 6
PATCH_RADIUS = 4  # 9x9 normalized patch
MIN_CORNER_SCORE = 3.0
MIN_PATCH_STD = 3.0
MIN_RELAX_ITERS = 15
MAX_KEYFRAMES = 2000


def compute_appearance_signature(gray: np.ndarray) -> np.ndarray:
    """Compute illumination-invariant 288-D gradient + zero-mean intensity signature."""
    assert gray.ndim == 2 and gray.shape[0] >= SIG_GRID_H and gray.shape[1] >= SIG_GRID_W
    h, w = gray.shape
    bh, bw = h // SIG_GRID_H, w // SIG_GRID_W
    y0, x0 = (h - bh * SIG_GRID_H) // 2, (w - bw * SIG_GRID_W) // 2
    crop = gray[y0 : y0 + bh * SIG_GRID_H, x0 : x0 + bw * SIG_GRID_W]
    raw_b = crop.reshape(SIG_GRID_H, bh, SIG_GRID_W, bw).mean(axis=(1, 3)).astype(np.float32)
    padded = np.pad(raw_b, ((0, 0), (1, 1)), mode="edge")
    blocks = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    l0 = blocks - float(np.mean(blocks))
    gx = np.diff(blocks, axis=1, prepend=blocks[:, :1])
    gy = np.diff(blocks, axis=0, prepend=blocks[:1, :])
    vec = np.concatenate([l0.ravel(), gx.ravel(), gy.ravel()]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-6 else np.zeros(SIG_DIM, dtype=np.float32)


def _corner_score_map(gray: np.ndarray) -> np.ndarray:
    """Compute smoothed Shi-Tomasi minimum-eigenvalue corner score map."""
    gx = 0.5 * (gray[1:-1, 2:] - gray[1:-1, :-2])
    gy = 0.5 * (gray[2:, 1:-1] - gray[:-2, 1:-1])
    ixx, iyy, ixy = gx * gx, gy * gy, gx * gy
    sxx = (
        ixx[:-2, :-2] + ixx[:-2, 1:-1] + ixx[:-2, 2:]
        + ixx[1:-1, :-2] + ixx[1:-1, 1:-1] + ixx[1:-1, 2:]
        + ixx[2:, :-2] + ixx[2:, 1:-1] + ixx[2:, 2:]
    ) / 9.0
    syy = (
        iyy[:-2, :-2] + iyy[:-2, 1:-1] + iyy[:-2, 2:]
        + iyy[1:-1, :-2] + iyy[1:-1, 1:-1] + iyy[1:-1, 2:]
        + iyy[2:, :-2] + iyy[2:, 1:-1] + iyy[2:, 2:]
    ) / 9.0
    sxy = (
        ixy[:-2, :-2] + ixy[:-2, 1:-1] + ixy[:-2, 2:]
        + ixy[1:-1, :-2] + ixy[1:-1, 1:-1] + ixy[1:-1, 2:]
        + ixy[2:, :-2] + ixy[2:, 1:-1] + ixy[2:, 2:]
    ) / 9.0
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = np.sqrt(np.maximum(0.0, 0.25 * tr * tr - det))
    out = np.zeros_like(gray, dtype=np.float32)
    out[2:-2, 2:-2] = (0.5 * tr - disc).astype(np.float32)
    return out


def _append_cell_peaks(
    gray: np.ndarray,
    sub: np.ndarray,
    x0: int,
    y0: int,
    r: int,
    pts: List[List[float]],
    descs: List[np.ndarray],
) -> None:
    """Extract up to 2 NMS-separated normalized patches from one spatial grid cell."""
    work = sub.copy()
    for _ in range(2):
        if work.size == 0 or float(np.max(work)) < MIN_CORNER_SCORE:
            break
        ly, lx = np.unravel_index(int(np.argmax(work)), work.shape)
        py, px = int(y0 + ly), int(x0 + lx)
        work[max(0, ly - 5) : ly + 6, max(0, lx - 5) : lx + 6] = -1.0
        patch = gray[py - r : py + r + 1, px - r : px + r + 1].astype(np.float32)
        if float(np.std(patch)) < MIN_PATCH_STD:
            continue
        zm = patch.ravel() - float(np.mean(patch))
        norm = float(np.linalg.norm(zm))
        if norm > 1e-5:
            pts.append([float(px), float(py)])
            descs.append(zm / norm)


def extract_keypoints_and_descriptors(
    gray: np.ndarray,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
    patch_radius: int = PATCH_RADIUS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select spatial grid Shi-Tomasi peaks and extract 9x9 ZNCC normalized patches."""
    h, w = gray.shape
    r = patch_radius
    p_len = (2 * r + 1) ** 2
    if h < 2 * r + 6 or w < 2 * r + 6:
        return (np.zeros((0, 2), dtype=np.float32), np.zeros((0, p_len), dtype=np.float32))
    scores = _corner_score_map(gray.astype(np.float32))
    ys = np.linspace(r + 2, h - r - 2, grid_rows + 1, dtype=int)
    xs = np.linspace(r + 2, w - r - 2, grid_cols + 1, dtype=int)
    pts: List[List[float]] = []
    descs: List[np.ndarray] = []
    for gy_i in range(grid_rows):
        for gx_i in range(grid_cols):
            sub = scores[ys[gy_i] : ys[gy_i + 1], xs[gx_i] : xs[gx_i + 1]]
            _append_cell_peaks(gray, sub, int(xs[gx_i]), int(ys[gy_i]), r, pts, descs)
    if not pts:
        return (np.zeros((0, 2), dtype=np.float32), np.zeros((0, p_len), dtype=np.float32))
    return (np.asarray(pts, dtype=np.float32), np.asarray(descs, dtype=np.float32))


def estimate_visual_yaw_zncc(
    kps_a: Tuple[np.ndarray, np.ndarray],
    kps_b: Tuple[np.ndarray, np.ndarray],
    fx: float,
    cx: float,
    min_zncc: float = 0.65,
    max_abs_dy_px: float = 18.0,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Estimate CCW yaw delta (rad) from frame A to B via mutual ZNCC matching + MAD filter."""
    pts_a, desc_a = kps_a
    pts_b, desc_b = kps_b
    if len(pts_a) < 4 or len(pts_b) < 4 or fx <= 0.0:
        return (None, {"inliers": 0, "reason": "insufficient_keypoints"})
    sim = desc_a @ desc_b.T
    best_b = np.argmax(sim, axis=1)
    best_a = np.argmax(sim, axis=0)
    cand_i, cand_j, scores = [], [], []
    for i in range(len(pts_a)):
        j = int(best_b[i])
        if int(best_a[j]) == i and float(sim[i, j]) >= min_zncc:
            if abs(float(pts_b[j, 1] - pts_a[i, 1])) <= max_abs_dy_px:
                cand_i.append(i)
                cand_j.append(j)
                scores.append(float(sim[i, j]))
    if len(cand_i) < 4:
        return (None, {"inliers": len(cand_i), "reason": "few_mutual_matches"})
    dth = np.array([normalize_angle(math.atan2(float(pts_b[j, 0]) - cx, fx) - math.atan2(float(pts_a[i, 0]) - cx, fx)) for i, j in zip(cand_i, cand_j)])
    med = float(np.median(dth))
    inlier_mask = np.array([abs(normalize_angle(float(v) - med)) <= 0.06 for v in dth])
    n_in = int(np.sum(inlier_mask))
    if n_in < 4:
        return (None, {"inliers": n_in, "reason": "consensus_failed"})
    u_span = float(np.ptp(pts_a[np.asarray(cand_i)[inlier_mask], 0]))
    if u_span < 35.0:
        return (None, {"inliers": n_in, "reason": "narrow_spatial_span"})
    in_vals = dth[inlier_mask]
    mean_dth = normalize_angle(float(math.atan2(np.mean(np.sin(in_vals)), np.mean(np.cos(in_vals)))))
    return (mean_dth, {"inliers": n_in, "median_zncc": round(float(np.median(np.asarray(scores)[inlier_mask])), 4), "yaw_std_rad": round(float(np.std(in_vals)), 5), "u_span_px": round(u_span, 1)})


@dataclass
class VisualKeyframe:
    """Persisted visual keyframe holding SE(2) pose, signature, keypoints, and ZNCC patches."""

    kf_id: int
    timestamp_s: float
    pose: Pose2D
    signature: List[float]
    pts: List[List[float]] = field(default_factory=list)
    descs: List[List[float]] = field(default_factory=list)
    mean_luma: float = 0.0
    prox_mm: float = 1000.0
    prox_found: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize keyframe compactly for JSON persistence."""
        return {
            "kf_id": self.kf_id,
            "timestamp_s": round(self.timestamp_s, 6),
            "pose": self.pose.to_dict(),
            "signature": [round(float(v), 5) for v in self.signature],
            "pts": [[round(float(c), 2) for c in pt] for pt in self.pts],
            "descs": [[round(float(d), 4) for d in row] for row in self.descs],
            "mean_luma": round(self.mean_luma, 2),
            "prox_mm": round(self.prox_mm, 1),
            "prox_found": self.prox_found,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "VisualKeyframe":
        """Load VisualKeyframe from JSON dict."""
        return cls(
            kf_id=int(raw["kf_id"]),
            timestamp_s=float(raw["timestamp_s"]),
            pose=Pose2D.from_dict(raw["pose"]),
            signature=[float(v) for v in raw["signature"]],
            pts=[[float(c) for c in pt] for pt in raw.get("pts", [])],
            descs=[[float(d) for d in row] for row in raw.get("descs", [])],
            mean_luma=float(raw.get("mean_luma", 0.0)),
            prox_mm=float(raw.get("prox_mm", 1000.0)),
            prox_found=bool(raw.get("prox_found", False)),
        )


@dataclass(frozen=True)
class PoseGraphEdge:
    """Relative SE(2) constraint u -> v expressed in node u frame."""

    u: int
    v: int
    dx_mm: float
    dy_mm: float
    dtheta_rad: float
    is_loop: bool = False
    weight_xy: float = 1.0
    weight_yaw: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to JSON dict."""
        return {
            "u": self.u,
            "v": self.v,
            "dx_mm": round(self.dx_mm, 3),
            "dy_mm": round(self.dy_mm, 3),
            "dtheta_rad": round(self.dtheta_rad, 6),
            "is_loop": self.is_loop,
            "weight_xy": self.weight_xy,
            "weight_yaw": self.weight_yaw,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PoseGraphEdge":
        """Reconstruct PoseGraphEdge from JSON dict."""
        return cls(int(raw["u"]), int(raw["v"]), float(raw["dx_mm"]), float(raw["dy_mm"]), float(raw["dtheta_rad"]), bool(raw.get("is_loop", False)), float(raw.get("weight_xy", 1.0)), float(raw.get("weight_yaw", 1.0)))


def _relative_se2(a: Pose2D, b: Pose2D) -> Tuple[float, float, float]:
    """Express Pose2D b inside Pose2D a local coordinates."""
    dxw, dyw = b.x_mm - a.x_mm, b.y_mm - a.y_mm
    c, s = math.cos(a.theta_rad), math.sin(a.theta_rad)
    return (c * dxw + s * dyw, -s * dxw + c * dyw, normalize_angle(b.theta_rad - a.theta_rad))


class VisualPoseGraphSLAM:
    """Host SE(2) visual-inertial pose-graph SLAM with darkness gating & loop closure."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        min_keyframe_mm: float = 45.0,
        min_keyframe_rad: float = 0.22,
        min_loop_gap: int = 4,
        min_sig_cos: float = 0.78,
    ) -> None:
        self.intrinsics = intrinsics
        self.min_keyframe_mm = float(min_keyframe_mm)
        self.min_keyframe_rad = float(min_keyframe_rad)
        self.min_loop_gap = max(2, int(min_loop_gap))
        self.min_sig_cos = float(min_sig_cos)
        self.keyframes: List[VisualKeyframe] = []
        self.edges: List[PoseGraphEdge] = []
        self.loop_closures: List[Dict[str, Any]] = []
        self.current_pose: Optional[Pose2D] = None
        self.trajectory: List[Pose2D] = []
        self._last_odom: Optional[InertialOdomSample] = None
        self._last_kps: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _predict_step_pose(
        self, odom: InertialOdomSample, kps: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[Pose2D, Optional[float]]:
        """Fuse wheel/gyro delta with optional inter-frame ZNCC visual yaw."""
        if self._last_odom is None or self.current_pose is None:
            return (Pose2D(odom.pose.x_mm, odom.pose.y_mm, odom.pose.theta_rad, odom.pose.origin_id), None)
        dx_o, dy_o, dth_o = _relative_se2(self._last_odom.pose, odom.pose)
        dt = max(0.0, min(0.5, odom.timestamp_s - self._last_odom.timestamp_s))
        dth_gyro = float(odom.gyro_xyz_rad_s[2]) * dt
        dth_iner = dth_o if abs(dth_o) >= abs(dth_gyro) * 0.25 else dth_gyro
        vis_dth: Optional[float] = None
        if self._last_kps is not None and kps is not None:
            vis_dth, _ = estimate_visual_yaw_zncc(self._last_kps, kps, self.intrinsics.fx, self.intrinsics.cx)
        dth = normalize_angle(0.75 * vis_dth + 0.25 * dth_iner) if vis_dth is not None else dth_iner
        c, s = math.cos(self.current_pose.theta_rad), math.sin(self.current_pose.theta_rad)
        nx = self.current_pose.x_mm + c * dx_o - s * dy_o
        ny = self.current_pose.y_mm + s * dx_o + c * dy_o
        nth = normalize_angle(self.current_pose.theta_rad + dth)
        return (Pose2D(nx, ny, nth, odom.pose.origin_id), vis_dth)

    def _should_add_keyframe(self, pose: Pose2D) -> bool:
        """Check bounded keyframe budget and spatial/angular displacement trigger."""
        if len(self.keyframes) >= MAX_KEYFRAMES:
            return False
        if not self.keyframes:
            return True
        prev = self.keyframes[-1].pose
        dist = math.hypot(pose.x_mm - prev.x_mm, pose.y_mm - prev.y_mm)
        ang = abs(normalize_angle(pose.theta_rad - prev.theta_rad))
        return dist >= self.min_keyframe_mm or ang >= self.min_keyframe_rad

    def _try_loop_closure(self, cur_kf: VisualKeyframe, sig: np.ndarray, kps: Tuple[np.ndarray, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Match cur_kf against non-adjacent older keyframes using signature + ZNCC."""
        limit = cur_kf.kf_id - self.min_loop_gap
        if limit <= 0:
            return None
        scored: List[Tuple[float, VisualKeyframe]] = []
        for prev in self.keyframes[:limit]:
            cos_sim = float(np.dot(sig, np.asarray(prev.signature, dtype=np.float32)))
            if cos_sim >= self.min_sig_cos:
                scored.append((cos_sim, prev))
        scored.sort(key=lambda item: item[0], reverse=True)
        for cos_sim, prev in scored[:3]:
            prev_kps = (np.asarray(prev.pts, dtype=np.float32), np.asarray(prev.descs, dtype=np.float32))
            vis_dth, info = estimate_visual_yaw_zncc(prev_kps, kps, self.intrinsics.fx, self.intrinsics.cx)
            if vis_dth is None or int(info.get("inliers", 0)) < 6:
                continue
            # The match measures heading only. A similar view does not mean the
            # same position (a far wall looks alike from 0.5 m apart), so the
            # edge must not constrain translation.
            self.edges.append(PoseGraphEdge(prev.kf_id, cur_kf.kf_id, 0.0, 0.0, vis_dth, is_loop=True, weight_xy=0.0, weight_yaw=2.5))
            event = {"from_kf": prev.kf_id, "to_kf": cur_kf.kf_id, "sig_cos": round(cos_sim, 4), "vis_dth_rad": round(vis_dth, 5), **info}
            self.loop_closures.append(event)
            self.optimize_pose_graph()
            self.current_pose = self.keyframes[-1].pose
            return event
        return None

    def process_frame(
        self,
        gray: np.ndarray,
        odom: InertialOdomSample,
        timestamp_s: float,
        exposure_ms: Optional[int] = None,
        gain: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process one synchronized frame + odom; reject visual keyframes when dark."""
        illum = assess_illumination(gray, exposure_ms=exposure_ms, gain=gain)
        if not illum["visual_slam_viable"]:
            pred_pose, _ = self._predict_step_pose(odom, None)
            self.current_pose = pred_pose
            self.trajectory.append(pred_pose)
            self._last_odom = odom
            # Keypoints from before the gap span several odometry steps; matching
            # them against the next lit frame would add that whole rotation twice.
            self._last_kps = None
            return {"accepted_visual": False, "illumination": illum, "pose": pred_pose.to_dict(), "keyframe_id": None, "loop_closure": None}
        sig = compute_appearance_signature(gray)
        kps = extract_keypoints_and_descriptors(gray)
        pred_pose, vis_dth = self._predict_step_pose(odom, kps)
        self.current_pose = pred_pose
        self._last_odom = odom
        self._last_kps = kps
        kf_id: Optional[int] = None
        loop_ev: Optional[Dict[str, Any]] = None
        if self._should_add_keyframe(pred_pose):
            kf_id = len(self.keyframes)
            kf = VisualKeyframe(kf_id, timestamp_s, pred_pose, sig.tolist(), kps[0].tolist(), kps[1].tolist(), illum["mean_luma"], odom.prox_mm, odom.prox_found)
            if self.keyframes:
                prev_kf = self.keyframes[-1]
                dx, dy, dth = _relative_se2(prev_kf.pose, pred_pose)
                self.edges.append(PoseGraphEdge(prev_kf.kf_id, kf_id, dx, dy, dth, is_loop=False))
            self.keyframes.append(kf)
            loop_ev = self._try_loop_closure(kf, sig, kps)
        self.trajectory.append(self.current_pose)
        return {"accepted_visual": True, "illumination": illum, "pose": self.current_pose.to_dict(), "vis_dth_rad": round(vis_dth, 5) if vis_dth is not None else None, "keyframe_id": kf_id, "loop_closure": loop_ev}

    def optimize_pose_graph(self, iterations: int = 30) -> float:
        """Run bounded Gauss-Seidel SE(2) relaxation keeping keyframe 0 pinned."""
        n = len(self.keyframes)
        if n < 2 or not self.edges:
            return 0.0
        max_step = 0.0
        for _ in range(max(1, min(60, iterations))):
            max_step = 0.0
            for e in self.edges:
                pu, pv = self.keyframes[e.u].pose, self.keyframes[e.v].pose
                c, s = math.cos(pu.theta_rad), math.sin(pu.theta_rad)
                ex, ey = c * e.dx_mm - s * e.dy_mm, s * e.dx_mm + c * e.dy_mm
                rx = ((pv.x_mm - pu.x_mm) - ex) * e.weight_xy
                ry = ((pv.y_mm - pu.y_mm) - ey) * e.weight_xy
                rth = normalize_angle((pv.theta_rad - pu.theta_rad) - e.dtheta_rad) * min(1.0, e.weight_yaw) * 0.35
                u_s, v_s = (0.0, 1.0) if e.u == 0 else (-0.5, 0.5)
                if e.u != 0:
                    self.keyframes[e.u].pose = Pose2D(pu.x_mm - 0.5 * u_s * rx, pu.y_mm - 0.5 * u_s * ry, normalize_angle(pu.theta_rad - u_s * rth), pu.origin_id)
                self.keyframes[e.v].pose = Pose2D(pv.x_mm - 0.5 * v_s * rx, pv.y_mm - 0.5 * v_s * ry, normalize_angle(pv.theta_rad - v_s * rth), pv.origin_id)
                max_step = max(max_step, abs(rx) + abs(ry) + abs(rth) * 100.0)
        return round(max_step, 4)

    def relocalize_frame(
        self, gray: np.ndarray, exposure_ms: Optional[int] = None, gain: Optional[float] = None
    ) -> Dict[str, Any]:
        """Relocalize a single query frame against stored keyframes, gated by illumination."""
        illum = assess_illumination(gray, exposure_ms=exposure_ms, gain=gain)
        if not illum["visual_slam_viable"]:
            return {"ok": False, "matched": False, "reason": illum["status"], "illumination": illum}
        if not self.keyframes:
            return {"ok": False, "matched": False, "reason": "empty_visual_map", "illumination": illum}
        sig = compute_appearance_signature(gray)
        kps = extract_keypoints_and_descriptors(gray)
        cands = sorted(self.keyframes, key=lambda kf: float(np.dot(sig, np.asarray(kf.signature, dtype=np.float32))), reverse=True)
        for kf in cands[:4]:
            cos_sim = float(np.dot(sig, np.asarray(kf.signature, dtype=np.float32)))
            if cos_sim < min(0.65, self.min_sig_cos):
                break
            kf_kps = (np.asarray(kf.pts, dtype=np.float32), np.asarray(kf.descs, dtype=np.float32))
            vis_dth, info = estimate_visual_yaw_zncc(kf_kps, kps, self.intrinsics.fx, self.intrinsics.cx)
            if vis_dth is not None:
                est = Pose2D(kf.pose.x_mm, kf.pose.y_mm, normalize_angle(kf.pose.theta_rad + vis_dth), kf.pose.origin_id)
                return {"ok": True, "matched": True, "matched_kf_id": kf.kf_id, "sig_cos": round(cos_sim, 4), "pose": est.to_dict(), "illumination": illum, **info}
        return {"ok": False, "matched": False, "reason": "no_visual_match", "illumination": illum}

    def sync_into_navmap_atlas(
        self, atlas: NavMapAtlas, session_tag: str, origin_id: int = 0
    ) -> Tuple[str, int]:
        """Project optimized keyframes' ToF obstacle hits into a persistent NavMapAtlas submap."""
        sid = f"{session_tag}:vslam_origin_{origin_id}"
        submap = atlas.ensure_submap(sid, origin_id)
        if atlas.canonical_submap_id == sid or not atlas.anchors.get(sid):
            atlas.assume_same_frame_as_canonical(sid)
        updates = 0
        for kf in self.keyframes:
            updates += submap.integrate_tof_ray(kf.pose, kf.prox_mm, kf.prox_found, False)
        atlas.sessions.append({"tag": session_tag, "kind": "visual_slam", "origin_id": origin_id, "submap_id": sid, "keyframes": len(self.keyframes), "loop_closures": len(self.loop_closures)})
        return (sid, updates)

    def save(self, path: Path) -> None:
        """Persist visual map atomically via temp file + os.replace."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "intrinsics": self.intrinsics.to_dict(),
            "keyframes": [kf.to_dict() for kf in self.keyframes],
            "edges": [e.to_dict() for e in self.edges],
            "loop_closures": self.loop_closures,
        }
        tmp = p.with_suffix(f"{p.suffix}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: Path) -> "VisualPoseGraphSLAM":
        """Load persisted visual pose-graph map from JSON."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        intr = CameraIntrinsics.from_dict(raw["intrinsics"])
        slam = cls(intr)
        slam.keyframes = [VisualKeyframe.from_dict(k) for k in raw.get("keyframes", [])]
        slam.edges = [PoseGraphEdge.from_dict(e) for e in raw.get("edges", [])]
        slam.loop_closures = list(raw.get("loop_closures", []))
        if slam.keyframes:
            slam.current_pose = slam.keyframes[-1].pose
        return slam
