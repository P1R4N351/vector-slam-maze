"""Core SE(2) geometry, NavMap content constants, and anchor transforms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Iterable, Sequence, Tuple

MAX_ANCHOR_PAIRS = 256


class NavContent(IntEnum):
    """Mirror of anki_vector.nav_map.NavNodeContentTypes integer values."""

    UNKNOWN = 0
    CLEAR_OF_OBSTACLE = 1
    CLEAR_OF_CLIFF = 2
    OBSTACLE_CUBE = 3
    OBSTACLE_PROXIMITY = 4
    OBSTACLE_PROXIMITY_EXPLORED = 5
    OBSTACLE_UNRECOGNIZED = 6
    CLIFF = 7
    INTERESTING_EDGE = 8
    NON_INTERESTING_EDGE = 9


def normalize_angle(rad: float) -> float:
    """Wrap angle in radians into [-pi, pi]."""
    assert math.isfinite(rad), "angle must be finite"
    return math.atan2(math.sin(rad), math.cos(rad))


@dataclass(frozen=True)
class Pose2D:
    """Planar pose in millimeters and radians inside a specific origin frame."""

    x_mm: float
    y_mm: float
    theta_rad: float
    origin_id: int = 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "x_mm": round(float(self.x_mm), 3),
            "y_mm": round(float(self.y_mm), 3),
            "theta_rad": round(float(normalize_angle(self.theta_rad)), 6),
            "origin_id": int(self.origin_id),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pose2D":
        """Deserialize from a dictionary."""
        assert isinstance(data, dict), "pose dict required"
        return cls(
            x_mm=float(data["x_mm"]),
            y_mm=float(data["y_mm"]),
            theta_rad=float(data.get("theta_rad", 0.0)),
            origin_id=int(data.get("origin_id", 0)),
        )


@dataclass(frozen=True)
class Transform2D:
    """Rigid SE(2) transform mapping p_dst = R(dtheta) * p_src + t."""

    tx_mm: float
    ty_mm: float
    dtheta_rad: float

    def apply_point(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """Transform a 2D point into the destination frame."""
        c = math.cos(self.dtheta_rad)
        s = math.sin(self.dtheta_rad)
        nx = c * x_mm - s * y_mm + self.tx_mm
        ny = s * x_mm + c * y_mm + self.ty_mm
        return (nx, ny)

    def apply_pose(self, pose: Pose2D, target_origin_id: int = 0) -> Pose2D:
        """Transform a Pose2D into the destination frame."""
        nx, ny = self.apply_point(pose.x_mm, pose.y_mm)
        nt = normalize_angle(pose.theta_rad + self.dtheta_rad)
        return Pose2D(nx, ny, nt, target_origin_id)

    def compose(self, other: "Transform2D") -> "Transform2D":
        """Return self(other(p)) mapping src_other -> dst_self."""
        c = math.cos(self.dtheta_rad)
        s = math.sin(self.dtheta_rad)
        tx = c * other.tx_mm - s * other.ty_mm + self.tx_mm
        ty = s * other.tx_mm + c * other.ty_mm + self.ty_mm
        return Transform2D(tx, ty, normalize_angle(self.dtheta_rad + other.dtheta_rad))

    def invert(self) -> "Transform2D":
        """Return inverse transform dst -> src."""
        c = math.cos(self.dtheta_rad)
        s = math.sin(self.dtheta_rad)
        inv_tx = -(c * self.tx_mm + s * self.ty_mm)
        inv_ty = s * self.tx_mm - c * self.ty_mm
        return Transform2D(inv_tx, inv_ty, normalize_angle(-self.dtheta_rad))

    def to_dict(self) -> dict:
        """Serialize to JSON dict."""
        return {
            "tx_mm": round(float(self.tx_mm), 4),
            "ty_mm": round(float(self.ty_mm), 4),
            "dtheta_rad": round(float(normalize_angle(self.dtheta_rad)), 6),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transform2D":
        """Deserialize from JSON dict."""
        assert isinstance(data, dict), "transform dict required"
        return cls(
            tx_mm=float(data["tx_mm"]),
            ty_mm=float(data["ty_mm"]),
            dtheta_rad=float(data["dtheta_rad"]),
        )

    @classmethod
    def from_anchor_pair(cls, src_pose: Pose2D, dst_pose: Pose2D) -> "Transform2D":
        """Compute exact SE(2) transform T such that T(src_pose) == dst_pose."""
        dtheta = normalize_angle(dst_pose.theta_rad - src_pose.theta_rad)
        c = math.cos(dtheta)
        s = math.sin(dtheta)
        tx = dst_pose.x_mm - (c * src_pose.x_mm - s * src_pose.y_mm)
        ty = dst_pose.y_mm - (s * src_pose.x_mm + c * src_pose.y_mm)
        return cls(tx, ty, dtheta)


def identity_transform() -> Transform2D:
    """Return the identity SE(2) transform."""
    return Transform2D(0.0, 0.0, 0.0)


def pose2d_from_sdk_pose(sdk_pose: object, fallback_origin_id: int = 0) -> Pose2D:
    """Extract Pose2D from anki_vector.util.Pose without requiring SDK import."""
    assert sdk_pose is not None, "sdk_pose required"
    pos = getattr(sdk_pose, "position", None)
    rot = getattr(sdk_pose, "rotation", None)
    x = float(getattr(pos, "x", 0.0)) if pos is not None else 0.0
    y = float(getattr(pos, "y", 0.0)) if pos is not None else 0.0
    ang = getattr(rot, "angle_z", None) if rot is not None else None
    theta = float(getattr(ang, "radians", 0.0)) if ang is not None else 0.0
    oid = int(getattr(sdk_pose, "origin_id", fallback_origin_id))
    return Pose2D(x, y, normalize_angle(theta), oid)


def fit_se2_from_anchors(pairs: Sequence[Tuple[Pose2D, Pose2D]]) -> Transform2D:
    """Fit least-squares SE(2) transform mapping src_pose -> dst_pose."""
    count = len(pairs)
    assert 1 <= count <= MAX_ANCHOR_PAIRS, f"invalid anchor pair count: {count}"
    if count == 1:
        return Transform2D.from_anchor_pair(pairs[0][0], pairs[0][1])
    sx, sy, dx, dy = _centroids(pairs)
    num, den = 0.0, 0.0
    for src, dst in pairs:
        px, py = src.x_mm - sx, src.y_mm - sy
        qx, qy = dst.x_mm - dx, dst.y_mm - dy
        num += px * qy - py * qx
        den += px * qx + py * qy
    dtheta = _angle_from_cross_dot(num, den, pairs)
    c, s = math.cos(dtheta), math.sin(dtheta)
    return Transform2D(dx - (c * sx - s * sy), dy - (s * sx + c * sy), dtheta)


def _centroids(pairs: Iterable[Tuple[Pose2D, Pose2D]]) -> Tuple[float, float, float, float]:
    """Compute source and destination centroids."""
    sx = sy = dx = dy = 0.0
    n = 0
    for src, dst in pairs:
        sx += src.x_mm
        sy += src.y_mm
        dx += dst.x_mm
        dy += dst.y_mm
        n += 1
    assert n > 0, "at least one pair required"
    return (sx / n, sy / n, dx / n, dy / n)


def _angle_from_cross_dot(
    num: float, den: float, pairs: Sequence[Tuple[Pose2D, Pose2D]]
) -> float:
    """Resolve rotation angle from Procrustes cross/dot or oriented anchor fallback."""
    if math.hypot(num, den) > 1e-6:
        return math.atan2(num, den)
    cos_sum = sum(math.cos(d.theta_rad - s.theta_rad) for s, d in pairs)
    sin_sum = sum(math.sin(d.theta_rad - s.theta_rad) for s, d in pairs)
    return math.atan2(sin_sum, cos_sum)
