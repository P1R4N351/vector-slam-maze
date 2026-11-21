"""Built-in CustomObjectMarker registration and Labyrinth maze marker localization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types_and_math import Pose2D, Transform2D, fit_se2_from_anchors, normalize_angle

MARKER_NAMES: tuple[str, ...] = (
    "Circles2", "Circles3", "Circles4", "Circles5",
    "Diamonds2", "Diamonds3", "Diamonds4", "Diamonds5",
    "Hexagons2", "Hexagons3", "Hexagons4", "Hexagons5",
    "Triangles2", "Triangles3", "Triangles4", "Triangles5",
)
MAX_DEFINED_MARKERS = len(MARKER_NAMES)


@dataclass(frozen=True)
class MazeMarkerAnchor:
    """Known maze cell + heading bound to an onboard CustomObjectMarker."""

    marker_name: str
    row: int
    col: int
    x_mm: float
    y_mm: float
    normal_deg: float = 0.0

    def to_pose(self, origin_id: int = 0) -> Pose2D:
        """Convert marker specification to a canonical Pose2D."""
        return Pose2D(self.x_mm, self.y_mm, normalize_angle(math.radians(self.normal_deg)), origin_id)


def load_marker_map(path: Path) -> Dict[str, MazeMarkerAnchor]:
    """Load JSON file mapping marker_name -> maze coordinates."""
    assert path.is_file(), f"marker map missing: {path}"
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("markers", raw) if isinstance(raw, dict) else raw
    out: Dict[str, MazeMarkerAnchor] = {}
    for item in entries[:MAX_DEFINED_MARKERS]:
        name = str(item["marker_name"])
        out[name] = MazeMarkerAnchor(
            marker_name=name,
            row=int(item["row"]),
            col=int(item["col"]),
            x_mm=float(item["x_mm"]),
            y_mm=float(item["y_mm"]),
            normal_deg=float(item.get("normal_deg", 0.0)),
        )
    return out


def register_custom_wall_markers(
    robot: Any,
    width_mm: float = 60.0,
    height_mm: float = 60.0,
    marker_mm: float = 50.0,
) -> List[str]:
    """Define Vector's 16 built-in CustomObjectMarkers as detectable walls."""
    assert robot is not None, "robot required"
    from anki_vector import objects  # type: ignore

    registered: List[str] = []
    for idx, marker_name in enumerate(MARKER_NAMES):
        marker_enum = getattr(objects.CustomObjectMarkers, marker_name, None)
        type_enum = getattr(objects.CustomObjectTypes, f"CustomType{idx:02d}", None)
        if marker_enum is None or type_enum is None:
            continue
        robot.world.define_custom_wall(
            custom_object_type=type_enum,
            marker=marker_enum,
            width_mm=width_mm,
            height_mm=height_mm,
            marker_width_mm=marker_mm,
            marker_height_mm=marker_mm,
            is_unique=True,
        )
        registered.append(marker_name)
    return registered


def anchor_name_from_object(obj: Any) -> Optional[str]:
    """Produce stable anchor ID for charger, cube, or custom marker."""
    if obj is None:
        return None
    arch = getattr(obj, "archetype", None)
    marker = getattr(arch, "marker_front", None) or getattr(arch, "marker", None)
    if marker is not None:
        return f"marker:{getattr(marker, 'name', str(marker))}"
    cls_name = type(obj).__name__
    if "Charger" in cls_name:
        return "charger"
    if "LightCube" in cls_name:
        return f"cube:{getattr(obj, 'object_id', '0')}"
    return None


def estimate_marker_transform(
    observed_anchors: Dict[str, Pose2D],
    known_markers: Dict[str, MazeMarkerAnchor],
    origin_id: int,
) -> Optional[Transform2D]:
    """Fit SE(2) transform from origin_id to canonical maze frame using visible markers."""
    assert isinstance(observed_anchors, dict) and isinstance(known_markers, dict)
    pairs = []
    for anchor_id in sorted(observed_anchors)[: MAX_DEFINED_MARKERS * 4]:
        if not anchor_id.startswith("marker:"):
            continue
        spec = known_markers.get(anchor_id.split(":", 1)[1])
        obs_pose = observed_anchors[anchor_id]
        if spec is None or obs_pose.origin_id != origin_id:
            continue
        pairs.append((obs_pose, spec.to_pose(origin_id=0)))
        if len(pairs) >= MAX_DEFINED_MARKERS:
            break
    assert len(pairs) <= MAX_DEFINED_MARKERS
    return fit_se2_from_anchors(pairs) if pairs else None


def localize_from_marker_anchors(
    observed_anchors: Dict[str, Pose2D],
    known_markers: Dict[str, MazeMarkerAnchor],
    robot_pose: Pose2D,
) -> Optional[Pose2D]:
    """Recover robot Pose2D in canonical maze frame using visible CustomObjectMarkers."""
    assert robot_pose is not None and isinstance(known_markers, dict)
    tf_src_to_maze = estimate_marker_transform(observed_anchors, known_markers, robot_pose.origin_id)
    if tf_src_to_maze is None:
        return None
    return tf_src_to_maze.apply_pose(robot_pose, target_origin_id=0)
