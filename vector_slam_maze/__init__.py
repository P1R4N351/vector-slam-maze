"""Persistent multi-session SLAM atlas, visual-inertial SLAM, sensor bridge, and maze solver."""

from __future__ import annotations

from typing import Any, List

from .maze_solver import FloodFillPlanner, MazeGrid, RightWallFollower
from .movement_directives import (
    MotionResult,
    STOP_BLOCKED,
    STOP_CLIFF,
    STOP_COMPLETED,
    STOP_REFUSED,
    STOP_UNSUPPORTED,
    bounded_drive,
    bounded_turn,
    execute_bounded_directive,
    execute_directive_batch,
    plan_directive,
    plan_directive_batch,
)
from .navmap_atlas import NavMapAtlas, SubmapLayer, compress_occupied_to_boxes, extract_quadtree_leaves
from .types_and_math import NavContent, Pose2D, Transform2D, fit_se2_from_anchors

__version__ = "0.4.1"

_LAZY_EXPORTS = {
    "CameraIntrinsics": "sensor_bridge",
    "VectorRos2Bridge": "sensor_bridge",
    "check_ros2_environment": "sensor_bridge",
    "export_slam_dataset": "sensor_bridge",
    "load_replay_dataset": "sensor_bridge",
    "synthesize_laserscan_from_tof_and_atlas": "sensor_bridge",
    "assess_illumination": "camera_characterize",
    "estimate_rolling_shutter_metrics": "camera_characterize",
    "evaluate_visual_vs_odom_drift": "camera_characterize",
    "measure_sharpness_tenengrad": "camera_characterize",
    "measure_stream_timing": "camera_characterize",
    "phase_correlate_1d_shift": "camera_characterize",
    "VisualKeyframe": "visual_slam",
    "VisualPoseGraphSLAM": "visual_slam",
}

__all__ = [
    "CameraIntrinsics",
    "FloodFillPlanner",
    "MazeGrid",
    "MotionResult",
    "NavContent",
    "NavMapAtlas",
    "Pose2D",
    "RightWallFollower",
    "STOP_BLOCKED",
    "STOP_CLIFF",
    "STOP_COMPLETED",
    "STOP_REFUSED",
    "STOP_UNSUPPORTED",
    "SubmapLayer",
    "Transform2D",
    "VectorRos2Bridge",
    "VisualKeyframe",
    "VisualPoseGraphSLAM",
    "__version__",
    "assess_illumination",
    "bounded_drive",
    "bounded_turn",
    "check_ros2_environment",
    "compress_occupied_to_boxes",
    "estimate_rolling_shutter_metrics",
    "evaluate_visual_vs_odom_drift",
    "execute_bounded_directive",
    "execute_directive_batch",
    "export_slam_dataset",
    "extract_quadtree_leaves",
    "fit_se2_from_anchors",
    "load_replay_dataset",
    "measure_sharpness_tenengrad",
    "measure_stream_timing",
    "phase_correlate_1d_shift",
    "plan_directive",
    "plan_directive_batch",
    "synthesize_laserscan_from_tof_and_atlas",
]


def __getattr__(name: str) -> Any:
    """Lazily load numpy/Pillow-dependent vision and bridge exports on demand."""
    assert isinstance(name, str) and len(_LAZY_EXPORTS) == 14
    mod_name = _LAZY_EXPORTS.get(name)
    if mod_name == "sensor_bridge":
        from . import sensor_bridge as mod
    elif mod_name == "camera_characterize":
        from . import camera_characterize as mod
    elif mod_name == "visual_slam":
        from . import visual_slam as mod
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(mod, name)


def __dir__() -> List[str]:
    """Expose both eager and lazy package symbols for introspection."""
    assert len(__all__) > 20 and "__version__" in __all__
    return sorted(__all__)
