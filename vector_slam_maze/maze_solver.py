"""Micromouse flood-fill and Cornell right-wall-following maze solvers for Vector."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cliff_guard import CliffHazard
from .landmarks import MazeMarkerAnchor
from .maze_closed_loop import (
    _sdk_units,
    execute_speed_run_segments,
    fuse_and_localize_step,
    run_floodfill_loop,
    turn_to_canonical_theta,
)
from .navmap_atlas import (
    NavMapAtlas,
    OCC_THRESHOLD,
    FREE_THRESHOLD,
    extract_quadtree_leaves,
    observe_world_anchors,
    rehydrate_onboard_obstacles,
)
from .sdk_session import check_safety, poll_latest_nav_map
from .types_and_math import Pose2D, normalize_angle, pose2d_from_sdk_pose

NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8
ALL_DIRS = (NORTH, EAST, SOUTH, WEST)
OPPOSITE_DIR = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}
DIR_DELTA = {NORTH: (1, 0), EAST: (0, 1), SOUTH: (-1, 0), WEST: (0, -1)}
DIR_THETA = {EAST: 0.0, NORTH: math.pi * 0.5, WEST: math.pi, SOUTH: -math.pi * 0.5}
TURN_RIGHT_OF = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}
TURN_LEFT_OF = {NORTH: WEST, WEST: SOUTH, SOUTH: EAST, EAST: NORTH}

MAX_MAZE_DIM = 128
MAX_MAZE_CELLS = 16384
INF_DIST = 999_999


@dataclass(frozen=True)
class WallFollowDecision:
    """Discrete single-step action produced by the right-wall-follow controller."""

    action: str
    turn_deg: float
    forward_mm: float


class MazeGrid:
    """2D Cartesian maze cell grid compatible with flood-fill and NavMapAtlas."""

    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size_mm: float = 180.0,
        origin_x_mm: float = -90.0,
        origin_y_mm: float = -90.0,
    ) -> None:
        assert 1 <= rows <= MAX_MAZE_DIM and 1 <= cols <= MAX_MAZE_DIM
        assert rows * cols <= MAX_MAZE_CELLS
        assert cell_size_mm > 0.0
        self.rows = int(rows)
        self.cols = int(cols)
        self.cell_size_mm = float(cell_size_mm)
        self.origin_x_mm = float(origin_x_mm)
        self.origin_y_mm = float(origin_y_mm)
        self.walls = [[0] * cols for _ in range(rows)]
        self.known = [[0] * cols for _ in range(rows)]
        self.blocked = [[False] * cols for _ in range(rows)]
        self._init_outer_walls()

    def _init_outer_walls(self) -> None:
        """Mark perimeter boundaries as known impassable walls."""
        for c in range(self.cols):
            self.set_wall(0, c, SOUTH, True)
            self.set_wall(self.rows - 1, c, NORTH, True)
        for r in range(self.rows):
            self.set_wall(r, 0, WEST, True)
            self.set_wall(r, self.cols - 1, EAST, True)

    def in_bounds(self, r: int, c: int) -> bool:
        """Check whether cell indices lie inside the grid."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def set_wall(self, r: int, c: int, direction: int, present: bool = True) -> None:
        """Record wall presence/absence symmetrically on cell and neighbor."""
        if not self.in_bounds(r, c) or direction not in OPPOSITE_DIR:
            return
        self.known[r][c] |= direction
        if present:
            self.walls[r][c] |= direction
        else:
            self.walls[r][c] &= ~direction
        dr, dc = DIR_DELTA[direction]
        nr, nc = r + dr, c + dc
        if self.in_bounds(nr, nc):
            opp = OPPOSITE_DIR[direction]
            self.known[nr][nc] |= opp
            if present:
                self.walls[nr][nc] |= opp
            else:
                self.walls[nr][nc] &= ~opp

    def has_wall(self, r: int, c: int, direction: int, known_only: bool = False) -> bool:
        """Return True if movement along direction is blocked (or unknown when known_only)."""
        if not self.in_bounds(r, c) or direction not in DIR_DELTA:
            return True
        if self.walls[r][c] & direction:
            return True
        if known_only and not (self.known[r][c] & direction):
            return True
        dr, dc = DIR_DELTA[direction]
        nr, nc = r + dr, c + dc
        return (not self.in_bounds(nr, nc)) or self.blocked[nr][nc]

    def world_to_cell(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Map canonical world coordinates (mm) to (row, col) clamped inside grid."""
        col = int(math.floor((x_mm - self.origin_x_mm) / self.cell_size_mm))
        row = int(math.floor((y_mm - self.origin_y_mm) / self.cell_size_mm))
        return (max(0, min(self.rows - 1, row)), max(0, min(self.cols - 1, col)))

    def cell_center_mm(self, r: int, c: int) -> Tuple[float, float]:
        """Return canonical center (x_mm, y_mm) of maze cell (r, c)."""
        return (
            self.origin_x_mm + (c + 0.5) * self.cell_size_mm,
            self.origin_y_mm + (r + 0.5) * self.cell_size_mm,
        )

    @classmethod
    def from_atlas(
        cls, atlas: NavMapAtlas, cell_size_mm: float = 180.0, robot_radius_mm: float = 52.0
    ) -> "MazeGrid":
        """Discretize canonical NavMapAtlas cells into a collision-padded MazeGrid."""
        fused = atlas.fuse_canonical_cells()
        ox, oy, rows, cols = _compute_maze_extent(fused, atlas.cell_mm, cell_size_mm)
        grid = cls(rows=rows, cols=cols, cell_size_mm=cell_size_mm, origin_x_mm=ox, origin_y_mm=oy)
        _populate_cells_from_fused(grid, fused, atlas.cell_mm, robot_radius_mm)
        return grid

    def render_ascii(
        self, path: Optional[Sequence[Tuple[int, int]]] = None, start: Optional[Tuple[int, int]] = None
    ) -> str:
        """Render top-down ASCII visualization (+, -, |, * path, S start)."""
        path_set = set(path or ())
        lines: List[str] = []
        for r in range(self.rows - 1, -1, -1):
            lines.append(_ascii_top_row(self, r))
            lines.append(_ascii_mid_row(self, r, path_set, start))
        lines.append(_ascii_bottom_row(self))
        return "\n".join(lines)


def _compute_maze_extent(
    fused: dict, sub_cell_mm: float, maze_cell_mm: float
) -> Tuple[float, float, int, int]:
    """Compute center-aligned maze origin and dimensions spanning fused map (min 5x5)."""
    if not fused:
        half = 2.5 * maze_cell_mm
        return (-half, -half, 5, 5)
    xs = [(ix + 0.5) * sub_cell_mm for (ix, _iy) in fused]
    ys = [(iy + 0.5) * sub_cell_mm for (_ix, iy) in fused]
    min_c = min(-2, int(round(min(xs) / maze_cell_mm)))
    max_c = max(2, int(round(max(xs) / maze_cell_mm)))
    min_r = min(-2, int(round(min(ys) / maze_cell_mm)))
    max_r = max(2, int(round(max(ys) / maze_cell_mm)))
    cols = min(MAX_MAZE_DIM, max(5, max_c - min_c + 1))
    rows = min(MAX_MAZE_DIM, max(5, max_r - min_r + 1))
    return ((min_c - 0.5) * maze_cell_mm, (min_r - 0.5) * maze_cell_mm, rows, cols)


def _populate_cells_from_fused(
    grid: MazeGrid, fused: dict, sub_cell_mm: float, radius_mm: float
) -> None:
    """Transfer occupied/cliff cells and free-passage evidence from fused atlas into MazeGrid."""
    for (ix, iy), st in fused.items():
        wx, wy = (ix + 0.5) * sub_cell_mm, (iy + 0.5) * sub_cell_mm
        r, c = grid.world_to_cell(wx, wy)
        cx, cy = grid.cell_center_mm(r, c)
        dx, dy = wx - cx, wy - cy
        is_occ = st.cliff_hits > 0 or st.log_odds >= OCC_THRESHOLD
        if is_occ:
            _mark_obstacle_in_cell(grid, r, c, dx, dy, radius_mm)
        elif st.log_odds <= FREE_THRESHOLD and math.hypot(dx, dy) < grid.cell_size_mm * 0.45:
            for d in ALL_DIRS:
                dr, dc = DIR_DELTA[d]
                if grid.in_bounds(r + dr, c + dc) and not (grid.walls[r][c] & d):
                    grid.known[r][c] |= d


def _mark_obstacle_in_cell(
    grid: MazeGrid, r: int, c: int, dx: float, dy: float, radius_mm: float
) -> None:
    """Classify an occupied point near edge vs core inside maze cell (r, c)."""
    assert grid.cell_size_mm > 0.0 and radius_mm >= 0.0
    assert grid.in_bounds(r, c)
    edge = grid.cell_size_mm * 0.5 - max(radius_mm * 0.5, 15.0)
    if abs(dx) < edge and abs(dy) < edge:
        grid.blocked[r][c] = True
        return
    # Exclude corner zones (35 mm on 180 mm cell) so one wall tip never spawns side stubs.
    tangential_limit = grid.cell_size_mm * 0.5 - (35.0 / 180.0) * grid.cell_size_mm
    if dy >= edge and abs(dx) <= tangential_limit:
        grid.set_wall(r, c, NORTH, True)
    if dy <= -edge and abs(dx) <= tangential_limit:
        grid.set_wall(r, c, SOUTH, True)
    if dx >= edge and abs(dy) <= tangential_limit:
        grid.set_wall(r, c, EAST, True)
    if dx <= -edge and abs(dy) <= tangential_limit:
        grid.set_wall(r, c, WEST, True)


def _ascii_top_row(grid: MazeGrid, r: int) -> str:
    """Render north wall boundary row for ASCII display."""
    out = []
    for c in range(grid.cols):
        out.append("+" + ("---" if (grid.walls[r][c] & NORTH) else "   "))
    out.append("+")
    return "".join(out)


def _ascii_mid_row(
    grid: MazeGrid, r: int, path_set: set[Tuple[int, int]], start: Optional[Tuple[int, int]]
) -> str:
    """Render interior cell row with west walls, path markers, and blocked blocks."""
    out = []
    for c in range(grid.cols):
        w = "|" if (grid.walls[r][c] & WEST) else " "
        mark = " # " if grid.blocked[r][c] else (" S " if start == (r, c) else (" * " if (r, c) in path_set else "   "))
        out.append(w + mark)
    out.append("|" if (grid.walls[r][grid.cols - 1] & EAST) else " ")
    return "".join(out)


def _ascii_bottom_row(grid: MazeGrid) -> str:
    """Render southern boundary row."""
    return "".join("+" + ("---" if (grid.walls[0][c] & SOUTH) else "   ") for c in range(grid.cols)) + "+"


class FloodFillPlanner:
    """Classic UK MarsBot micromouse flood-fill shortest-path planner."""

    def __init__(self, grid: MazeGrid) -> None:
        self.grid = grid

    def compute_distances(
        self, goals: Sequence[Tuple[int, int]], known_only: bool = False
    ) -> List[List[int]]:
        """Run bounded BFS from goal cell(s) returning Manhattan step potential."""
        assert goals, "at least one goal cell required"
        dist = [[INF_DIST] * self.grid.cols for _ in range(self.grid.rows)]
        q: deque[Tuple[int, int]] = deque()
        for gr, gc in goals:
            if self.grid.in_bounds(gr, gc) and not self.grid.blocked[gr][gc]:
                dist[gr][gc] = 0
                q.append((gr, gc))
        steps = 0
        while q and steps < MAX_MAZE_CELLS:
            steps += 1
            r, c = q.popleft()
            next_d = dist[r][c] + 1
            for d in ALL_DIRS:
                if self.grid.has_wall(r, c, d, known_only=known_only):
                    continue
                dr, dc = DIR_DELTA[d]
                nr, nc = r + dr, c + dc
                if next_d < dist[nr][nc]:
                    dist[nr][nc] = next_d
                    q.append((nr, nc))
        return dist

    def next_direction(
        self,
        curr: Tuple[int, int],
        goals: Sequence[Tuple[int, int]],
        heading: int = EAST,
        dist: Optional[List[List[int]]] = None,
        known_only: bool = False,
    ) -> Optional[int]:
        """Select lowest-potential open neighbor, preferring straight travel on ties."""
        dmap = dist or self.compute_distances(goals, known_only=known_only)
        r, c = curr
        if not self.grid.in_bounds(r, c) or dmap[r][c] == 0:
            return None
        order = _preferred_dir_order(heading)
        best_dir, best_val = None, dmap[r][c]
        for d in order:
            if self.grid.has_wall(r, c, d, known_only=known_only):
                continue
            dr, dc = DIR_DELTA[d]
            val = dmap[r + dr][c + dc]
            if val < best_val:
                best_val = val
                best_dir = d
        return best_dir

    def plan_path(
        self,
        start: Tuple[int, int],
        goals: Sequence[Tuple[int, int]],
        initial_heading: int = EAST,
        known_only: bool = False,
    ) -> List[Tuple[int, int]]:
        """Extract discrete cell path from start to nearest reachable goal."""
        dmap = self.compute_distances(goals, known_only=known_only)
        r, c = start
        if not self.grid.in_bounds(r, c) or dmap[r][c] >= INF_DIST:
            return []
        path = [(r, c)]
        heading = initial_heading
        limit = self.grid.rows * self.grid.cols
        for _ in range(limit):
            if dmap[r][c] == 0:
                return path
            nd = self.next_direction((r, c), goals, heading=heading, dist=dmap, known_only=known_only)
            if nd is None:
                break
            dr, dc = DIR_DELTA[nd]
            r, c = r + dr, c + dc
            heading = nd
            path.append((r, c))
        return path


def _preferred_dir_order(heading: int) -> Tuple[int, ...]:
    """Order candidate directions straight -> right -> left -> reverse."""
    if heading not in OPPOSITE_DIR:
        return ALL_DIRS
    return (heading, TURN_RIGHT_OF[heading], TURN_LEFT_OF[heading], OPPOSITE_DIR[heading])


class RightWallFollower:
    """Cornell ECE 3400 zero-map right-wall-following + heading-trim controller."""

    def __init__(self, cell_step_mm: float = 140.0, wall_thresh_mm: float = 130.0) -> None:
        assert cell_step_mm > 0.0 and wall_thresh_mm > 0.0
        self.cell_step_mm = float(cell_step_mm)
        self.wall_thresh_mm = float(wall_thresh_mm)

    def decide_step(
        self, front_mm: float, right_mm: float, cliff_detected: bool
    ) -> WallFollowDecision:
        """Choose next discrete move from ToF right/front clearance + cliff safety."""
        if cliff_detected:
            return WallFollowDecision("ABORT_CLIFF", 0.0, 0.0)
        if right_mm > self.wall_thresh_mm:
            return WallFollowDecision("TURN_RIGHT_ADVANCE", -90.0, self.cell_step_mm)
        if front_mm > self.wall_thresh_mm:
            return WallFollowDecision("ADVANCE", 0.0, self.cell_step_mm)
        return WallFollowDecision("TURN_LEFT", 90.0, 0.0)


def execute_wall_follow_run(
    robot: Any,
    atlas: NavMapAtlas,
    submap_id: str,
    max_steps: int = 8,
    step_mm: float = 120.0,
) -> Dict[str, Any]:
    """Execute bounded live Cornell right-wall-follow steps while logging NavMapAtlas."""
    degrees, distance_mm, speed_mmps = _sdk_units()

    follower = RightWallFollower(cell_step_mm=step_mm)
    submap = atlas.ensure_submap(submap_id, int(getattr(robot.pose, "origin_id", 0)))
    trace: List[Dict[str, Any]] = []
    for idx in range(max(1, min(max_steps, 64))):
        ok, reason, tel = check_safety(robot, allow_on_charger=False)
        if not ok:
            return {"status": "halted", "reason": reason, "steps": idx, "trace": trace}
        front_mm = float(tel["prox_mm"])
        try:
            robot.behavior.turn_in_place(degrees(-90.0))
            ok2, reason2, rtel = check_safety(robot, allow_on_charger=False)
            right_mm = float(rtel["prox_mm"])
            dec = follower.decide_step(front_mm, right_mm, not ok2)
            if dec.action == "ABORT_CLIFF":
                return {"status": "halted", "reason": reason2, "steps": idx, "trace": trace}
            if dec.action == "TURN_RIGHT_ADVANCE":
                robot.behavior.drive_straight(distance_mm(step_mm), speed_mmps(60.0))
            else:
                robot.behavior.turn_in_place(degrees(90.0))
                if dec.action == "ADVANCE":
                    robot.behavior.drive_straight(distance_mm(step_mm), speed_mmps(60.0))
                elif dec.action == "TURN_LEFT":
                    robot.behavior.turn_in_place(degrees(90.0))
        except CliffHazard as exc:
            reason = "cliff_lockout" if exc.refused else ("cliff" if exc.hazard in ("cliff_detected", "control_lost") else exc.hazard)
            return {"status": "halted", "reason": reason, "hazard": exc.hazard, "steps": idx, "trace": trace}
        _fuse_live_step(robot, atlas, submap, submap_id)
        trace.append({"step": idx, "action": dec.action, "front_mm": front_mm, "right_mm": right_mm})
    return {"status": "completed", "reason": "max_steps", "steps": len(trace), "trace": trace}


def _turn_to_world_theta(robot: Any, target_rad: float) -> None:
    """Rotate Vector in place to align heading with canonical target_rad."""
    assert robot is not None and math.isfinite(target_rad)
    turn_to_canonical_theta(robot, target_rad, None)


def execute_speed_run(
    robot: Any,
    atlas: NavMapAtlas,
    submap_id: str,
    goal_cell: Optional[Tuple[int, int]] = None,
    cell_size_mm: float = 180.0,
    marker_map: Optional[Dict[str, MazeMarkerAnchor]] = None,
    grid: Optional[MazeGrid] = None,
) -> Dict[str, Any]:
    """Execute shortest known-only collinear-coalesced speed run to goal_cell."""
    assert robot is not None and cell_size_mm > 0.0
    g = grid or MazeGrid.from_atlas(atlas, cell_size_mm=cell_size_mm)
    planner = FloodFillPlanner(g)
    return execute_speed_run_segments(robot, atlas, g, planner, submap_id, goal_cell=goal_cell, cell_size_mm=cell_size_mm, marker_map=marker_map)


def execute_floodfill_run(
    robot: Any,
    atlas: NavMapAtlas,
    submap_id: str,
    goal_cell: Optional[Tuple[int, int]] = None,
    max_steps: int = 12,
    cell_size_mm: float = 180.0,
    marker_map: Optional[Dict[str, MazeMarkerAnchor]] = None,
    peek_unknown: bool = False,
    rehydrate_before_solve: bool = False,
    two_pass: bool = False,
) -> Dict[str, Any]:
    """Run closed-loop flood-fill exploration with optional rehydration, ToF peek, and two-pass speed-run."""
    assert robot is not None and cell_size_mm > 0.0
    if rehydrate_before_solve and submap_id in atlas.transforms_to_canonical:
        rehydrate_onboard_obstacles(robot, atlas, submap_id, max_boxes=48)
    grid = MazeGrid.from_atlas(atlas, cell_size_mm=cell_size_mm)
    planner = FloodFillPlanner(grid)
    loc0 = fuse_and_localize_step(robot, atlas, submap_id, marker_map=marker_map)
    start_cell = grid.world_to_cell(loc0["cpose"].x_mm, loc0["cpose"].y_mm)
    goal = goal_cell or (grid.rows - 1, grid.cols - 1)
    res1 = run_floodfill_loop(robot, atlas, grid, planner, submap_id, goal, max_steps, cell_size_mm, marker_map, peek_unknown)
    if two_pass and res1.get("status") == "reached":
        res2 = execute_speed_run(robot, atlas, submap_id, goal_cell=start_cell, cell_size_mm=cell_size_mm, marker_map=marker_map, grid=grid)
        return {**res1, "two_pass": True, "pass2_speed_run": res2}
    return res1


def _fuse_live_step(robot: Any, atlas: NavMapAtlas, submap: Any, submap_id: str) -> None:
    """Fold NavMapGrid + ToF ray + world anchors after a live step."""
    ok, _reason, tel = check_safety(robot, allow_on_charger=True)
    pose = pose2d_from_sdk_pose(robot.pose)
    submap.integrate_tof_ray(pose, float(tel["prox_mm"]), bool(tel["prox_found"]), bool(tel["prox_lift_in_fov"]))
    nm = poll_latest_nav_map(robot, max_wait_s=0.5)
    if nm is not None:
        submap.integrate_leaves(extract_quadtree_leaves(nm))
    for aid, apose in observe_world_anchors(robot).items():
        atlas.record_anchor(submap_id, aid, apose)
    assert isinstance(ok, bool)
