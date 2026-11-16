# vector-slam-maze

Persistent multi-session SLAM atlas, fiducial `SE(2)` frame stitching, ROS Cartographer `grid.pgm` / `grid.yaml` export, micromouse flood-fill + wall-following maze solver, and Phase 2 classical visual pose-graph SLAM + ROS 2 / TUM / EuRoC sensor bridge for Anki/DDL Vector.

## Capabilities (`v0.4.1`)

- **Bounded Translation & Movement Directives (`v0.4.1`, `movement_directives.py`)**:
  - Stdlib-only module (`numpy`/`PIL` moved to lazy PEP 562 imports in `__init__.py`) exposing Ariadne-compatible `MotionResult`, `bounded_drive`, `bounded_turn`, `plan_directive`, `plan_directive_batch`, `execute_bounded_directive`, and `execute_directive_batch`.
  - Fail-closed payload/pose guards (`v0.4.1`): monotonic caller `strict` floor + boolean validation, caller `standoff_mm` floor (payloads may raise standoff but never lower it), strict per-verb key whitelisting (`missing_distance` / `unknown_keys:*` instead of defaulting missing `dx_m` to `0.1`), and `pose_unavailable` refusal instead of fabricating displacement/turn from commanded deltas.
  - Decomposes body-frame 2D `(dx_mm, dy_mm)` `translate` into non-holonomic `turn -> chunked drive -> optional counter-turn` (with `<= 150 mm` pure-reverse shortcut when `|dy_mm| <= 5 mm`), enforces asymmetric caps (`+600 mm` / `-150 mm` / `±360 deg`, `-22..45 deg` head, `0..1` / `32..92 mm` lift, `16`-step / `1200 mm` / `720 deg` / `30 s` batch budgets), chunks forward drives `<= 180 mm` and reverse `<= 50 mm`, fails closed on missing ToF (`prox_valid == False`) or lift-occluded ToF (`prox_lift_in_fov == True` unless `allow_occluded_tof=True`), and halts on ToF standoff (`80 mm`), cliff, pickup, falling, wheel slip, or `origin_id` jump.
  - Adds CLI `translate` subcommand and extended `act` verbs (`move`, `rotate`, `turn`, `translate`, `head`, `lift`, `undock`, `dock`, `halt`, `directives`) with `--dry-run` (offline, zero SDK/network) and `--strict` bounds rejection.
- **Persistent `NavMapAtlas` (`v0.1.0`)**:
  - Extracts Vector's quad-tree `NavMapGrid` leaves iteratively (non-recursive P10 traversal) down to native `8 mm` cells and fuses log-odds occupancy, cliff hazards (`NavContent.CLIFF`), and front ToF raycasts into sparse submaps.
  - Registers `origin_id`-reset sessions across disconnects/reboots via shared landmarks (`Charger`, `LightCube`, and all 16 firmware `CustomObjectMarkers`) using closed-form `SE(2)` / Procrustes frame alignment, keeping unanchored submaps isolated until a shared landmark links them.
  - Exports atomic ROS `map_server` `grid.pgm` (`254` free / `0` occupied / `205` unknown) and `grid.yaml` compatible with Yamauchi frontier detection (`_cartographer_frontiers.py`).
  - Optionally rehydrates persisted obstacle boxes back into Vector's onboard world frame via `create_custom_fixed_object`.
- **Phase 2 Classical Visual Pose-Graph SLAM (`visual_slam.py`)** (zero vision/VLM models):
  - Computes 288-D binomial-smoothed zero-mean intensity + gradient appearance signatures and extracts 8x6 bucketed Shi-Tomasi 2-peak NMS keypoints with 9x9 ZNCC descriptors.
  - Estimates frame-to-frame visual yaw delta via mutual ZNCC (`>= 6` inliers) fused with wheel/IMU odometry (`SE(2)`), detects appearance + ZNCC loop closures, relaxes the pose graph with bounded Gauss-Seidel optimization, relocalizes saved keyframes across sessions (`relocalize_frame`), and syncs visual-odom + ToF submaps back into `NavMapAtlas`.
  - Enforces passive preflight illumination + calm-power-mode gating (`MIN_USABLE_LUMA = 22.0`, `MIN_USABLE_CONTRAST = 7.0`) before active undock (`visual-slam --via-vector --undock-if-needed` exits `4` when dark or paused in calm power mode).
- **ROS 2 / TUM / EuRoC Sensor Bridge (`sensor_bridge.py`) & Camera Characterization (`camera_characterize.py`)**:
  - Auto-scales Vector `CameraConfig` intrinsics across `320x180`, `640x360`, and `1280x720`, synthesizes virtual `/vector/scan` 2D `LaserScan` from front ToF + `NavMapAtlas` walls, and publishes `/vector/camera/{image_raw,camera_info}`, `/vector/imu`, `/vector/odom`, `/vector/scan` via `VectorRos2Bridge` (offline bus or optional `rclpy`).
  - Exports TUM (`rgb.txt`, `groundtruth_odom.txt`), EuRoC (`imu.csv`, `odom.csv`), calibrated `orb_slam3_vector.yaml`, `rtabmap_ros2_launch.py`, and `dataset_manifest.json` for desktop `RTAB-Map` / `ORB-SLAM3` or `visual-slam --replay-dataset`.
  - Measures stream FPS/jitter/drops, exposure/gain saturation, 1D FFT phase-correlated rolling-shutter readout skew, and visual-vs-odom drift (`evaluate_visual_vs_odom_drift`).
- **Three Maze-Solving Modes (`maze_solver.py`, `landmarks.py`)**:
  - **Micromouse Flood-Fill (`FloodFillPlanner`)**: UK MarsBot Manhattan potential BFS with straight-heading tie-breaking, online wall discovery, and re-flooding.
  - **Landmark Localization (`landmarks.py`)**: Defines `CustomObjectMarkers` (`Circles2..5`, `Diamonds2..5`, `Hexagons2..5`, `Triangles2..5`) in firmware to pin maze coordinates (`Vector-and-the-Minotaur/Labyrinth` approach).
  - **Right-Wall Follower (`RightWallFollower`)**: Cornell ECE 3400 zero-map ToF + cliff-safe controller.
- **Python 3.11 SDK Hardening (`sdk_session.py`)**:
  - Enforces `--via-vector` consent + exclusive takeover lockfile for active motion, non-blocking `_latest_nav_map` and `_latest_image` polling (avoiding `@util.block_while_none()` stalls/exceptions), direct `_nav_map_feed_task` + `_camera_feed_task` cancellation (avoiding the 10s `close_camera_feed()` spin), and `os._exit` on `entrypoint()`.

## Quick Start

```bash
# Passive state + NavMapGrid check (zero robot motion)
python3 -m vector_slam_maze --serial 0dd1cafe probe

# Passive camera calibration, exposure bounds, illumination & dataset export
python3 -m vector_slam_maze --serial 0dd1cafe characterize-camera --duration-s 5.0 --high-res-probe --export-dataset /tmp/vector-ds

# Offline ROS 2 environment check & dataset bridge
python3 -m vector_slam_maze ros2-bridge --check

# Offline visual-inertial pose-graph replay + cross-session relocalization
python3 -m vector_slam_maze --atlas /tmp/vslam-atlas.json --export-dir /tmp/vslam-ros --session-id replay-1 visual-slam --replay-dataset /tmp/vector-ds
python3 -m vector_slam_maze --atlas /tmp/vslam-atlas.json visual-relocalize --replay-frame /tmp/vector-ds/frames/frame_000001.png

# Persist passive NavMap snapshot + export ROS grid.pgm / grid.yaml
python3 -m vector_slam_maze --serial 0dd1cafe --assume-docked-canonical snapshot

# Authorized 360-degree visual + ToF + NavMap scan (refuses rc=4 if dark/calm unless --force-dark)
python3 -m vector_slam_maze --serial 0dd1cafe --assume-docked-canonical visual-slam --via-vector --undock-if-needed --return-charger

# Plan flood-fill maze path on persisted atlas
python3 -m vector_slam_maze --atlas /home/pi/maps/demo-persistent-atlas.json plan-maze

# Offline bounded 2D translation plan (zero robot motion / no numpy or SDK required)
python3 -m vector_slam_maze translate --dx-mm 250 --dy-mm -100 --preserve-heading --dry-run

# Live authorized bounded 2D translation (requires --via-vector consent)
python3 -m vector_slam_maze --serial 0dd1cafe act translate --args-json '{"dx_mm":250,"dy_mm":-100,"preserve_heading":true}' --via-vector

# Run unit tests (68 offline tests)
python3 -m unittest discover -s tests -v
```

## Live Hardware Verification (`DEMO`, `0dd1cafe`, `2026-11-13`)

- **v1 active 360° ToF/NavMap scan**: Undocked (`3.944V`), 8x45° steps (`145` leaves, `172` leaf updates, `5` ToF ray updates), `33 x 43` `0.02 m/px` ROS grid (`74` free / `29` occ / `103` known cells, `2` Cartographer frontiers), Session 2 multi-session merge (`registered_submaps: 2`), `create_custom_fixed_object` null result on `NavMapGrid` leaves (`223 == 223`), and clean `act dock`.
- **v0.2.0 passive dark/calm verification (`escapepod`)**: Live docked `characterize-camera` confirmed calibrated `640x360` `CameraConfig` (`fx=361.1599`, `fy=356.2424`, `cx=310.661`, `cy=183.827`, `fov_x=83.084°`, `fov_y=53.613°`) and hardware exposure ceilings (`1..66 ms`, gain `0.25..3.8`), reported `camera_paused_calm_power_mode` (`is_on_charger: true`, `is_in_calm_power_mode: true`, `exposure_ms: 0`, `gain: 0.0`, `4.069V`), gated `visual-slam --via-vector --undock-if-needed` to `rc=4` (`dark_room_refuse_undock`) with zero motion, and passed offline 14-frame calibrated replay + relocalization (`inliers=80`, `median_zncc=1.0`, `orb_slam3_vector.yaml`) on `escapepod`.

## Known Limitations (audit, `2026-11-13`)

- **No real camera frame has passed through `VisualPoseGraphSLAM`.** Every visual test and the `escapepod` replay use frames from the synthetic renderer in `tests/`. The live `DEMO` check covered intrinsics readout and the dark/calm gate only, because the camera was paused. The yaw sign convention is therefore checked against the renderer, not against the robot.
- **Vision measures heading only.** Position comes from wheel odometry, so monocular scale drift is not corrected. The yaw estimate assumes rotation in place; parallax while driving biases it. Loop closures constrain heading and never translation.
- **`visual-relocalize` is place recognition, not metric relocalization.** It returns the matched keyframe's position with an adjusted heading.
- **The ORB-SLAM3 and RTAB-Map exports are unverified.** Neither tool has loaded `orb_slam3_vector.yaml` or `rtabmap_ros2_launch.py`. Third-party documentation of ORB-SLAM3 settings lists `File.version`, `IMU.T_b_c1`, `ORBextractor.iniThFAST` and `ORBextractor.minThFAST`, none of which the YAML has, and shows a 200 Hz IMU where this file declares 30 Hz. RTAB-Map's core modes are RGB-D, stereo and lidar; monocular-only use is at best experimental. Check both against the official examples before building on them.
- **`characterize-camera` exits `0` when no frames arrive.** Read `ok` in its JSON.
- **Map-free `bounded_turn` does not check lateral swept clearance.** `bounded_turn` checks cliff/pickup/falling/charger safety before and after rotation, and `execute_bounded_directive("translate", ...)` re-checks front ToF on the post-turn driving heading, but single-point ToF cannot verify side-wall clearance during an in-place turn inside tight corridors without an external map.
- **`maze_closed_loop.py` and `navmap_atlas.py` remain unchanged from `v0.3.3` and do not import `movement_directives.py`.** GPTPCS-Astra findings `A`–`J` (stand-off, fail-open ranging, chunk exhaustion, heading ToF, swept clearance, relock quorum, lever arm, stale-cell `curr`) remain present in the closed-loop maze driver until `P-BACKLOG.md` `[0e8a96e]` lands; live `solve-maze` runs must wait for `[0e8a96e]`.
