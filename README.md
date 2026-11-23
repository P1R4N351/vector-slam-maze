# vector-slam-maze

Persistent multi-session SLAM atlas, fiducial `SE(2)` frame stitching, ROS Cartographer `grid.pgm` / `grid.yaml` export, micromouse flood-fill + wall-following maze solver, and Phase 2 classical visual pose-graph SLAM + ROS 2 / TUM / EuRoC sensor bridge for Anki/DDL Vector.

## Capabilities (`v0.5.0`)

- **Persistent Basal `MemoryMap` & Origin Rehydration (`v0.5.0`, `basal_persist.py`, `navmap_atlas.py`, `sdk_session.py`, `cli.py`)**:
  - Adds passive `basal-daemon [--pod-url URL] [--ip HOST] [--sqlite PATH] [--export-navmap-json PATH] [--ensure-pod-timeouts] [--rehydrate-on-new-origin]` subscribing to `NavMapGrid` broadcasts and `RobotState` heartbeats without blocking idle feeds (eliminating 45 s `TimeoutError` crashes when stationary/docked).
  - Writes seated `10 mm` occupancy SQLite stores (`revisions`, `seated`) and atomic `seated_robot` `NavMap` JSON exports alongside incremental `NavMapAtlas` (`INTERESTING_EDGE` delta-baselined without spurious log-odds creep) and ROS `grid.pgm` / `grid.yaml` updates.
  - Rehydrates canonical obstacles across `origin_id` resets using `FixedCustomObject` with robot (`110 mm` radius) and charger-corridor (`-40..220 x -90..90 mm` 2D SAT) clearance exclusions, gating off during `calm_power_mode` or pod sleep-mute mode (`GET /api-sdk/cryo_mute`), and optionally asserting pod `MemoryMap` timeout persistence (`POST /api-sdk/basal_map?enable=true`).
  - Upgrades `rehydrate` CLI and `sdk_session.poll_fresh_nav_map(..., after_map=prev)` to wait specifically for a post-injection `NavMapGrid` broadcast and report `content == 3` (`OBSTACLE_CUBE`) leaf counts directly.
- **4-Mechanism Physical Characterization & Teardown Hardening (`v0.4.3`, `mechanism_characterize.py`, `cli_phase2.py`, `sdk_session.py`)**:
  - Adds `characterize-mechanisms --via-vector [--docked-only | --include-wheels --confirm-safe-floor] [--allow-occluded-tof] [--ip HOST] [--out FILE]` (exiting `0` when `evaluate_physical_realism` passes or `rc=6` on safety/realism failure) to empirically measure transport latency, `10-90%` rise time (`rise_10_90_ms`), effective time constant (`tau_eff_ms`), overshoot (`%`), steady-state error, unpowered gravity-hold drift, high-speed tread braking distance, and closed-loop `80 mm` drive / `60°` turn accuracy across `head`, `lift`, `left_wheel`, and `right_wheel`.
  - Hardens `safe_disconnect` / `_release_active_control` / `_force_close_transport` in `sdk_session.py` to await `release_control()` futures (`2.0 s` timeout), catch `_sdk_exception_types()`, and force-close `events` / `conn` transports if `robot.disconnect()` aborts mid-teardown after cliff/control loss.
- **Active Cliff Guard & SDK/Maze Safety Hardening (`v0.4.2`, `cliff_guard.py`, `sdk_session.py`, `maze_solver.py`, `maze_closed_loop.py`)**:
  - Active `VectorSDKSession` requests control at `DEFAULT_PRIORITY` (never `OVERRIDE_BEHAVIORS`) so Vector's firmware cliff reflex stays active, wraps `robot.behavior` in `GuardedRobot` (`CliffGuard`), catches mid-motion cliffs/falls/pickups and `wirepod_vector_sdk 0.8.1`'s `CancelledError -> None` + delayed `control_granted_event.clear()` race (`_await_control_clear`), re-acquires control at `DEFAULT` before subsequent commands, and refuses further forward drives (`cliff_lockout`) until the robot turns `>= 45°` or backs up `>= 30 mm`.
  - Maze drivers (`execute_wall_follow_run`, `run_floodfill_loop`), `cli.py`, and `cli_phase2.py` surface `CliffHazard` as structured JSON halts (`rc=6`, `stopped_reason` `"cliff"` or `"cliff_lockout"`).
  - Hardens `_safe_battery_state` against `VectorException`, `_ensure_control` against `Future` returns, `_reopen_camera_stream` silence revival (cancels `_camera_feed_task` before `EnableImageStreaming(True)`), and exports 3-line TUM `rgb.txt` headers matching `ORB-SLAM3` `mono_tum.cc`.
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

# Passive multi-session basal MemoryMap daemon (SQLite + simulator JSON + atlas + pod timeouts)
python3 -m vector_slam_maze --serial 00000000 --atlas /tmp/basal-atlas.json --export-dir /tmp/basal-ros basal-daemon --pod-url http://127.0.0.1:8080 --sqlite /tmp/native.sqlite --export-navmap-json /tmp/shelf-navmap.json --ensure-pod-timeouts --rehydrate-on-new-origin

# Authorized 360-degree visual + ToF + NavMap scan (refuses rc=4 if dark/calm unless --force-dark)
python3 -m vector_slam_maze --serial 0dd1cafe --assume-docked-canonical visual-slam --via-vector --undock-if-needed --return-charger

# Plan flood-fill maze path on persisted atlas
python3 -m vector_slam_maze --atlas /home/pi/maps/demo-persistent-atlas.json plan-maze

# Offline bounded 2D translation plan (zero robot motion / no numpy or SDK required)
python3 -m vector_slam_maze translate --dx-mm 250 --dy-mm -100 --preserve-heading --dry-run

# Live authorized bounded 2D translation (requires --via-vector consent)
python3 -m vector_slam_maze --serial 0dd1cafe act translate --args-json '{"dx_mm":250,"dy_mm":-100,"preserve_heading":true}' --via-vector

# Docked-safe head & lift electromechanical characterization (--via-vector consent)
python3 -m vector_slam_maze --serial 0dd1cafe characterize-mechanisms --via-vector --docked-only

# Full 4-mechanism characterization on safe flat floor
python3 -m vector_slam_maze --serial 0dd1cafe characterize-mechanisms --via-vector --include-wheels --confirm-safe-floor --out /tmp/mechanisms.json

# Run unit tests (112 offline tests; 2 optional SDK/rosbags skipped when absent)
python3 -m unittest discover -s tests -v
```

## Live Hardware & Headless SLAM Verification (`DEMO` / headless container, `2026-11-13`–`2026-11-16`)

- **v1 active 360° ToF/NavMap scan (`DEMO`, `2026-11-13`)**: Undocked (`3.944V`), 8x45° steps (`145` leaves, `172` leaf updates, `5` ToF ray updates), `33 x 43` `0.02 m/px` ROS grid (`74` free / `29` occ / `103` known cells, `2` Cartographer frontiers), Session 2 multi-session merge (`registered_submaps: 2`), `create_custom_fixed_object` null result on `NavMapGrid` leaves (`223 == 223`), and clean `act dock`.
- **v0.2.0 passive dark/calm verification (wire-pod host, `2026-11-13`)**: Live docked `characterize-camera` confirmed calibrated `640x360` `CameraConfig` (`fx=361.1599`, `fy=356.2424`, `cx=310.661`, `cy=183.827`, `fov_x=83.084°`, `fov_y=53.613°`) and hardware exposure ceilings (`1..66 ms`, gain `0.25..3.8`), reported `camera_paused_calm_power_mode` (`is_on_charger: true`, `is_in_calm_power_mode: true`, `exposure_ms: 0`, `gain: 0.0`, `4.069V`), gated `visual-slam --via-vector --undock-if-needed` to `rc=4` (`dark_room_refuse_undock`) with zero motion, and passed offline 14-frame calibrated replay + relocalization (`inliers=80`, `median_zncc=1.0`) on the wire-pod host.
- **`v0.4.2` simulator cliff proof (`vectorsim` + real `wirepod_vector_sdk 0.8.1`, `2026-11-16`)**: `slam_maze_cliff_sim.py` passed both `VectorSDKSession` raw drive and `bounded_drive`/`bounded_turn` scenarios (`"pass": true`, `falls=0`, `cliff_events=1`, `reflexes=1`, `control_lost=1`, lockout enforced across `-10 mm` reverse and lifted by `-60 mm` + `180°` escape).
- **`v0.4.3` 4-mechanism dynamics benchmark (`vectorsim` + `mechanism_bench_sim.py`, `2026-11-18`)**: Passed (`"pass": true`, `floor_4mech: true` with `Falls: 0`, `docked_2mech: true`, `docked_wheel_refuse: true` `rc=6` `on_charger`, `ideal_rejected: true` rejecting `-ideal-motors` with `rc=6` and 8 realism failures). Executed floor metrics: `head` `step_up` `rise_10_90_ms=319.31`, `tau_eff_ms=234.92`, `overshoot_pct=0.59`, `steady_state_error=0.325°`, `hold_drift_deg=0.0`; `lift` `step_up` (`83 mm`) `rise_10_90_ms=250.72`, `tau_eff_ms=181.72`, `overshoot_pct=4.63`, `steady_state_error=0.96 mm`, `hold_drift_mm=0.0`; `left_wheel` `rise_10_90_ms=49.22`, `tau_eff_ms=27.39`, `overshoot_pct=14.7`, `steady_state_error=0.724 mm/s`, `braking_dist_mm=8.128`; `right_wheel` `rise_10_90_ms=47.56`, `tau_eff_ms=26.2`, `overshoot_pct=14.7`, `steady_state_error=0.724 mm/s`, `braking_dist_mm=8.331`; `chassis` `drive_measured_mm=79.21` (`-0.79 mm`), `turn_measured_deg=58.96` (`-1.04°`). Live hardware baseline capture (`--out`) on physical `DEMO` (`192.0.2.130`) remains pending (`DEMO` offline during `2026-11-18` runs).
- **`v0.4.2` headless `ORB-SLAM3` + `evo` + `VisualPoseGraphSLAM` (headless Linux container, `2026-11-16`)**:
  - **Real `DEMO` lit docked `demo_wake_probe` (`18` `640x360` frames, `4 Hz`)**: `VisualPoseGraphSLAM` created `1` keyframe (`mean_yaw_diff_deg=0.036`, `final_yaw_diff_deg=0.005`) and `visual-relocalize` matched frame `17` (`matched_kf_id=0`, `sig_cos=0.9598`, `inliers=19`, `median_zncc=0.9465`, `u_span_px=555.0`); `ORB-SLAM3` `mono_tum` parsed `orb_slam3_vector.yaml` cleanly (`0` config errors, `median=0.0247 s/frame`) and emitted `0` keyframes on the zero-baseline stationary capture.
  - **Synthetic forward `synth_parallax_tum` (`75` frames, `10 Hz`, `2x2` tiles)**: `VisualPoseGraphSLAM` produced `16` keyframes (`evo_ape -a -s` RMSE `0.002382 m`, `evo_rpe -a -s` RMSE `0.000217 m`); `ORB-SLAM3` did not initialize on repeating identical `2x2` tiles / `9 mm/frame` forward expansion.
  - **Calibrated synthetic `synth_orbit_tum` (`100` frames, `10 Hz`, `220` unique 3D patches + lateral S-curve)**: `ORB-SLAM3` `mono_tum` initialized (`New Map created with 302 points`), tracked at `0.0100 s/frame` median, emitted `27` keyframes, and scored `evo_ape -a -s` Sim(3) RMSE **`0.001660 m`** (`max=0.002677 m`); `VisualPoseGraphSLAM` produced `39` keyframes with `33` loop closures (`evo_ape -a -s` RMSE `0.031370 m`, `trans_rmse_mm=69.547`, empirically confirming pure-bearing lateral-parallax heading bias).

## Known Limitations (`2026-11-16`)

- **No moving live `DEMO` camera sequence has passed through visual SLAM yet.** Only `18` stationary lit docked `DEMO` frames (`demo_wake_probe`) and synthetic `DEMO`-calibrated sequences have been processed; verifying pixel-bearing `+CCW` yaw sign and live 2.4 GHz Wi-Fi rolling-shutter skew on moving real `DEMO` frames remains open.
- **Vision measures heading only.** Position comes from wheel odometry, so monocular scale drift is not corrected. The yaw estimate assumes rotation in place; parallax while driving biases it (measured `69.547 mm` drift / `0.031370 m` ATE on `synth_orbit_tum`). Loop closures constrain heading and never translation.
- **`visual-relocalize` is place recognition, not metric relocalization.** It returns the matched keyframe's position with an adjusted heading.
- **`rtabmap_ros2_launch.py` remains unverified in an upstream `RTAB-Map` container; `ORB-SLAM3` `orb_slam3_vector.yaml` is verified in `mono_tum`.** Upstream `ORB-SLAM3 v1.0` `mono_tum` parsed `orb_slam3_vector.yaml` cleanly (`File.version: "1.0"`, `ORBextractor.iniThFAST: 20`, `minThFAST: 7`, `T_b_c1`); `IMU.Frequency` remains `30.0 Hz` vs `200 Hz` expected by `ORB-SLAM3` Mono-Inertial. `RTAB-Map`'s core modes are RGB-D, stereo and lidar; monocular-only use is experimental.
- **`characterize-camera` exits `0` when no frames arrive.** Read `ok` in its JSON.
- **Map-free `bounded_turn` does not check lateral swept clearance.** `bounded_turn` checks cliff/pickup/falling/charger safety before and after rotation, and `execute_bounded_directive("translate", ...)` re-checks front ToF on the post-turn driving heading, but single-point ToF cannot verify side-wall clearance during an in-place turn inside tight corridors without an external map.
- **`maze_closed_loop.py` and `navmap_atlas.py` remain unchanged from `v0.3.3` and do not import `movement_directives.py`.** Audit findings `A`–`J` (stand-off, fail-open ranging, chunk exhaustion, heading ToF, swept clearance, relock quorum, lever arm, stale-cell `curr`) remain present in the closed-loop maze driver until it is moved onto the bounded primitives; live `solve-maze` runs must wait for that change.
