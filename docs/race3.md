# Race 3: Map-Based Methods with Dynamic Obstacle Avoidance

## Description

Race 3 combines waypoint tracking with dynamic obstacle avoidance.

Before the race, [Raceline Optimization](https://github.com/ronchuxia/raceline-optimization.git) is used to generate an optimized raceline and a velocity profile given the map and vehicle dynamics.

During normal driving, a **pure pursuit tracker** is used to follow the optimized raceline. Speed is capped by the velocity profile in the raceline file.

Obstacle avoidance is handled by a **Frenet planner**. When avoidance is triggered, the planner samples several lateral-offset trajectories around the raceline, checks them against the LiDAR scan, and chooses the lowest-cost trajectory that clears obstacles. The selected trajectory is then followed using a pure pursuit tracker with a different set of parameters.

Obstacle avoidance is triggered when a LiDAR range is close to the upcoming waypoints.

## Running

Run Race 3 with:

```shell
bash scripts/race3.bash
```

## Parameters

Race 3 parameters are configured in `src/race3/config/race3.yaml`.

Refer to [Race 2](docs/race2.md) for more details on parameters related to pure pursuit.

- `waypoints_file`: Path to the optimized raceline CSV file.
- `sim`: Uses simulation topics and frames when `true`; uses real-car topics and frames when `false`.
- `vis`, `vis_traker`, `vis_obs`: Enable RViz visualization for the controller, tracker markers, and obstacle markers.
- `wheelbase`, `laser_offset`, `max_steering_angle`: Vehicle geometry and steering limits.
- `scan_self_filter_range`: Removes very close LiDAR points to filter out LiDAR ranges caused by the car body.
- `enable_pose_prediction`: Predicts pose forward to compensate for odometry latency when enabled.
- `max_speed`, `mu`, `p`: Pure pursuit speed limit, friction estimate, and steering gain.
- `l_min`, `l_max`, `l_k`, `l_eps`, `n_ahead`: Adaptive lookahead parameters for the pure pursuit tracker.
- `use_waypoint_speed_profile`: Caps speed using the velocity profile in the raceline file.
- `avoidance_trigger`: Selects when obstacle avoidance can activate. Current value is `waypoint_radius`.
- `trigger_wp_radius`: Distance threshold for marking a waypoint as blocked. If a LiDAR point is within this radius of an upcoming waypoint, the raceline is considered blocked.
- `trigger_wp_lookahead`: Forward distance used when checking for blocked waypoints. Only waypoints in front of the car and within this distance are considered.
- `avoidance_frames`, `reacquire_frames`: Number of consecutive blocked or clear frames required to switch between `following` and `avoiding`.
- `frenet_car_width`: Width used for collision checking. A candidate trajectory is rejected if a LiDAR point is within half this width of the trajectory.
- `frenet_lookahead`: Arc length ahead of the car used to generate avoidance trajectories.
- `frenet_num_offsets`: Number of lateral-offset candidate trajectories sampled around the raceline.
- `frenet_offset_spacing`: Lateral spacing between neighboring candidate trajectories.
- `frenet_num_samples`: Number of points sampled along each candidate trajectory for collision checking.
- `frenet_w_offset`: Cost weight that penalizes moving far away from the raceline.
- `frenet_w_jerk`: Cost weight that penalizes sharp or uncomfortable trajectory shapes.
- `frenet_w_clear`: Cost weight that rewards trajectories with more clearance from LiDAR obstacles.
- `frenet_max_speed`, `frenet_mu`, `frenet_p`, `frenet_pp_lookahead`: Speed, steering, and target-selection parameters used while following the selected Frenet trajectory.
