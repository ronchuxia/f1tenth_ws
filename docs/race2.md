# Race 2: Map-Based Methods

## Description

Race 2 uses a **pure pursuit** controller to follow a logged waypoint path.

Pure pursuit finds a target waypoint ahead of the car at the current lookahead distance. It transforms that target into the car frame and computes the steering angle using the kinematic bicycle model.

The lookahead distance is adjusted based on the car speed and the curvature of upcoming waypoints.

Speed is limited by the maximum configured speed and by the steering angle based on vehicle dynamics. Sharper turns produce a lower speed.

## Running

Run pure pursuit with:

```shell
bash scripts/race2.bash
```

## Parameters

Race 2 parameters are passed in `scripts/race2.bash`.

- `waypoints_file`: Path to the waypoint CSV file.
- `sim`: Uses simulation topics and frames when `true`; uses real-car topics and frames when `false`.
- `vis`: Publishes RViz markers when `true`.
- `l_mode`: Lookahead mode. Current value is `curvature_speed`, which adapts lookahead using both speed and upcoming path curvature.
- `l`, `l_min`, `l_max`: Fixed lookahead value and adaptive lookahead bounds.
- `l_k`: Gain for speed-based lookahead scaling.
- `l_eps`: Curvature regularizer that prevents very large lookahead on straight segments.
- `n_ahead`: Number of upcoming waypoints used to estimate curvature.
- `p`: Steering gain applied to the pure pursuit steering command.
- `max_speed`: Upper speed limit.
- `mu`: Tire-road friction estimate used to limit speed in turns.
- `max_steering_angle`: Steering command limit.
- `interpolate`: Only used by the vectorized target-selection function, which is not used by the current controller loop.
- `wheelbase`: Vehicle wheelbase used by the kinematic bicycle model.
- `laser_offset`: Offset from the base link to the laser frame, used on the real car.
- `dead_reckon`: Predicts pose forward to compensate for localization latency when enabled.
