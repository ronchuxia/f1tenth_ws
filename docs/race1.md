# Race 1: Reactive Methods

## Description

Race 1 uses a **disparity extender** controller. 

Disparity extender looks for large jumps between adjacent LiDAR range values. Those jumps mark obstacle edges, so the controller extends a safety bubble around the closer side of each jump to avoid steering too close to obstacles.

After adjusting the LiDAR ranges by extending the safety bubble, the controller chooses the farthest reachable point as the target steering direction.

Steering is rejected when the controller tries to turn toward a side with a LiDAR range closer than 0.2m.

Speed is adjusted based on the steering angle.

## Running 

Run disparity extender with:

```shell
bash scripts/race1.bash
```

## Parameters

Race 1 parameters are hardcoded in `src/race1/race1/disparity_extender.py`.

- `p`: `0.5`. Steering gain. The steering angle is `p` times the angle of the furthest LiDAR point.
- `bubble_radius`: `0.3 m`. Points inside this radius around a disparity are extended to form a safety bubble.
- `disparity_threshold`: `0.5 m`. Adjacent LiDAR ranges with a jump larger than this are treated as a disparity.
- Velocity profile:
  - `3.8 m/s` when `abs(steering_angle) < 10 deg` and target range is greater than `0.5 m`.
  - `2.8 m/s` when `abs(steering_angle) < 20 deg` and target range is greater than `0.5 m`.
  - `2.3 m/s` otherwise.
