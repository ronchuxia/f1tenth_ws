# F1TENTH Workspace

ROS 2 workspace for the races and final project of team 9 of 16-663 F1TENTH Autonomous Racing at CMU, 2026 spring. Including reactive method, waypoint following, dynamic obstacle avoidance, waypoint logger, odometry tuner and anti-skid system.

## Contents

- `race1`: reactive method (disparity extender).
- `race2`: map-based method (pure pursuit).
- `race3`: map-based method with dynamic obstacle avoidance (raceline optimization + frenet planner + pure pursuit).
- `waypoint_logger`: waypoint collection from teleop in simulation, RViz clicked points, or particle-filter localization on real car.
- `f1tenth_utils`: shared helper and visualization utilities.
- `odom_tuner`: odometry tuner for turning VESC parameters.
- `maps`: maps used in race2 and race3.
- `waypoints`: waypoints used in race2 and race3.
- `scripts`: bash scripts for launching nodes.

## Installation

[Installation](docs/installation.md)

## Remote Visualization

See [Remote Visualization](docs/remote_visualization.md) for the CycloneDDS and RViz setup used to visualize Jetson ROS 2 topics from a laptop.

## Running

Source the ROS2 installation and this workspace before running nodes:

```shell
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Run the F1TENTH simulator:

```shell
bash scripts/sim.bash
```

Run teleop:

```shell
bash scripts/teleop.bash
```

Run race controllers:

```shell
bash scripts/race1.bash
bash scripts/race2.bash
bash scripts/race3.bash
```

Log waypoints:

```shell
bash scripts/waypoint_logger.bash
```

## Strategies

[Race 1](docs/race1.md)

[Race 2](docs/race2.md)

[Race 3](docs/race3.md)
