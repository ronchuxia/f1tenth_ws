# Installation

## Install ROS2 humble

[Install ROS2 humble](https://docs.ros.org/en/humble/Installation.html)

## Clone the repo

```shell
git clone https://github.com/ronchuxia/f1tenth_ws.git
```

## Install the simulation (for simulation)

```shell
git clone https://github.com/f1tenth/f1tenth_gym
cd f1tenth_gym && pip3 install -e .
```

## Install the simulation ROS2 bridge (for simulation)

1. Clone the simulation ROS2 bridge.

    ```shell
    git clone https://github.com/f1tenth-cmu/f1tenth_gym_ros.git
    ```

2. Modify `map_path` in `f1tenth_gym_ros/config/sim.yaml` to be the actual map path on your machine.

3. Run.
    ```shell
    source /opt/ros/humble/setup.bash
    sudo rosdep init
    rosdep update
    rosdep install -i --from-path src --rosdistro humble -y
    colcon build
    ```

## Install the F1TENTH system (for real car)

[Install the F1TENTH system](https://github.com/f1tenth-cmu/course-resources?tab=readme-ov-file#software-setup-ubuntu-2204-ros2-humble)

## Install the paticle filter (for real car)

[Install the paticle filter](https://docs.google.com/presentation/d/1Hyb1tX576u7adukdh18C8gnibawEBhmE3w5rTCdUSSM/edit?slide=id.p1#slide=id.p1)