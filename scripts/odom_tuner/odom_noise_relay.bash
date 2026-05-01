#!/bin/bash
ros2 run odom_tuner odom_noise_relay --ros-args \
    -p pos_noise_std:=0.01 \
    -p yaw_noise_std:=0.01
