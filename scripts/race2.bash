#!/bin/bash
ros2 run race2 pure_pursuit --ros-args \
    -p waypoints_file:="/home/xiachu/f1tenth_ws/waypoints/race2.csv" \
    -p sim:=true \
    -p vis:=true \
    -p l_mode:="curvature_speed" \
    -p l:=1.0 \
    -p l_min:=0.75 \
    -p l_max:=1.25 \
    -p l_k:=0.1 \
    -p l_eps:=0.5 \
    -p n_ahead:=10 \
    -p p:=1.1 \
    -p max_speed:=5.3 \
    -p mu:=0.4 \
    -p max_steering_angle:=0.6 \
    -p interpolate:=false \
    -p wheelbase:=0.3302 \
    -p laser_offset:=0.27 \
    -p dead_reckon:=false

# fixed, l=1.0, mu=0.7
# speed, l_min=0.4, l_max=1.25, l_k=0.15, mu=0.7
# curvature, l_min=0.4, l_max=1.25, l_k=0.15, l_eps=0.5, mu=0.2
# curvature_speed, l_min=0.4, l_max=1.25, l_k=0.15, l_eps=0.8, mu=0.2
# curvature_speed, l_min=0.4, l_max=1.25, l_k=0.15, l_eps=0.5, mu=0.2

# real car 
# curvature_speed, l_min=0.4, l_max=1.0, l_k=0.15, l_eps=0.8

# Best on thursday:
# curvature_speed, l_min=0.75, l_max=1.25, l_k=0.1, l_eps=0.5, mu=0.3, dead_reckon=false
