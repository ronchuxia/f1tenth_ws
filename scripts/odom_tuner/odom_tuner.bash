#!/bin/bash

# robot
ARGS="-p drive_start_delay:=0.0"        # seconds to wait before publishing to /drive
ARGS="$ARGS -p speed:=1.0"
ARGS="$ARGS -p steering_angle:=0.0"
ARGS="$ARGS -p wheelbase:=0.3302"           # m, used for Ackermann curvature reference
ARGS="$ARGS -p laser_offset:=0.27"          # m, lidar forward offset from base_link
ARGS="$ARGS -p g:=9.81"                     # m/s², scales raw IMU output to SI

# windowed velocity estimate
ARGS="$ARGS -p enable_vel_window:=true"
ARGS="$ARGS -p vel_window:=30"              # poses averaged; larger = smoother, more latency
ARGS="$ARGS -p vel_project:=true"           # project displacement onto heading (vs euclidean arc length)
ARGS="$ARGS -p vel_window_ema_alpha:=1.0"   # EMA weight on new sample [0,1]; smaller = smoother, more latency

# CTRV EKF  (state: px py yaw v omega;  noise: a ~ N(0,sigma_a^2), omega_dot ~ N(0,sigma_omega_dot^2))
ARGS="$ARGS -p enable_vel_ctrv:=true"
ARGS="$ARGS -p kf_ctrv_sigma_a:=8.0"           # longitudinal acceleration std (m/s²); larger = less latency, noisier
ARGS="$ARGS -p kf_ctrv_sigma_omega_dot:=0.4"   # angular acceleration std (rad/s²);    larger = less latency, noisier
ARGS="$ARGS -p kf_ctrv_r_pos:=0.05"            # PF position std (m);   larger = smoother, more latency
ARGS="$ARGS -p kf_ctrv_r_yaw:=0.05"            # PF heading std (rad);  larger = smoother, more latency
ARGS="$ARGS -p kf_ctrv_r_from_cov:=false"      # use PF covariance for R instead of fixed values above
ARGS="$ARGS -p kf_ctrv_p0_v:=100.0"            # initial P: v variance (m²/s²);    larger = faster convergence, more initial noise
ARGS="$ARGS -p kf_ctrv_p0_omega:=1.0"         # initial P: ω variance (rad²/s²)

# visualisation
ARGS="$ARGS -p trail_length:=5000"
ARGS="$ARGS -p curve_window:=400"
ARGS="$ARGS -p show_curve_hist:=false"
ARGS="$ARGS -p show_vel_hist:=true"
ARGS="$ARGS -p show_trail:=false"
ARGS="$ARGS -p show_arc:=false"
ARGS="$ARGS -p show_center:=false"
ARGS="$ARGS -p show_theory:=true"
ARGS="$ARGS -p show_arc_thru_base_link:=false"

# logging / recording
ARGS="$ARGS -p log_imu:=false"
ARGS="$ARGS -p log_odom:=false"
ARGS="$ARGS -p log_pf:=true"
ARGS="$ARGS -p record_estimates:=true"

ros2 run odom_tuner odom_tuner --ros-args $ARGS

# 1m/s, 0.3rad, 200pts half circle, 400pts full circle
