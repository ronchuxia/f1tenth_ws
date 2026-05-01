#!/usr/bin/env python3

from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.time import Time

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf_transformations import euler_from_quaternion

from f1tenth_utils.vis_utils import visualize_point, visualize_points, visualize_trajectory

from race3.avoidance.frenet_planner import FrenetPlanner

from race3.tracker.pp import PurePursuitTracker


class Race3(Node):
    def __init__(self):
        super().__init__('race3')
        self.initialize_parameters()

        self.pose_odom_topic = "/ego_racecar/odom_throttle" if self.sim else "/pf/pose/odom"
        self.velocity_odom_topic = "/odom"
        self.scan_topic = "/scan"
        self.drive_topic = "/drive"
        latest_qos = QoSProfile(depth=1)
        self.pose_subscription = self.create_subscription(Odometry, self.pose_odom_topic, self.pose_callback, latest_qos)
        self.velocity_subscription = self.create_subscription(Odometry, self.velocity_odom_topic, self.velocity_callback, latest_qos)
        self.scan_subscription = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, latest_qos)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)

        self.waypoint_columns, self.waypoint_data = self._load_waypoint_data(self.waypoints_file)
        self.waypoints = self.waypoint_data[:, :2]
        self.waypoint_speed_profile = self._get_waypoint_column('vx_mps')
        self.scan = None

        self.base_link_frame = 'ego_racecar/base_link' if self.sim else 'base_link'

        self.tracker = PurePursuitTracker(
            wheelbase=self.wheelbase,
            max_speed=self.max_speed,
            mu=self.mu,
            max_steering_angle=self.max_steering_angle,
            l_min=self.l_min,
            l_max=self.l_max,
            l_k=self.l_k,
            l_eps=self.l_eps,
            n_ahead=self.n_ahead,
            p=self.p,
            waypoints=self.waypoints,
            speed_profile=self.waypoint_speed_profile,
            use_speed_profile=self.use_waypoint_speed_profile,
            sim=self.sim,
        )

        self.avoidance = FrenetPlanner(
            max_speed=self.frenet_max_speed,
            mu=self.frenet_mu,
            p=self.frenet_p,
            max_steering_angle=self.max_steering_angle,
            wheelbase=self.wheelbase,
            laser_offset=self.laser_offset,
            car_width=self.frenet_car_width,
            lookahead=self.frenet_lookahead,
            num_offsets=self.frenet_num_offsets,
            offset_spacing=self.frenet_offset_spacing,
            num_samples=self.frenet_num_samples,
            w_offset=self.frenet_w_offset,
            w_jerk=self.frenet_w_jerk,
            w_clear=self.frenet_w_clear,
            pp_lookahead=self.frenet_pp_lookahead,
            waypoints=self.waypoints,
            sim=self.sim,
        )

        if self.vis:
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.waypoints_publisher = self.create_publisher(Marker, '/rviz_waypoints', qos)
            points_marker = visualize_points(
                self.waypoints,
                self.get_clock().now().to_msg(),
                frame_id='/map',
                color=(0.0, 1.0, 0.0, 1.0),
                color_end=(0.0, 0.0, 1.0, 1.0),
            )
            self.waypoints_publisher.publish(points_marker)

            self.markers = MarkerArray()
            self.markers_publisher = self.create_publisher(MarkerArray, '/rviz_markers', 10)

    def initialize_parameters(self):
        # general
        self.declare_parameter('vis', True)
        self.declare_parameter('vis_obs', True)
        self.declare_parameter('vis_traker', False)
        self.declare_parameter('waypoints_file', '/home/xiachu/f1tenth_race_ws/waypoints_postprocessed.csv')
        self.declare_parameter('sim', False)
        self.declare_parameter('scan_self_filter_range', 0.0)
        self.declare_parameter('enable_pose_prediction', False)

        # pure pursuit
        self.declare_parameter('l_min', 0.4)
        self.declare_parameter('l_max', 2.0)
        self.declare_parameter('l_k', 0.15)
        self.declare_parameter('l_eps', 0.5)
        self.declare_parameter('n_ahead', 10)
        self.declare_parameter('use_waypoint_speed_profile', False)

        # drive
        self.declare_parameter('p', 1.0)
        self.declare_parameter('max_speed', 5.0)
        self.declare_parameter('mu', 0.7)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('wheelbase', 0.3302)
        self.declare_parameter('laser_offset', 0.27)

        # state machine
        self.declare_parameter('reacquire_frames', 10)
        self.declare_parameter('avoidance_frames', 1)

        # avoidance
        self.declare_parameter('avoidance_trigger', 'waypoint_radius')  # 'always' | 'never' | 'waypoint_radius'
        self.declare_parameter('trigger_wp_radius', 0.3)
        self.declare_parameter('trigger_wp_lookahead', 2.0)
        self.declare_parameter('frenet_car_width', 0.35)
        self.declare_parameter('frenet_lookahead', 2.0)
        self.declare_parameter('frenet_num_offsets', 7)
        self.declare_parameter('frenet_offset_spacing', 0.20)
        self.declare_parameter('frenet_num_samples', 20)
        self.declare_parameter('frenet_w_offset', 1.0)
        self.declare_parameter('frenet_w_jerk', 0.01)
        self.declare_parameter('frenet_w_clear', 0.5)
        self.declare_parameter('frenet_max_speed', 5.0)
        self.declare_parameter('frenet_mu', 0.7)
        self.declare_parameter('frenet_p', 1.0)
        self.declare_parameter('frenet_pp_lookahead', 0.8)

        self.vis = self.get_parameter('vis').value
        self.vis_obs = self.get_parameter('vis_obs').value
        self.vis_traker = self.get_parameter('vis_traker').value
        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.sim = self.get_parameter('sim').value
        self.scan_self_filter_range = self.get_parameter('scan_self_filter_range').value
        self.enable_pose_prediction = self.get_parameter('enable_pose_prediction').value
        self.l_min = self.get_parameter('l_min').value
        self.l_max = self.get_parameter('l_max').value
        self.l_k = self.get_parameter('l_k').value
        self.l_eps = self.get_parameter('l_eps').value
        self.n_ahead = self.get_parameter('n_ahead').value
        self.use_waypoint_speed_profile = self.get_parameter('use_waypoint_speed_profile').value
        self.p = self.get_parameter('p').value
        self.max_speed = self.get_parameter('max_speed').value
        self.mu = self.get_parameter('mu').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.laser_offset = self.get_parameter('laser_offset').value
        self.reacquire_frames = self.get_parameter('reacquire_frames').value
        self.avoidance_frames = self.get_parameter('avoidance_frames').value
        self.avoidance_trigger = self.get_parameter('avoidance_trigger').value
        self.trigger_wp_radius = self.get_parameter('trigger_wp_radius').value
        self.trigger_wp_lookahead = self.get_parameter('trigger_wp_lookahead').value
        self.frenet_car_width = self.get_parameter('frenet_car_width').value
        self.frenet_lookahead = self.get_parameter('frenet_lookahead').value
        self.frenet_num_offsets = self.get_parameter('frenet_num_offsets').value
        self.frenet_offset_spacing = self.get_parameter('frenet_offset_spacing').value
        self.frenet_num_samples = self.get_parameter('frenet_num_samples').value
        self.frenet_w_offset = self.get_parameter('frenet_w_offset').value
        self.frenet_w_jerk = self.get_parameter('frenet_w_jerk').value
        self.frenet_w_clear = self.get_parameter('frenet_w_clear').value
        self.frenet_max_speed = self.get_parameter('frenet_max_speed').value
        self.frenet_mu = self.get_parameter('frenet_mu').value
        self.frenet_p = self.get_parameter('frenet_p').value
        self.frenet_pp_lookahead = self.get_parameter('frenet_pp_lookahead').value

        self.v = 0.0
        self.angle = 0.0
        self.state = 'following'
        self.blocked_frames = 0
        self.clear_frames = 0
        self.measured_speed = None

    def _load_waypoint_data(self, path):
        columns = []
        with open(path, 'r', encoding='utf-8') as waypoint_file:
            for line in waypoint_file:
                stripped = line.strip()
                if not stripped.startswith('#'):
                    break
                header = stripped.lstrip('#').strip()
                if ',' in header:
                    columns = [item.strip() for item in header.split(',')]

        data = np.loadtxt(path, delimiter=',', comments='#')
        data = np.atleast_2d(data)
        return columns, data

    def _get_waypoint_column(self, name):
        if name not in self.waypoint_columns:
            return None
        idx = self.waypoint_columns.index(name)
        if idx >= self.waypoint_data.shape[1]:
            return None
        return self.waypoint_data[:, idx]

    def scan_callback(self, scan_msg):
        if self.scan_self_filter_range <= 0.0:
            self.scan = scan_msg
            return

        filtered_scan = deepcopy(scan_msg)
        ranges = np.asarray(filtered_scan.ranges, dtype=np.float32)
        min_valid_range = max(float(filtered_scan.range_min), float(self.scan_self_filter_range))
        ranges[ranges < min_valid_range] = np.inf
        filtered_scan.ranges = ranges.tolist()
        self.scan = filtered_scan

    def velocity_callback(self, odom_msg):
        self.measured_speed = float(odom_msg.twist.twist.linear.x)

    def _predict_pose(self, pos, yaw, v_current, delta_current, dt):
        if dt <= 0.0:
            return pos, yaw

        delta = float(np.clip(delta_current, -self.max_steering_angle, self.max_steering_angle))
        x, y = float(pos[0]), float(pos[1])
        if abs(delta) < 1e-6:
            x += v_current * np.cos(yaw) * dt
            y += v_current * np.sin(yaw) * dt
            return np.array([x, y]), yaw

        yaw_rate = v_current * np.tan(delta) / self.wheelbase
        yaw_next = yaw + yaw_rate * dt
        x += v_current * np.cos(yaw) * dt
        y += v_current * np.sin(yaw) * dt
        return np.array([x, y]), yaw_next

    def pose_callback(self, pose_msg):
        self.timestamp = pose_msg.header.stamp
        quaternion = np.array([pose_msg.pose.pose.orientation.x,
                               pose_msg.pose.pose.orientation.y,
                               pose_msg.pose.pose.orientation.z,
                               pose_msg.pose.pose.orientation.w])
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]

        # Transform pose to base_link frame
        pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        if not self.sim:
            pos[0] -= self.laser_offset * np.cos(yaw)
            pos[1] -= self.laser_offset * np.sin(yaw)

        v_meas = self.measured_speed
        if v_meas is None:
            v_meas = float(pose_msg.twist.twist.linear.x)
        delta_current = self.angle
        pose_latency = 0.0
        if self.enable_pose_prediction:
            pose_latency = max(
                0.0,
                (self.get_clock().now() - Time.from_msg(pose_msg.header.stamp)).nanoseconds * 1e-9,
            )
            pos, yaw = self._predict_pose(pos, yaw, v_meas, delta_current, pose_latency)

        # Check if avoidance should be triggered, and update state machine
        if self.avoidance_trigger == 'always':
            blocked = True
        elif self.avoidance_trigger == 'never':
            blocked = False
        elif self.avoidance_trigger == 'waypoint_radius':
            blocked = self.is_waypoint_blocked(pos, yaw)
        else:
            raise NotImplementedError(f"avoidance_trigger '{self.avoidance_trigger}' not supported")
        self._update_state(blocked)

        # Run exactly one controller per cycle: avoidance when blocked,
        # otherwise the normal tracker.
        tracker_markers = []
        avoid_markers = []
        if self.state == 'avoiding' and self.scan is not None:
            self.angle, self.v, avoid_markers = self.avoidance.compute_drive(
                self.scan, pos[0], pos[1], yaw, self.waypoints, vis=self.vis and self.vis_obs,
                stamp=self.timestamp,
            )
            controller_source = 'avoidance'
        else:
            out = self.tracker.compute_drive(
                pose_x=pos[0], pose_y=pos[1], pose_yaw=yaw,
                v_current=v_meas, delta_current=delta_current,
                waypoints=self.waypoints, stamp=self.timestamp, vis=self.vis,
            )
            self.angle = out['steering']
            self.v = out['speed']
            tracker_markers = out['markers']
            controller_source = 'tracker'

        self.get_logger().info(
            f"[{self.state}] source: {controller_source}, angle: {self.angle:.3f}, speed: {self.v:.3f}, pose_latency: {pose_latency:.3f}s"
        )

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = self.v
        drive_msg.drive.steering_angle = self.angle
        self.drive_publisher.publish(drive_msg)

        if self.vis:
            self.markers = MarkerArray()
            self.markers.markers.append(visualize_point(
                pos, self.timestamp, frame_id='/map', ns='base_link', id=0,
                color=(0.0, 1.0, 1.0, 1.0),
            ))
            self.markers.markers.append(visualize_trajectory(
                self.angle, self.timestamp,
                frame_id=self.base_link_frame,
                ns='commanded_trajectory', id=2,
                color=(0.0, 0.8, 1.0, 1.0),
                wheelbase=self.wheelbase,
            ))
            if self.vis_traker:
                self.markers.markers.extend(tracker_markers)
            if self.vis_obs:
                if self.state == 'following':   # clear, following
                    color = (0.0, 1.0, 0.0, 0.6)
                elif blocked:   # blocked, avoiding
                    color = (1.0, 0.0, 0.0, 0.8)
                else:   # clear, avoiding (transitioning back to following)
                    color = (1.0, 1.0, 0.0, 0.7)
                if self.avoidance_trigger == 'waypoint_radius':
                    wp_points = self._forward_waypoints_bl(pos, yaw)
                    if wp_points.shape[0] > 0:
                        self.markers.markers.append(visualize_points(
                            wp_points, self.timestamp,
                            frame_id=self.base_link_frame,
                            ns='trigger_waypoints', id=1, color=color,
                        ))
                self.markers.markers.extend(avoid_markers)
            self.markers_publisher.publish(self.markers)

    def _update_state(self, blocked):
        if self.state == 'following':
            if blocked:
                self.blocked_frames += 1
                if self.blocked_frames >= self.avoidance_frames:
                    self.state = 'avoiding'
                    self.blocked_frames = 0
                    self.clear_frames = 0
                    self.get_logger().warn("state: following -> avoiding")
            else:
                self.blocked_frames = 0
        else:  # 'avoiding'
            self.blocked_frames = 0
            if blocked:
                self.clear_frames = 0
            else:
                self.clear_frames += 1
                if self.clear_frames >= self.reacquire_frames:
                    self.state = 'following'
                    self.clear_frames = 0
                    self.get_logger().info("state: avoiding -> following")

    def _forward_waypoints_bl(self, pos, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        dx = self.waypoints[:, 0] - pos[0]
        dy = self.waypoints[:, 1] - pos[1]
        wx =  dx * c + dy * s
        wy = -dx * s + dy * c
        fwd = (wx > 0.0) & (np.hypot(wx, wy) < self.trigger_wp_lookahead)
        return np.column_stack([wx[fwd], wy[fwd]])

    def is_waypoint_blocked(self, pos, yaw):
        if self.scan is None:
            return False

        c, s = np.cos(yaw), np.sin(yaw)
        dx = self.waypoints[:, 0] - pos[0]
        dy = self.waypoints[:, 1] - pos[1]
        wx =  dx * c + dy * s
        wy = -dx * s + dy * c
        fwd = (wx > 0.0) & (np.hypot(wx, wy) < self.trigger_wp_lookahead)
        if not np.any(fwd):
            return False
        wxy = np.column_stack([wx[fwd], wy[fwd]])

        ranges = np.asarray(self.scan.ranges, dtype=np.float64)
        angles = self.scan.angle_min + np.arange(ranges.shape[0]) * self.scan.angle_increment
        valid = np.isfinite(ranges) & (ranges > self.scan.range_min) & (ranges < self.scan.range_max)
        sx = self.laser_offset + ranges * np.cos(angles)
        sy = ranges * np.sin(angles)
        keep = valid & (sx > 0.0)
        if not np.any(keep):
            return False
        sxy = np.column_stack([sx[keep], sy[keep]])

        diff = wxy[:, None, :] - sxy[None, :, :]
        min_d = np.linalg.norm(diff, axis=2).min(axis=1)
        return bool(np.any(min_d < self.trigger_wp_radius))


def main(args=None):
    rclpy.init(args=args)
    print("Race3 Initialized")
    race3 = Race3()
    rclpy.spin(race3)

    race3.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
