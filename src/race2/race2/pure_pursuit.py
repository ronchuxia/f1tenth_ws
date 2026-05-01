#!/usr/bin/env python3

import rclpy
import rclpy.time
from rclpy.node import Node

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf_transformations import euler_from_quaternion

from f1tenth_utils.vis_utils import visualize_trajectory, visualize_point, visualize_points
from f1tenth_utils.utils import max_speed_from_steering_angle, dead_reckon


class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.initialize_parameters()
        
        self.odom_subscription = self.create_subscription(Odometry, '/ego_racecar/odom_throttle' if self.sim else '/pf/pose/odom', self.pose_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # read waypoints from file
        self.waypoints = np.loadtxt(self.waypoints_file, delimiter=',')[:,:2]
        self.path_index = None

        if self.vis:
            # publisher for all waypoints (publish once)
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.waypoints_publisher = self.create_publisher(Marker, '/rviz_waypoints', qos)
            points_marker = visualize_points(self.waypoints, self.get_clock().now().to_msg(), color=(0.0, 1.0, 0.0, 1.0), color_end=(0.0, 0.0, 1.0, 1.0),)
            self.waypoints_publisher.publish(points_marker)

            self.markers = MarkerArray()
            self.markers_publisher = self.create_publisher(MarkerArray, '/rviz_markers', 10)

    def initialize_parameters(self):
        self.declare_parameter('sim', False)
        self.declare_parameter('vis', True)
        self.declare_parameter('waypoints_file', '/home/xiachu/f1tenth_race_ws/waypoints_postprocessed.csv')

        self.declare_parameter('l_mode', 'fixed')  # 'fixed', 'speed', 'curvature', 'curvature_speed'
        # speed adaptive
        self.declare_parameter('l', 0.5)
        self.declare_parameter('l_min', 0.4)
        self.declare_parameter('l_max', 2.0)
        self.declare_parameter('l_k', 0.15)
        # curvature adaptive
        self.declare_parameter('l_eps', 0.5)   # curvature regularizer: prevents division by zero on straights
        self.declare_parameter('n_ahead', 10)  # number of waypoints to look ahead for curvature
        # drive
        self.declare_parameter('p', 1.0)    # p for steering angle p controller
        self.declare_parameter('max_speed', 5.0)
        self.declare_parameter('mu', 0.7)   # friction coefficient between the tire and the ground, used to calculate max speed from steering angle. Reduce this value to reduce speed.
        self.declare_parameter('max_steering_angle', 0.4189) # max steering angle in sim
        self.declare_parameter('interpolate', False)
        self.declare_parameter('wheelbase', 0.3302)
        self.declare_parameter('laser_offset', 0.27)
        self.declare_parameter('dead_reckon', False)  # compensate for PF latency, only used when sim=False

        self.sim = self.get_parameter('sim').value
        self.vis = self.get_parameter('vis').value
        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.l_mode = self.get_parameter('l_mode').value
        self.l = self.get_parameter('l').value
        self.l_min = self.get_parameter('l_min').value
        self.l_max = self.get_parameter('l_max').value
        self.l_k = self.get_parameter('l_k').value
        self.l_eps = self.get_parameter('l_eps').value
        self.n_ahead = self.get_parameter('n_ahead').value
        self.p = self.get_parameter('p').value
        self.max_speed = self.get_parameter('max_speed').value
        self.mu = self.get_parameter('mu').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.interpolate = self.get_parameter('interpolate').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.laser_offset = self.get_parameter('laser_offset').value
        self.dead_reckon = self.get_parameter('dead_reckon').value

        self.v = 0.0     # previous speed, used for adaptive lookahead and dead reckoning
        self.angle = 0.0 # previous steering angle, used for dead reckoning
        
    def pose_callback(self, pose_msg):
        self.timestamp = pose_msg.header.stamp
        quaternion = np.array([pose_msg.pose.pose.orientation.x, 
                            pose_msg.pose.pose.orientation.y, 
                            pose_msg.pose.pose.orientation.z, 
                            pose_msg.pose.pose.orientation.w])
        euler = euler_from_quaternion(quaternion)
        self.euler_z = euler[2]

        # find the current waypoint to track
        self.pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        # NOTE: If using pf/pose/odom, position is the laser position, not the base_link position. Subtract the laser offset to get base_link position.
        if not self.sim:
            self.pos[0] -= self.laser_offset * np.cos(self.euler_z)
            self.pos[1] -= self.laser_offset * np.sin(self.euler_z)
        
        if self.dead_reckon:
            dt = (self.get_clock().now() - rclpy.time.Time.from_msg(pose_msg.header.stamp)).nanoseconds * 1e-9
            self.get_logger().info(f"pose_latency: {dt:.5f}")
            self.pos[0], self.pos[1], self.euler_z = dead_reckon(
                self.pos[0], self.pos[1], self.euler_z,
                self.v, self.angle, dt, self.wheelbase
            )

        if self.l_mode == 'speed':
            l = np.clip(self.l_min + self.l_k * self.v, self.l_min, self.l_max)
        elif self.l_mode == 'curvature':
            kappa = self.compute_ahead_curvature(self.waypoints, self.path_index or 0, self.n_ahead)
            l = np.clip(self.l_min + 0.5 / (kappa + self.l_eps), self.l_min, self.l_max)
        elif self.l_mode == 'curvature_speed':
            kappa = self.compute_ahead_curvature(self.waypoints, self.path_index or 0, self.n_ahead)
            l = np.clip(self.l_min + self.l_k * self.v / (kappa + self.l_eps), self.l_min, self.l_max)
        else:  # fixed
            l = self.l
        
        target, actual_l = self.find_target_l_loop(self.waypoints, self.pos, l)

        # convert target from map frame to base_link frame.
        dx = target[0] - self.pos[0]                  
        dy = target[1] - self.pos[1]                          
        self.target_x = dx * np.cos(self.euler_z) + dy * np.sin(self.euler_z)                       
        self.target_y = -dx * np.sin(self.euler_z) + dy * np.cos(self.euler_z)

        # calculate curvature/steering angle
        gamma = 2 * self.target_y / actual_l ** 2
        angle = self.p * np.arctan(gamma * self.wheelbase)
        self.angle = np.clip(angle, -self.max_steering_angle, self.max_steering_angle)

        # calculate speed
        max_speed = max_speed_from_steering_angle(angle, mu=self.mu)
        self.v = min(self.max_speed, max_speed)
        self.get_logger().info(f"angle: {self.angle:.3f}, speed: {self.v:.3f}, actual_l: {actual_l:.3f}")

        # publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = self.v
        drive_msg.drive.steering_angle = self.angle
        self.drive_publisher.publish(drive_msg)

        # visualize goal point and trajectory
        if self.vis:
            vis_frame = '/ego_racecar/base_link' if self.sim else '/base_link'
            self.markers = MarkerArray()
            trajectory_marker = visualize_trajectory(self.angle, self.timestamp, wheelbase=self.wheelbase, frame_id=vis_frame)
            target_marker = visualize_point(target, self.timestamp)
            self.markers.markers.append(trajectory_marker)
            self.markers.markers.append(target_marker)
            self.markers_publisher.publish(self.markers)

    def compute_ahead_curvature(self, waypoints, start_idx, n_ahead):
        n = len(waypoints)
        max_kappa = 0.0
        for i in range(n_ahead - 2):
            a = waypoints[(start_idx + i    ) % n]
            b = waypoints[(start_idx + i + 1) % n]
            c = waypoints[(start_idx + i + 2) % n]
            cross = abs((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
            denom = np.linalg.norm(b-a) * np.linalg.norm(c-b) * np.linalg.norm(c-a)
            if denom > 1e-6:
                max_kappa = max(max_kappa, 2 * cross / denom)
        return max_kappa

    def find_target_l_vec(self, waypoints, pos, l):
        # NOTE: This function currently does not support curvature adaptive lookahead distance
        # because it does not update self.path_index.
        dist = np.linalg.norm(waypoints - pos, axis=1)

        smaller = (dist <= l)
        larger = (dist > l)
        target = np.bitwise_and(smaller[:-1], larger[1:])
        target_idx = target.nonzero()[0][0] if target.any() else (np.argmin(dist) + 5) % len(waypoints)

        # self.get_logger().info(f"{target_idx}")

        waypoint_smaller = self.waypoints[target_idx]
        waypoint_larger = self.waypoints[(target_idx+1)%len(waypoints)]

        if not self.interpolate:
            target = waypoint_larger
            actual_l = dist[(target_idx+1)%len(waypoints)]
        else:
            d = waypoint_larger - waypoint_smaller
            f = waypoint_smaller - self.pos
            a = d @ d                                  
            b = 2 * (f @ d)                            
            c = (f @ f) - self.l ** 2                  
            t = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)   
            target = waypoint_smaller + t * d                
            actual_l = self.l

        return target, actual_l

    def find_target_l_loop(self, waypoints, pos, l):
        # NOTE: This function loops from waypoint n-1 back to waypoint 0.
        # If waypoint 0 is already more than l meters away from the car at this time, 
        # waypoint 0 will be chosen as the target even though it is behind the car.
        n = len(waypoints)

        # initialize path_index to the closest waypoint at first callback
        if self.path_index is None:
            self.path_index = int(np.argmin(np.linalg.norm(waypoints - pos, axis=1)))

        target = waypoints[self.path_index]
        actual_l = l

        for i in range(n):
            idx = (self.path_index + i) % n
            dx = waypoints[idx, 0] - pos[0]
            dy = waypoints[idx, 1] - pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist >= l:
                target = waypoints[idx]
                actual_l = dist
                self.path_index = idx
                break

        return target, actual_l


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
