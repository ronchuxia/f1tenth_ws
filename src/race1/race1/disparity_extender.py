#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from collections import deque

class DisparityExtender(Node):
    def __init__(self):
        super().__init__('disparity_extender')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscribe to LIDAR
        self.lidarscan_subscription = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        # Publish to drive
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.odom_subscription = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.v_x = 0.
        self.omega_z = 0.

        # A window of ranges
        self.window_size = 1   # dt ~ 0.005s * window_size
        self.ranges_window = deque()

        self.disparity_threshold = 0.5
        self.bubble_radius = 0.3 # 0.3
        self.debug = False

        self.angle_speed_publisher = self.create_publisher(Marker, 'angle_speed', 10)
        self.widest_gap_publisher = self.create_publisher(Marker, 'widest_gap', 10)
        self.lidar_timestamp = None

    def odom_callback(self, odom_msg):
        v_x = odom_msg.twist.twist.linear.x
        omega_z = odom_msg.twist.twist.angular.z

        self.v_x = v_x
        self.omega_z = omega_z

    def preprocess_lidar(self, data):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        ranges = np.array(data.ranges)
        ranges[np.isnan(ranges)] = data.range_min
        ranges[np.isinf(ranges)] = data.range_max

        # Maintain a window of ranges
        if len(self.ranges_window) >= self.window_size:
            self.ranges_window.popleft()
        self.ranges_window.append(ranges)
        
        # Compute mean over window and clip
        proc_ranges = np.mean(np.stack(self.ranges_window), axis=0)
        proc_ranges = np.clip(proc_ranges, data.range_min, data.range_max)

        num_ranges = len(ranges) 
        angle_min = data.angle_min
        angle_increment = data.angle_increment
        angles = np.arange(0, num_ranges) * angle_increment + angle_min

        idx_forward = np.abs(angles) < np.pi / 2.0
        proc_ranges = proc_ranges[idx_forward]
        angles = angles[idx_forward]

        return proc_ranges, angles

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        widest_gap_start = 0
        widest_gap_size = len(free_space_ranges)
        return widest_gap_start, widest_gap_start + widest_gap_size
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        gap = ranges[start_i:end_i]
        best_point_idx = np.argmax(gap) + start_i
        return best_point_idx

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        self.lidar_timestamp = data.header.stamp
        proc_ranges, angles = self.preprocess_lidar(data)
        num_ranges = len(proc_ranges)

        disparity_left = np.insert((proc_ranges[:-1] - proc_ranges[1:]), 0, 0) > self.disparity_threshold
        disparity_left_idx = np.where(disparity_left)[0]
        disparity_right = np.append((proc_ranges[1:] - proc_ranges[:-1]), 0) > self.disparity_threshold
        disparity_right_idx = np.where(disparity_right)[0]

        free_space_ranges = np.copy(proc_ranges)
        for i in disparity_left_idx:
            dist_to_disparity = proc_ranges[i] * np.sin(np.abs(angles - angles[i]))  
            points_in_bubble = (dist_to_disparity < self.bubble_radius) & (np.arange(num_ranges) < i)
            free_space_ranges[points_in_bubble] = np.minimum(proc_ranges[i], free_space_ranges[points_in_bubble])
        for i in disparity_right_idx:
            dist_to_disparity = proc_ranges[i] * np.sin(np.abs(angles - angles[i]))  
            points_in_bubble = (dist_to_disparity < self.bubble_radius) & (np.arange(num_ranges) > i)
            free_space_ranges[points_in_bubble] = np.minimum(proc_ranges[i], free_space_ranges[points_in_bubble])
        
        # Find max length gap 
        widest_gap_start, widest_gap_end = self.find_max_gap(free_space_ranges)

        if widest_gap_end == widest_gap_start:  # No gap found
            return

        # Find the best point in the gap
        best_point_idx = self.find_best_point(widest_gap_start, widest_gap_end, free_space_ranges)
        best_point_range = proc_ranges[best_point_idx]
        best_point_angle = angles[best_point_idx]

        best_point_angle /= 2
        # l = np.minimum(1, np.min(proc_ranges))
        # best_point_angle = 2 * np.arctan2(l * np.sin(best_point_angle), l * np.cos(best_point_angle) + 0.275)

        angle_deg = best_point_angle / math.pi * 180
        if abs(angle_deg) < 10 and best_point_range > 0.5:
            velocity = 3.8
        elif abs(angle_deg) < 20 and best_point_range > 0.5:
            velocity = 2.8
        else:
            velocity = 2.3
        # 1.5, 1.0, 1.0
        # 2.0, 1.0, 1.0
        # 2.0, 1.5, 1.0
        # 2.5, 1.5, 1.0
        # 2.5, 2.0, 1.0
        # 3.0, 2.0, 1.0
        # 3.0, 2.5, 1.0
        # 3.0, 2.5, 1.5
        # 3.5, 2.5, 1.5: crashes
        # 3.0, 2.5, 2.0
        # 3.3, 2.8, 2.3 (6s?)
        # 4.0, 2.8, 2.3
        # 3.8, 2.8, 2.3

        if best_point_angle < 0:
            if np.any(proc_ranges[angles < 0] < 0.2):
                best_point_angle = 0.0
        else:
            if np.any(proc_ranges[angles > 0] < 0.2):
                best_point_angle = 0.0

        if self.debug:
            self.draw_angle_speed(best_point_angle, velocity)
            self.draw_widest_gap(free_space_ranges, angles)

        # Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = best_point_angle
        self.drive_publisher.publish(drive_msg)
        
    def draw_angle_speed(self, angle, speed):
        arrow_length = speed

        arrow = Marker()
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.header.frame_id = 'ego_racecar/laser'
        arrow.ns = 'ttc_arrow'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        arrow.points = []
        start = Point()
        start.x = 0.0
        start.y = 0.0
        start.z = 0.0
        end = Point()
        end.x = arrow_length * np.cos(angle)
        end.y = arrow_length * np.sin(angle)
        end.z = 0.0
        arrow.points.append(start)
        arrow.points.append(end)

        arrow.scale.x = 0.05  # shaft diameter
        arrow.scale.y = 0.1   # head diameter

        arrow.color.r = 0.0
        arrow.color.g = 1.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0

        self.angle_speed_publisher.publish(arrow) 

    def draw_widest_gap(self, ranges, angles):
        points = Marker()
        points.header.stamp = self.lidar_timestamp
        points.header.frame_id = 'ego_racecar/laser'
        points.ns = 'widest_gap'
        points.id = 0
        points.type = Marker.POINTS
        points.action = Marker.ADD

        for i in range(len(ranges)):
            point = Point()
            point.x = ranges[i] * np.cos(angles[i])
            point.y = ranges[i] * np.sin(angles[i])
            point.z = 0.0
            points.points.append(point)
        
        points.scale.x = 0.1
        points.scale.y = 0.1
        points.scale.z = 0.1

        points.color.r = 1.0
        points.color.g = 0.0
        points.color.b = 0.0
        points.color.a = 1.0

        self.widest_gap_publisher.publish(points)


def main(args=None):
    rclpy.init(args=args)
    print("DisparityExtender Initialized")
    disparity_extender = DisparityExtender()
    rclpy.spin(disparity_extender)

    disparity_extender.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
