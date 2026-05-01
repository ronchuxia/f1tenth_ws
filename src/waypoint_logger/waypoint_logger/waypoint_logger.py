#!/usr/bin/env python3

"""
Log waypoints from:
1. Odometry from particle filter.
2. Odometry from simulation.
3. Clicked points from RViz 2.
"""

import rclpy
from rclpy.node import Node

import numpy as np
from numpy import linalg as LA
from time import gmtime, strftime
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from f1tenth_utils.vis_utils import visualize_point
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, DurabilityPolicy

import csv


class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger_node')

        self.declare_parameter('file_name', strftime('wp-%Y-%m-%d-%H-%M-%S', gmtime()) + '.csv')
        self.declare_parameter('mode', 'sim')
        self.declare_parameter('laser_offset', 0.275)   # laser offset from base_link in meters, only used for pf odom

        self.file_name = self.get_parameter('file_name').value
        self.mode = self.get_parameter('mode').value
        self.laser_offset = self.get_parameter('laser_offset').value

        if self.mode == 'pf':
            self.odom_subscription = self.create_subscription(Odometry, '/pf/pose/odom', self.save_waypoint_from_odometry, 10)  # real life, particle filter
        elif self.mode == 'sim':
            self.odom_subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.save_waypoint_from_odometry, 10)    # simulation, odom
        elif self.mode == 'rviz':
            self.odom_subscription = self.create_subscription(PointStamped, '/clicked_point', self.save_waypoint_from_pointstamped, 10)   # simulation, clicked point

        self.file = open(self.file_name, 'a+', newline='')
        self.csv_writer = csv.writer(self.file)

        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/logged_waypoints', qos)
        self.markers = MarkerArray()
        self.marker_id = 0

        self.load_existing_waypoints()

    def save_waypoint_from_odometry(self, data):
        quaternion = np.array([data.pose.pose.orientation.x, 
                            data.pose.pose.orientation.y, 
                            data.pose.pose.orientation.z, 
                            data.pose.pose.orientation.w])
        euler_z = euler_from_quaternion(quaternion)[2]

        pos_x = data.pose.pose.position.x
        pos_y = data.pose.pose.position.y

        if self.mode == 'pf':
            # transform from laser frame to base_link frame
            pos_x -= self.laser_offset * np.cos(euler_z)
            pos_y -= self.laser_offset * np.sin(euler_z)

        speed = LA.norm(np.array([data.twist.twist.linear.x, 
                                data.twist.twist.linear.y, 
                                data.twist.twist.linear.z]), 2)

        if data.twist.twist.linear.x > 1e-5:
            self.csv_writer.writerow([pos_x,
                                      pos_y,
                                      euler_z,
                                      speed])
            self.get_logger().info(f"Waypoint {self.marker_id} logged.")

            marker = visualize_point((pos_x, pos_y), self.get_clock().now().to_msg(), id=self.marker_id)
            self.markers.markers.append(marker)
            self.marker_pub.publish(self.markers)
            self.marker_id += 1

    def save_waypoint_from_pointstamped(self, data):
        self.csv_writer.writerow([data.point.x, data.point.y])
        self.get_logger().info(f"Waypoint {self.marker_id} logged.")

        marker = visualize_point((data.point.x, data.point.y), self.get_clock().now().to_msg(), id=self.marker_id)
        self.markers.markers.append(marker)
        self.marker_pub.publish(self.markers)
        self.marker_id += 1

    def destroy_node(self):
        self.file.close()
        super().destroy_node()

    def load_existing_waypoints(self):
        self.file.seek(0)
        reader = csv.reader(self.file)

        for row in reader:
            x, y = float(row[0]), float(row[1])
            marker = visualize_point((x, y), self.get_clock().now().to_msg(), id=self.marker_id)
            self.markers.markers.append(marker)
            self.marker_id += 1
        
        self.marker_pub.publish(self.markers)
        self.get_logger().info(f"Loaded {self.marker_id} pre-existing waypoints.")


def main(args=None):
    rclpy.init(args=args)
    print("WaypointLogger Initialized")
    waypoint_logger_node = WaypointLogger()
    rclpy.spin(waypoint_logger_node)

    waypoint_logger_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()