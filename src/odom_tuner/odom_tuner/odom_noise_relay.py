#!/usr/bin/env python3

"""
This file simulates noisy particle filter localization by adding Gaussian noise.

Subscriptions:
    - /ego_racecar/odom_throttle (nav_msgs/Odometry)
    
Publications:
    - /pf/pose/odom (nav_msgs/Odometry)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class OdomNoiseRelay(Node):
    def __init__(self):
        super().__init__('odom_noise_relay')
        self.declare_parameter('pos_noise_std', 0.05)   # m, std dev added to x and y
        self.declare_parameter('yaw_noise_std', 0.01)   # rad, std dev added to yaw

        self._pos_std = self.get_parameter('pos_noise_std').value
        self._yaw_std = self.get_parameter('yaw_noise_std').value

        self._pub = self.create_publisher(Odometry, '/pf/pose/odom', 10)
        self.create_subscription(Odometry, '/ego_racecar/odom_throttle', self._cb, 10)

    def _cb(self, msg: Odometry):
        out = Odometry()
        out.header = msg.header
        out.header.frame_id = 'map'
        out.child_frame_id = msg.child_frame_id

        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        noisy_x   = msg.pose.pose.position.x + np.random.normal(0.0, self._pos_std)
        noisy_y   = msg.pose.pose.position.y + np.random.normal(0.0, self._pos_std)
        noisy_yaw = yaw + np.random.normal(0.0, self._yaw_std)

        out.pose.pose.position.x = noisy_x
        out.pose.pose.position.y = noisy_y
        out.pose.pose.position.z = msg.pose.pose.position.z

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, noisy_yaw)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        out.twist = msg.twist

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomNoiseRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
