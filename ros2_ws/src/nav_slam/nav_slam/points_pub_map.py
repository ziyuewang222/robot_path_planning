#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2025 <Ming2zun:https://github.com/Ming2zun/Pure-tracking-slam-automatic-navigation-system>
#                <喵了个水蓝蓝:https://www.bilibili.com/video/BV1kzEwzuEFw?spm_id_from=333.788.videopod.sections&vd_source=134c12873ff478ea447a06d652426f8f>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math


def laser_scan_to_xyz(scan_msg):
    """将 LaserScan 消息转换为 (N, 3) 的 xyz 点云数组"""
    angle_min = scan_msg.angle_min
    angle_max = scan_msg.angle_max
    angle_increment = scan_msg.angle_increment
    range_min = scan_msg.range_min
    range_max = scan_msg.range_max

    points = []
    angle = angle_min
    for r in scan_msg.ranges:
        if range_min <= r <= range_max:
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = 0.0
            points.append([x, y, z])
        angle += angle_increment

    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(points, dtype=np.float32)


class PointCloudTransformNode(Node):
    def __init__(self):
        super().__init__('pointcloud_transform_node')

        # Declare parameters
        self.declare_parameter('frame_id', 'map')

        # Create subscribers — 改为订阅 /scan (LaserScan)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Create publisher
        self.transformed_pointcloud_pub = self.create_publisher(PointCloud2, '/mapokk', 10)

        # Initialize variables
        self.odom_data = None
        self.rotation_matrix = None
        self.translation = None

    def odom_callback(self, msg):
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        translation = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )

        # Check if the new data is different from the old one to avoid unnecessary computation
        if (quaternion != getattr(self.odom_data, 'pose.pose.orientation', None) or
            translation != getattr(self.odom_data, 'pose.pose.position', None)):

            self.odom_data = msg

            qx, qy, qz, qw = quaternion
            sqx, sqy, sqz = qx * qx, qy * qy, qz * qz

            m00, m01, m02 = 1 - 2*(sqy + sqz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)
            m10, m11, m12 = 2*(qx*qy + qw*qz), 1 - 2*(sqx + sqz), 2*(qy*qz - qw*qx)
            m20, m21, m22 = 2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(sqx + sqy)

            self.rotation_matrix = np.array([
                [m00, m01, m02, translation[0]],
                [m10, m11, m12, translation[1]],
                [m20, m21, m22, translation[2]],
                [0, 0, 0, 1]
            ])

    def scan_callback(self, msg):
        if self.odom_data is None or self.rotation_matrix is None:
            return

        # 将 LaserScan 转换为 xyz 点云
        points = laser_scan_to_xyz(msg)

        if len(points) == 0:
            return

        # 将点转换为齐次坐标
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])

        # 应用变换矩阵
        transformed_points = (self.rotation_matrix @ points_homogeneous.T).T[:, :3]

        header = msg.header
        header.frame_id = self.get_parameter('frame_id').value
        cloud_msg = pc2.create_cloud_xyz32(header, transformed_points)
        self.transformed_pointcloud_pub.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudTransformNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()



