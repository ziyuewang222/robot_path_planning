#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""里程计 TF 广播节点（入口名 odom_tf_broadcaster）。

用途：当底盘驱动**只发布 /odom 话题、不发布 odom->base_link 的 TF**时，用本节点把
/odom 里的位姿广播成 TF，补上 map->odom->base_link 链路里缺的那一段。

输入：
  - nav_msgs/msg/Odometry（默认话题 odom）

输出：
  - TF: <odom_frame> -> <base_frame>（默认 odom -> base_link），时间戳用 msg.header.stamp。

参数：
  - odom_topic | "odom"      | 里程计话题
  - odom_frame | "odom"      | TF 父坐标系（里程计系）
  - base_frame | "base_link" | TF 子坐标系（机器人本体）

注意：
  * 若 msg.header.frame_id 与 odom_frame 不一致、或 child_frame_id 与 base_frame 不一致，
    **只 warn 一次**（避免刷屏）；广播仍按参数指定的坐标系进行。
  * 四元数全零（部分驱动未填）时按 w=1 处理，避免发布非法姿态。
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    """订阅 Odometry，把其 pose 广播为 odom_frame -> base_frame。"""

    def __init__(self) -> None:
        super().__init__("odom_tf_broadcaster")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        odom_topic = self.get_parameter("odom_topic").value

        self._broadcaster = TransformBroadcaster(self)
        self._warned = False

        # 传感器类话题用 best_effort 订阅，兼容 best_effort 与 reliable 发布端。
        self.create_subscription(
            Odometry, odom_topic, self._on_odom, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"odom_tf_broadcaster 启动 | topic={odom_topic} | "
            f"tf={self._odom_frame}->{self._base_frame}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        # 帧名不一致时只 warn 一次（不阻断广播）。
        if not self._warned:
            if msg.header.frame_id and msg.header.frame_id != self._odom_frame:
                self.get_logger().warn(
                    f"odom.header.frame_id={msg.header.frame_id!r} 与参数 odom_frame="
                    f"{self._odom_frame!r} 不一致（仍按参数广播）"
                )
                self._warned = True
            elif msg.child_frame_id and msg.child_frame_id != self._base_frame:
                self.get_logger().warn(
                    f"odom.child_frame_id={msg.child_frame_id!r} 与参数 base_frame="
                    f"{self._base_frame!r} 不一致（仍按参数广播）"
                )
                self._warned = True

        transform = TransformStamped()
        # 时间戳用消息自带 stamp（保持与里程计数据一致）。
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self._odom_frame
        transform.child_frame_id = self._base_frame

        pose = msg.pose.pose
        transform.transform.translation.x = float(pose.position.x)
        transform.transform.translation.y = float(pose.position.y)
        transform.transform.translation.z = float(pose.position.z)

        q = pose.orientation
        # 四元数全零时按 w=1 处理（单位旋转），避免发布非法姿态。
        if q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0:
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0
        else:
            transform.transform.rotation.x = float(q.x)
            transform.transform.rotation.y = float(q.y)
            transform.transform.rotation.z = float(q.z)
            transform.transform.rotation.w = float(q.w)

        self._broadcaster.sendTransform(transform)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = OdomTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
