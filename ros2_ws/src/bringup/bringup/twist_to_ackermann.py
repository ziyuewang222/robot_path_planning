#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Twist -> 阿克曼指令转换节点（入口名 twist_to_ackermann）。

用途：上游 path_follower 输出的是 geometry_msgs/Twist（linear.x / angular.z），而阿克曼
底盘驱动往往接受 ackermann_msgs/AckermannDriveStamped（steering_angle / speed）。本节点
在两者之间做一层无损转换。

输入：
  - geometry_msgs/msg/Twist（默认话题 cmd_vel_raw）

输出：
  - ackermann_msgs/msg/AckermannDriveStamped（默认话题 ackermann_cmd）

参数：
  - twist_topic           | "cmd_vel_raw" | 输入 Twist 话题
  - ackermann_topic       | "ackermann_cmd"| 输出阿克曼指令话题
  - wheelbase             | 0.16          | 轴距 L（m）
  - max_steer             | 0.6           | 前轮最大转角（rad）
  - min_speed_for_steer   | 0.05          | 反解转角时的最小线速度（m/s），防除零
  - frame_id              | "base_link"   | 输出消息 header.frame_id

转换公式（自行车模型）：
  steering = atan(L * angular_z / v)
  * v 用**带符号**线速度；|v| < min_speed_for_steer 时用 copysign(min_speed_for_steer, v)
    替代，防止除零（保持符号，倒车时转角自动反向）。
  * steering 截断到 ±max_steer。
  * 输出 speed = v（原始带符号线速度）。

依赖说明：
  ackermann_msgs 是**可选依赖**（真机底盘用阿克曼接口时才需要）。导入失败时本节点打印明确
  中文错误并以非零码退出，不影响其他节点启动。
"""

from __future__ import annotations

import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# ---- ackermann_msgs 为可选依赖：导入失败时给出清晰中文提示并退出 ----
try:
    from ackermann_msgs.msg import AckermannDriveStamped

    _ACK_OK = True
    _ACK_ERR = ""
except Exception as _exc:  # noqa: BLE001
    _ACK_OK = False
    _ACK_ERR = f"{type(_exc).__name__}: {_exc}"


class TwistToAckermann(Node):
    """订阅 Twist，按自行车模型反解前轮转角，发布 AckermannDriveStamped。"""

    def __init__(self) -> None:
        super().__init__("twist_to_ackermann")
        self.declare_parameter("twist_topic", "cmd_vel_raw")
        self.declare_parameter("ackermann_topic", "ackermann_cmd")
        self.declare_parameter("wheelbase", 0.16)
        self.declare_parameter("max_steer", 0.6)
        self.declare_parameter("min_speed_for_steer", 0.05)
        self.declare_parameter("frame_id", "base_link")

        self._wheelbase = float(self.get_parameter("wheelbase").value)
        self._max_steer = abs(float(self.get_parameter("max_steer").value))
        self._min_speed = abs(float(self.get_parameter("min_speed_for_steer").value))
        self._frame_id = self.get_parameter("frame_id").value
        twist_topic = self.get_parameter("twist_topic").value
        ackermann_topic = self.get_parameter("ackermann_topic").value

        self._pub = self.create_publisher(AckermannDriveStamped, ackermann_topic, 10)
        self.create_subscription(
            Twist, twist_topic, self._on_twist, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"twist_to_ackermann 启动 | in={twist_topic} | out={ackermann_topic} | "
            f"wheelbase={self._wheelbase} | max_steer={self._max_steer} | "
            f"min_speed_for_steer={self._min_speed} | frame={self._frame_id}"
        )

    def _on_twist(self, msg: Twist) -> None:
        v = float(msg.linear.x)
        w = float(msg.angular.z)

        # |v| 过小时用最小速度代替，防止除零（保留符号；v=0 时符号取 +）。
        v_for_steer = v
        if abs(v) < self._min_speed:
            v_for_steer = math.copysign(self._min_speed, v if v != 0.0 else 1.0)

        steering = math.atan(self._wheelbase * w / v_for_steer)
        # 截断到前轮最大转角。
        steering = max(-self._max_steer, min(self._max_steer, steering))

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._frame_id
        out.drive.steering_angle = float(steering)
        out.drive.speed = v
        self._pub.publish(out)


def main(argv=None) -> int:
    # ackermann_msgs 不可用时：打印明确中文错误、以非零码退出（不影响其他节点）。
    if not _ACK_OK:
        print("[FAIL] 无法导入 ackermann_msgs（" + _ACK_ERR + "）。")
        print("       twist_to_ackermann 需要该消息包才能把 Twist 转成 AckermannDriveStamped。")
        print("       真机底盘若接受 AckermannDrive，请安装对应消息包；若接受 Twist，"
              "则不要用 use_ackermann_interface:=true 启动本节点。")
        return 1

    rclpy.init(args=argv)
    node = TwistToAckermann()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
