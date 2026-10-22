#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 LaserScan 的“无回波/超量程”射线改写成 Cartographer 可识别的 missing-data 距离。

背景（2026-10-22）：
  Gazebo ray sensor 与部分真机 LD14 驱动在“量程内没有打到任何物体”时，会发布
  `inf`（或 0）。Cartographer 的 LaserScan 转换只保留
  `range_min <= r <= range_max` 的有限值，`inf` 会被直接丢弃；于是这些方向不会
  写入任何 free/miss 栅格，地图上会残留从机器人向外发散的灰色未知楔形。

策略：
  - 0 < r < slam_max_range：真实回波，原样保留；
  - NaN / inf / r <= 0 / r >= slam_max_range：视为“该方向 slam_max_range 内无物体”，
    改写为 sentinel_range（略大于 slam_max_range）；
  - 同时把消息的 range_max 抬到 >= sentinel_range + 1e-3，保证 Cartographer 的
    LaserScan->PointCloud 转换不会把这些点过滤掉。

  Cartographer 的 TRAJECTORY_BUILDER_2D.missing_data_ray_length（本工程取 7.0 m）
  会把 sentinel 点投影到该长度处并标记为 free，从而在无回波方向清出空地。
  也就是说：本节点负责“让无回波点进入 Cartographer”，lua 负责“清到多远”。

注意：
  - 输出话题只给 Cartographer 使用；path_follower 等避障节点仍应订阅原始 /scan。
  - 本节点不改变传感器量程，仿真雷达仍是 0.15~8.0 m（对齐真机 LD14P）。
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanNoReturnFilter(Node):
    def __init__(self):
        super().__init__("scan_no_return_filter")

        self.declare_parameter("input_topic", "scan")
        self.declare_parameter("output_topic", "scan_for_slam")
        self.declare_parameter("slam_max_range", 8.0)
        self.declare_parameter("missing_range", 7.0)
        self.declare_parameter("sentinel_range", 8.5)
        self.declare_parameter("report_period_sec", 10.0)

        self._input_topic = str(self.get_parameter("input_topic").value)
        self._output_topic = str(self.get_parameter("output_topic").value)
        self._slam_max_range = float(self.get_parameter("slam_max_range").value)
        self._missing_range = float(self.get_parameter("missing_range").value)
        self._sentinel_range = float(self.get_parameter("sentinel_range").value)
        self._report_period_sec = float(
            self.get_parameter("report_period_sec").value
        )

        if not (
            0.0 < self._missing_range < self._slam_max_range < self._sentinel_range
        ):
            raise ValueError(
                "参数必须满足 0 < missing_range < slam_max_range < sentinel_range，"
                f"当前为 missing_range={self._missing_range}, "
                f"slam_max_range={self._slam_max_range}, "
                f"sentinel_range={self._sentinel_range}"
            )

        self._converted_total = 0
        self._messages_total = 0
        self._last_report_time = self.get_clock().now()

        self._pub = self.create_publisher(
            LaserScan, self._output_topic, qos_profile_sensor_data
        )
        self._sub = self.create_subscription(
            LaserScan, self._input_topic, self._on_scan, qos_profile_sensor_data
        )

        self.get_logger().info(
            "无回波预处理已启动："
            f"{self._input_topic} -> {self._output_topic} | "
            f"真实回波范围 (0, {self._slam_max_range:g}) m | "
            f"无回波清空 {self._missing_range:g} m | "
            f"sentinel {self._sentinel_range:g} m"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        # Cartographer 的转换要求 sentinel <= range_max，否则又会把 sentinel 丢掉。
        out.range_max = max(msg.range_max, self._sentinel_range + 1e-3)

        out.ranges = list(msg.ranges)
        converted = 0
        for i, r in enumerate(out.ranges):
            if (
                not math.isfinite(r)
                or r <= 0.0
                or r >= self._slam_max_range
            ):
                out.ranges[i] = self._sentinel_range
                converted += 1

        # 强度数组要么与 ranges 等长，要么置空；长度不一致时置空，避免下游校验失败。
        if len(msg.intensities) == len(msg.ranges):
            out.intensities = list(msg.intensities)
        else:
            out.intensities = []

        self._pub.publish(out)
        self._converted_total += converted
        self._messages_total += 1

        now = self.get_clock().now()
        elapsed = (now - self._last_report_time).nanoseconds * 1e-9
        if self._report_period_sec > 0.0 and elapsed >= self._report_period_sec:
            rays = len(out.ranges)
            mean = self._converted_total / max(self._messages_total, 1)
            self.get_logger().info(
                f"最近 {elapsed:.1f}s：{self._messages_total} 帧，"
                f"平均每帧 {mean:.1f}/{rays} 条无回波射线改写为 "
                f"{self._missing_range:g} m 空地"
            )
            self._converted_total = 0
            self._messages_total = 0
            self._last_report_time = now


def main(args=None):
    rclpy.init(args=args)
    node = ScanNoReturnFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
