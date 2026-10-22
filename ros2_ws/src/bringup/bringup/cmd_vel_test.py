#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安全的手动速度测试工具（入口名 cmd_vel_test）。

用途：真机第一次通电、**架空轮子**时，用它在命令行发速度指令，核对
  * 轮子转向与符号约定（正 linear.x 是否前进、正 angular.z 是否左转）；
  * 起步/停止是否正常、有无 NaN、底盘是否响应；
  * 同时在 /odom 上观察实测位姿与 twist，判断里程计是否有响应。

安全约束：
  * **必须显式传 --confirm**，否则只打印"请架空轮子"的警告并退出（返回码 2）。
  * 线性指令绝对值超过 --max-linear（默认 0.2 m/s）时拒绝执行，除非再传 --allow-fast。
  * 任何异常 / Ctrl-C 都在 finally 中发零速，避免车一直跑。

参数：
  --linear          | 0.0    | 线速度指令（m/s）
  --angular         | 0.0    | 角速度指令（rad/s）
  --duration        | 2.0    | 持续时长（s）
  --rate            | 10.0   | 发布频率（Hz）
  --cmd-vel-topic   | cmd_vel| 速度指令话题
  --odom-topic      | odom   | 里程计话题
  --base-frame      | base_link | 机器人本体坐标系（仅用于日志/与 odom 对照）
  --odom-frame      | odom   | 里程计坐标系
  --max-linear      | 0.2    | 允许的最大线速度绝对值（m/s）
  --allow-fast      | 关闭   | 允许超过 max-linear（谨慎）
  --sweep           | 关闭   | 依次做前进/后退/左转/右转，检查符号与转向方向
  --confirm         | 关闭   | 确认已架空轮子（必须）

运行流程（普通模式）：
  3 秒倒计时 → 以 --rate 持续发布 Twist → 同时订阅 /odom 打印实测 x/y/yaw 与 twist
  → 到时发零速并再保持 0.5 秒 → 打印位移与实测速度。

--sweep 模式：
  前进 0.08 m/s 2s → 停 1s → 后退 0.08 2s → 停 1s → 左转 (0.08, +0.3) 2s → 停 1s
  → 右转 (0.08, -0.3) 2s → 停 1s。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """从四元数提取绕 Z 轴偏航角（rad）。"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle: float) -> float:
    """把角度归一化到 [-pi, pi]。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class CmdVelTester(Node):
    """发布 Twist 并按固定频率打印 /odom 实测状态。"""

    def __init__(self, cmd_vel_topic: str, odom_topic: str, rate: float) -> None:
        super().__init__("cmd_vel_test")
        self._pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self._odom: Optional[Odometry] = None
        self._odom_count = 0
        # 传感器类话题用 best_effort 订阅，兼容 reliable 发布端。
        self.create_subscription(Odometry, odom_topic, self._on_odom, qos_profile_sensor_data)
        self._rate = max(float(rate), 1.0)
        # 累计发布条数（用于日志）。
        self._sent = 0

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg
        self._odom_count += 1

    def publish(self, linear: float, angular: float) -> None:
        """发布一帧 Twist（linear.x / angular.z）。"""
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._pub.publish(msg)
        self._sent += 1

    def pose(self) -> Optional[Tuple[float, float, float]]:
        """取当前 /odom 位姿 (x, y, yaw)；未收到返回 None。"""
        if self._odom is None:
            return None
        p = self._odom.pose.pose.position
        q = self._odom.pose.pose.orientation
        return (p.x, p.y, _yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def twist(self) -> Optional[Tuple[float, float, float]]:
        """取当前 /odom twist (vx, vy, wz)；未收到返回 None。"""
        if self._odom is None:
            return None
        t = self._odom.twist.twist
        return (t.linear.x, t.linear.y, t.angular.z)

    def odom_count(self) -> int:
        return self._odom_count


def _run_segment(tester: CmdVelTester, linear: float, angular: float,
                 duration: float, print_every: float = 0.5) -> None:
    """持续发布一段速度指令，并按固定间隔打印 /odom 实测值。"""
    period = 1.0 / tester._rate
    end = time.monotonic() + max(0.0, duration)
    last_print = 0.0
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(tester, timeout_sec=0.0)
        tester.publish(linear, angular)
        now = time.monotonic()
        if now - last_print >= print_every:
            last_print = now
            pose = tester.pose()
            twist = tester.twist()
            if pose is None:
                print(f"  [WARN] 尚未收到 /odom（指令 v={linear:+.3f} w={angular:+.3f}）")
            else:
                vx, vy, wz = twist if twist is not None else (0.0, 0.0, 0.0)
                print(f"  指令 v={linear:+.3f} w={angular:+.3f} | "
                      f"实测 pose=({pose[0]:+.3f},{pose[1]:+.3f}) yaw={pose[2]:+.3f} | "
                      f"twist vx={vx:+.3f} wz={wz:+.3f}")
        time.sleep(max(0.0, period - (time.monotonic() - now)))


def _stop(tester: CmdVelTester, hold_sec: float = 0.5) -> None:
    """发零速并保持一段时间（确保底盘收到停止指令）。"""
    end = time.monotonic() + max(0.0, hold_sec)
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(tester, timeout_sec=0.0)
        tester.publish(0.0, 0.0)
        time.sleep(1.0 / tester._rate)


def _report_displacement(tester: CmdVelTester, start, end) -> None:
    """打印本段测试的位移、航向变化与实测 twist。"""
    if start is None or end is None:
        print("[WARN] /odom 数据不足，无法计算位移与实测速度")
        return
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    dyaw = _normalize_angle(end[2] - start[2])
    print(f"[INFO] 位移 = {dist:.4f} m，航向变化 = {math.degrees(dyaw):+.2f} °")
    print(f"[INFO] 实测 twist: {tester.twist()}")


def _parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cmd_vel_test",
        description="安全的手动速度测试工具（必须 --confirm 且建议架空轮子）",
    )
    parser.add_argument("--linear", type=float, default=0.0, help="线速度指令（m/s）")
    parser.add_argument("--angular", type=float, default=0.0, help="角速度指令（rad/s）")
    parser.add_argument("--duration", type=float, default=2.0, help="持续时长（s）")
    parser.add_argument("--rate", type=float, default=10.0, help="发布频率（Hz）")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel", help="速度指令话题")
    parser.add_argument("--odom-topic", default="odom", help="里程计话题")
    parser.add_argument("--base-frame", default="base_link", help="机器人本体坐标系（日志用）")
    parser.add_argument("--odom-frame", default="odom", help="里程计坐标系（日志用）")
    parser.add_argument("--max-linear", type=float, default=0.2, help="允许的最大线速度绝对值（m/s）")
    parser.add_argument("--allow-fast", action="store_true", help="允许超过 max-linear（谨慎）")
    parser.add_argument("--sweep", action="store_true", help="依次测试前进/后退/左转/右转")
    parser.add_argument("--confirm", action="store_true", help="确认已架空轮子（必须）")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # ---- 安全门 1：必须显式确认 ----
    if not args.confirm:
        print("[WARN] 未指定 --confirm：本工具会直接驱动电机。")
        print("       首次使用请务必**架空轮子**（或垫起驱动轮），确认转向与符号无误后再落地。")
        print("       确认后请加 --confirm 重新运行。已退出。")
        return 2

    # ---- 安全门 2：线速度上限 ----
    max_allowed = abs(float(args.max_linear))
    if abs(float(args.linear)) > max_allowed and not args.allow_fast:
        print(f"[FAIL] 线速度指令 {args.linear:+.3f} m/s 超过上限 {max_allowed:.3f} m/s。")
        print("       如确需更快的测试，请显式加 --allow-fast（风险自负）。已退出。")
        return 3

    rclpy.init()
    tester: Optional[CmdVelTester] = None
    try:
        limit = "（已开启 --allow-fast）" if args.allow_fast else ""
        print(f"[INFO] 目标: {args.cmd_vel_topic} | 观察: {args.odom_topic} | "
              f"frame: {args.odom_frame}->{args.base_frame} | 限速 {max_allowed:.2f} m/s{limit}")
        tester = CmdVelTester(args.cmd_vel_topic, args.odom_topic, args.rate)

        # 3 秒倒计时，给操作者最后确认的机会。
        print("[INFO] 3 秒后开始，请确认小车已架空、手边有急停…")
        for k in (3, 2, 1):
            print(f"  {k}…")
            time.sleep(1.0)

        start_pose = tester.pose()

        if args.sweep:
            seg_list = [
                ("前进", 0.08, 0.0, 2.0),
                ("停", 0.0, 0.0, 1.0),
                ("后退", -0.08, 0.0, 2.0),
                ("停", 0.0, 0.0, 1.0),
                ("左转", 0.08, 0.3, 2.0),
                ("停", 0.0, 0.0, 1.0),
                ("右转", 0.08, -0.3, 2.0),
                ("停", 0.0, 0.0, 1.0),
            ]
            for name, lin, ang, dur in seg_list:
                print(f"\n[INFO] 阶段: {name}  v={lin:+.2f} w={ang:+.2f} {dur:.1f}s")
                _run_segment(tester, lin, ang, dur)
            _stop(tester, 0.5)
        else:
            print(f"\n[INFO] 持续发布 v={args.linear:+.3f} w={args.angular:+.3f} {args.duration:.1f}s")
            _run_segment(tester, args.linear, args.angular, args.duration)
            print("[INFO] 到时，发零速并保持 0.5s…")
            _stop(tester, 0.5)

        end_pose = tester.pose()
        _report_displacement(tester, start_pose, end_pose)

        if tester.odom_count() == 0:
            print("[WARN] 全程未收到 /odom：请检查里程计话题名或驱动是否发布里程计。")
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C：准备发零速并退出")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 运行异常: {type(exc).__name__}: {exc}")
        return 1
    finally:
        # 无论如何都发零速，确保底盘停下（异常路径也不漏）。
        if tester is not None:
            try:
                for _ in range(5):
                    tester.publish(0.0, 0.0)
                    rclpy.spin_once(tester, timeout_sec=0.0)
                    time.sleep(0.05)
            except Exception:  # noqa: BLE001
                pass
            try:
                tester.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
