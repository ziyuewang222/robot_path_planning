#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真机诊断 / 标定 CLI 工具（入口名 real_diagnostics）。

用途：真机接线后、第一次细致调试时，用数据而不是假设来核对"接口 + 物理参数 + 标定"。
只依赖标准库 + rclpy（不依赖 numpy / scipy）。

子命令：
  - discover          打印全部话题的 type / 发布者数 / 订阅者数；按消息类型自动识别
                      LaserScan / Imu / Odometry / OccupancyGrid / Twist / TwistStamped /
                      AckermannDriveStamped，并给出建议话题名。
  - check --seconds N 对 scan/imu/odom/map/cmd_vel 做链路检查：类型、发布者数量、QoS
                      （reliability/durability）、实际频率、header.frame_id、关键字段；
                      再做 TF 检查（odom->base、base->laser、base->imu、map->base）与
                      /clock 检查；输出 [PASS]/[WARN]/[FAIL] 行；--out report.md 时写
                      markdown 报告与整改建议。
  - scan-calib --seconds N   雷达静止标定：每束距离的均值/标准差、实测最小/最大有效距离、
                      无效点比例、angle_min/max、频率；提示把结果填回 AGENTS 回填表。
  - imu-calib --seconds N    IMU 静止标定：gyro 三轴均值/标准差（零偏+噪声）、accel 均值、
                      用四元数算 yaw 首尾漂移（度/分钟）、频率。
  - odom-check --seconds N   同时采集 /odom.pose 与 TF odom->base，统计位置差最大值/均值、
                      yaw 差最大值；用位姿差分估计实测速度；提示需手推或低速行驶。

公共参数：
  --scan-topic / --imu-topic / --odom-topic / --map-topic / --cmd-vel-topic
  --base-frame / --laser-frame / --imu-frame / --odom-frame / --out

QoS：传感器话题（scan/imu/odom）用 rclpy.qos.qos_profile_sensor_data（best_effort 订阅端
与 reliable 发布端也兼容）；其余（map/cmd_vel）用 depth=10。
所有缺失数据一律优雅地打印 WARN/FAIL，绝不 crash；Ctrl-C 正常退出。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import Buffer, TransformListener

# ------------------------------------------------------------------ 常量
_MSG_LASER = "sensor_msgs/msg/LaserScan"
_MSG_IMU = "sensor_msgs/msg/Imu"
_MSG_ODOM = "nav_msgs/msg/Odometry"
_MSG_MAP = "nav_msgs/msg/OccupancyGrid"
_MSG_TWIST = "geometry_msgs/msg/Twist"
_MSG_TWIST_STAMPED = "geometry_msgs/msg/TwistStamped"
_MSG_ACK = "ackermann_msgs/msg/AckermannDriveStamped"

_TYPE_CATEGORY = {
    _MSG_LASER: "LaserScan",
    _MSG_IMU: "Imu",
    _MSG_ODOM: "Odometry",
    _MSG_MAP: "OccupancyGrid",
    _MSG_TWIST: "Twist",
    _MSG_TWIST_STAMPED: "TwistStamped",
    _MSG_ACK: "AckermannDriveStamped",
}

_CATEGORY_SUGGESTIONS = {
    "LaserScan": "scan",
    "Imu": "imu",
    "Odometry": "odom",
    "OccupancyGrid": "map",
    "Twist": "cmd_vel（差速）/ cmd_vel_raw（阿克曼前级）",
    "TwistStamped": "cmd_vel",
    "AckermannDriveStamped": "ackermann_cmd",
}


# ------------------------------------------------------------------ 小工具
def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """从四元数提取绕 Z 轴偏航角（rad），避免引入 tf_transformations。"""
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


def _enum_label(value) -> str:
    """把 rclpy QoS 枚举值转成可读字符串（RELIABLE / BEST_EFFORT / TRANSIENT_LOCAL ...）。"""
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value)


def _qos_label(info) -> Tuple[str, str]:
    """从 get_publishers_info_by_topic 返回的端点信息里取 (reliability, durability)。"""
    q = getattr(info, "qos_profile", None)
    if q is None:
        return ("?", "?")
    return (_enum_label(getattr(q, "reliability", "?")),
            _enum_label(getattr(q, "durability", "?")))


def _valid_range(raw, msg) -> bool:
    """判断一束测距是否有效（非 None/NaN/inf，且在 [range_min, range_max] 内）。"""
    try:
        if raw is None:
            return False
        value = float(raw)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value):
        return False
    if value <= 0.0:
        return False
    rmin = float(getattr(msg, "range_min", 0.0) or 0.0)
    rmax = float(getattr(msg, "range_max", 0.0) or 0.0)
    if value < rmin:
        return False
    if rmax > 0.0 and value > rmax:
        return False
    return True


def _rate(arrivals: List[float]) -> float:
    """由到达时刻列表估算频率（Hz）。"""
    if len(arrivals) < 2:
        return 0.0
    span = arrivals[-1] - arrivals[0]
    if span <= 1e-9:
        return 0.0
    return (len(arrivals) - 1) / span


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))


# ------------------------------------------------------------------ 报告
class Report:
    """收集 [PASS]/[WARN]/[FAIL]/[INFO] 行，并可选写出 markdown。"""

    def __init__(self, title: str = "") -> None:
        self.title = title
        self.rows: List[Tuple[str, str]] = []
        self.extra: List[str] = []

    def add(self, level: str, text: str) -> None:
        self.rows.append((level, text))
        print(f"[{level}] {text}")

    def add_raw(self, text: str) -> None:
        self.extra.append(text)
        print(text)

    def summary(self) -> Counter:
        return Counter(level for level, _ in self.rows)

    def write_markdown(self, path: str) -> bool:
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {self.title}\n\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                c = self.summary()
                f.write(
                    "汇总: PASS={} WARN={} FAIL={} INFO={}\n\n".format(
                        c.get("PASS", 0), c.get("WARN", 0),
                        c.get("FAIL", 0), c.get("INFO", 0),
                    )
                )
                f.write("## 检查项\n\n")
                for level, text in self.rows:
                    f.write(f"- [{level}] {text}\n")
                if self.extra:
                    f.write("\n## 建议 / 明细\n\n")
                    for line in self.extra:
                        f.write(f"{line}\n")
            print(f"[INFO] 已写出报告: {path}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 写报告失败: {exc}")
            return False


# ------------------------------------------------------------------ 节点与采集
class DiagnosticsNode(Node):
    """诊断用节点：内置 TF Buffer + TransformListener（订阅 /tf、/tf_static）。"""

    def __init__(self) -> None:
        super().__init__("real_diagnostics")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


def _spin_for(node: Node, seconds: float, tick=None) -> None:
    """在最多 seconds 秒内持续 spin_once；tick 为可选的周期性回调（异常被吞掉）。"""
    end = time.monotonic() + max(0.0, seconds)
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if tick is not None:
            try:
                tick()
            except Exception:  # noqa: BLE001
                pass


def _topic_type_map(node: Node) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    try:
        for name, types in node.get_topic_names_and_types():
            out[name] = list(types)
    except Exception:  # noqa: BLE001
        pass
    return out


def _publishers_info(node: Node, topic: str) -> List:
    try:
        return list(node.get_publishers_info_by_topic(topic))
    except Exception:  # noqa: BLE001
        return []


def _count_pubsub(node: Node, topic: str) -> Tuple[int, int]:
    try:
        return node.count_publishers(topic), node.count_subscribers(topic)
    except Exception:  # noqa: BLE001
        return 0, 0


def _lookup_tf(node: DiagnosticsNode, parent: str, child: str, timeout: float = 2.0):
    """在 timeout 秒内尝试查 TF，返回 (transform 或 None, 最后一次错误信息)。"""
    deadline = time.monotonic() + timeout
    err: Optional[str] = None
    while rclpy.ok() and time.monotonic() < deadline:
        try:
            return node.tf_buffer.lookup_transform(parent, child, Time()), None
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            rclpy.spin_once(node, timeout_sec=0.1)
    return None, err


def _report_tf(report: Report, node: DiagnosticsNode, parent: str, child: str) -> None:
    """查一条 TF 并打印 translation/rotation；失败给 WARN。"""
    tf, err = _lookup_tf(node, parent, child)
    if tf is None:
        report.add("WARN", f"TF {parent}->{child} 查询失败：{err}")
        return
    t = tf.transform.translation
    q = tf.transform.rotation
    report.add(
        "PASS",
        f"TF {parent}->{child} 可解：translation=({t.x:.4f},{t.y:.4f},{t.z:.4f}) "
        f"rotation=({q.x:.4f},{q.y:.4f},{q.z:.4f},{q.w:.4f})",
    )


# ------------------------------------------------------------------ discover
def cmd_discover(args) -> int:
    rclpy.init()
    node = DiagnosticsNode()
    try:
        _spin_for(node, 1.0)  # 让 DDS 图发现填充
        type_map = _topic_type_map(node)
        print("=" * 84)
        print("话题清单（topic / type / 发布者数 / 订阅者数）")
        print("=" * 84)
        suggestions: Dict[str, List[str]] = {}
        for topic in sorted(type_map):
            types = ", ".join(type_map[topic])
            npub, nsub = _count_pubsub(node, topic)
            print(f"{topic:<40} {types:<40} pub={npub} sub={nsub}")
            for t in type_map[topic]:
                cat = _TYPE_CATEGORY.get(t)
                if cat:
                    suggestions.setdefault(cat, []).append(topic)
        print()
        print("按消息类型识别（实测话题 -> 建议话题名）：")
        for cat, suggested in _CATEGORY_SUGGESTIONS.items():
            topics = suggestions.get(cat)
            if topics:
                print(f"  {cat:<24} 实测: {topics}  -> 建议: {suggested}")
            else:
                print(f"  {cat:<24} 实测: (未发现)  -> 建议: {suggested}")
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ------------------------------------------------------------------ check
def cmd_check(args) -> int:
    rclpy.init()
    node = DiagnosticsNode()
    report = Report("真机链路检查报告 (check)")
    seconds = float(args.seconds)

    state = {
        "count": {},
        "last": {},
        "arrival": {},
        "scan_ratio_sum": 0.0,
        "scan_ratio_n": 0,
        "gyro_sum": [0.0, 0.0, 0.0],
        "gyro_n": 0,
        "accel_sum": [0.0, 0.0, 0.0],
    }

    def rec(key: str):
        def _cb(msg):
            state["count"][key] = state["count"].get(key, 0) + 1
            state["last"][key] = msg
            state["arrival"].setdefault(key, []).append(time.monotonic())
            if key == "scan":
                ranges = list(getattr(msg, "ranges", []))
                if ranges:
                    valid = sum(1 for r in ranges if _valid_range(r, msg))
                    state["scan_ratio_sum"] += valid / len(ranges)
                    state["scan_ratio_n"] += 1
            elif key == "imu":
                g = msg.angular_velocity
                a = msg.linear_acceleration
                state["gyro_sum"][0] += g.x
                state["gyro_sum"][1] += g.y
                state["gyro_sum"][2] += g.z
                state["accel_sum"][0] += a.x
                state["accel_sum"][1] += a.y
                state["accel_sum"][2] += a.z
                state["gyro_n"] += 1
        return _cb

    try:
        node.create_subscription(LaserScan, args.scan_topic, rec("scan"), qos_profile_sensor_data)
        node.create_subscription(Imu, args.imu_topic, rec("imu"), qos_profile_sensor_data)
        node.create_subscription(Odometry, args.odom_topic, rec("odom"), qos_profile_sensor_data)
        node.create_subscription(OccupancyGrid, args.map_topic, rec("map"), QoSProfile(depth=10))
        node.create_subscription(Twist, args.cmd_vel_topic, rec("cmd_vel"), QoSProfile(depth=10))

        print(f"[INFO] 采集 {seconds:.0f}s（话题: {args.scan_topic} / {args.imu_topic} / "
              f"{args.odom_topic} / {args.map_topic} / {args.cmd_vel_topic}）…")
        _spin_for(node, seconds)

        type_map = _topic_type_map(node)

        # ---- 逐话题：类型 / 发布者 / QoS / 频率 / 关键字段 ----
        _check_topic(report, node, type_map, state, "scan", args.scan_topic, _MSG_LASER)
        _check_topic(report, node, type_map, state, "imu", args.imu_topic, _MSG_IMU)
        _check_topic(report, node, type_map, state, "odom", args.odom_topic, _MSG_ODOM)
        _check_topic(report, node, type_map, state, "map", args.map_topic, _MSG_MAP)
        _check_topic(report, node, type_map, state, "cmd_vel", args.cmd_vel_topic, _MSG_TWIST)

        # scan 关键字段 + 有效点比例
        scan = state["last"].get("scan")
        if scan is not None:
            ratio = (state["scan_ratio_sum"] / state["scan_ratio_n"]
                     if state["scan_ratio_n"] else 0.0)
            report.add("INFO", f"scan 字段: angle_min={scan.angle_min:.4f} angle_max={scan.angle_max:.4f} "
                               f"angle_increment={scan.angle_increment:.6f} range_min={scan.range_min:.3f} "
                               f"range_max={scan.range_max:.3f} 束数={len(scan.ranges)} 有效点比例≈{ratio:.3f}")
            if scan.header.frame_id and scan.header.frame_id != args.laser_frame:
                report.add("WARN", f"scan.frame_id={scan.header.frame_id!r} 与 --laser-frame={args.laser_frame!r} "
                                   f"不一致：建议把 launch 的 laser_frame 改成 {scan.header.frame_id!r}，"
                                   f"或让雷达驱动发布正确的 frame_id。")
        # imu 关键字段
        if state["gyro_n"]:
            n = state["gyro_n"]
            report.add("INFO", "imu 均值: gyro=({:.5f},{:.5f},{:.5f}) rad/s  accel=({:.3f},{:.3f},{:.3f}) m/s^2".format(
                state["gyro_sum"][0] / n, state["gyro_sum"][1] / n, state["gyro_sum"][2] / n,
                state["accel_sum"][0] / n, state["accel_sum"][1] / n, state["accel_sum"][2] / n))
            imu = state["last"].get("imu")
            if imu is not None and imu.header.frame_id and imu.header.frame_id != args.imu_frame:
                report.add("WARN", f"imu.frame_id={imu.header.frame_id!r} 与 --imu-frame={args.imu_frame!r} 不一致。")
        # odom 关键字段
        odom = state["last"].get("odom")
        if odom is not None:
            p = odom.pose.pose.position
            t = odom.twist.twist
            report.add("INFO", f"odom 最新: pose=({p.x:.3f},{p.y:.3f}) "
                               f"twist=({t.linear.x:.3f},{t.angular.z:.3f}) "
                               f"frame_id={odom.header.frame_id!r} child={odom.child_frame_id!r}")
        # map 关键字段
        ocmap = state["last"].get("map")
        if ocmap is not None:
            report.add("INFO", f"map 最新: {ocmap.info.width}x{ocmap.info.height} "
                               f"resolution={ocmap.info.resolution:.4f} m/格")
            if abs(ocmap.info.resolution - 0.05) > 1e-6:
                report.add("WARN", f"map resolution={ocmap.info.resolution} ！= 0.05 m/格（真机硬约束）")

        # ---- TF 检查 ----
        print("[INFO] TF 检查…")
        _report_tf(report, node, args.odom_frame, args.base_frame)
        _report_tf(report, node, args.base_frame, args.laser_frame)
        _report_tf(report, node, args.base_frame, args.imu_frame)
        _report_tf(report, node, "map", args.base_frame)

        # ---- /clock 检查 ----
        if "/clock" in type_map:
            report.add("WARN", "检测到 /clock 话题：说明可能有节点在用仿真时间。真机全链路应为 "
                               "use_sim_time=false，请确认没有启动 Gazebo/rosbag 时钟源。")
        else:
            report.add("PASS", "未发现 /clock（真机应为真实时间，符合预期）")

        c = report.summary()
        report.add_raw(f"\n检查汇总：PASS={c.get('PASS', 0)} WARN={c.get('WARN', 0)} "
                       f"FAIL={c.get('FAIL', 0)} INFO={c.get('INFO', 0)}")
        if args.out:
            report.write_markdown(args.out)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _check_topic(report: Report, node: DiagnosticsNode, type_map, state,
                 key: str, topic: str, expected_type: str) -> None:
    """对单个话题做类型/发布者/QoS/频率检查。"""
    actual = type_map.get(topic, [])
    pubinfos = _publishers_info(node, topic)
    npub, nsub = _count_pubsub(node, topic)
    if not actual:
        report.add("FAIL", f"{key}: 话题 {topic} 不存在（既无发布者也未注册）")
        return
    if expected_type in actual:
        report.add("PASS", f"{key}: 话题 {topic} 类型 {expected_type}")
    else:
        report.add("WARN", f"{key}: 话题 {topic} 实际类型 {actual}，期望 {expected_type}")
    if not pubinfos:
        report.add("WARN", f"{key}: 话题 {topic} 未发现发布者（pub={npub} sub={nsub}）")
    else:
        for info in pubinfos:
            rel, dur = _qos_label(info)
            report.add("INFO", f"{key}: 发布者 {getattr(info, 'node_name', '?')} "
                               f"QoS reliability={rel} durability={dur}")
    cnt = state["count"].get(key, 0)
    if cnt == 0:
        report.add("FAIL", f"{key}: {topic} 在采样窗口内未收到任何消息")
    else:
        report.add("PASS", f"{key}: 收到 {cnt} 条，实测频率 ~{_rate(state['arrival'][key]):.2f} Hz")


# ------------------------------------------------------------------ scan-calib
def cmd_scan_calib(args) -> int:
    rclpy.init()
    node = DiagnosticsNode()
    report = Report("雷达静止标定报告 (scan-calib)")
    scans: List[LaserScan] = []
    arrivals: List[float] = []

    def cb(msg):
        scans.append(msg)
        arrivals.append(time.monotonic())

    try:
        node.create_subscription(LaserScan, args.scan_topic, cb, qos_profile_sensor_data)
        print(f"[INFO] 采集 {float(args.seconds):.0f}s 的 {args.scan_topic}，请保持小车静止…")
        _spin_for(node, float(args.seconds))

        if not scans:
            report.add("FAIL", f"{args.scan_topic} 未收到任何 LaserScan（检查驱动/话题名）")
            if args.out:
                report.write_markdown(args.out)
            return 1

        first = scans[0]
        n = len(first.ranges)
        report.add("PASS", f"{args.scan_topic}: 收到 {len(scans)} 帧，每帧 {n} 束，"
                           f"频率 ~{_rate(arrivals):.2f} Hz")
        report.add("INFO", f"angle_min={first.angle_min:.4f} rad angle_max={first.angle_max:.4f} rad "
                           f"angle_increment={first.angle_increment:.6f} rad "
                           f"frame_id={first.header.frame_id!r}")
        report.add("INFO", f"range_min={first.range_min:.3f} m range_max={first.range_max:.3f} m")

        sums = [0.0] * n
        sumsq = [0.0] * n
        counts = [0] * n
        invalid = 0
        total = 0
        dmin: Optional[float] = None
        dmax: Optional[float] = None
        for msg in scans:
            for i, r in enumerate(msg.ranges):
                total += 1
                if _valid_range(r, msg):
                    value = float(r)
                    if dmin is None or value < dmin:
                        dmin = value
                    if dmax is None or value > dmax:
                        dmax = value
                    if i < n:
                        sums[i] += value
                        sumsq[i] += value * value
                        counts[i] += 1
                else:
                    invalid += 1

        beam_means: List[Optional[float]] = []
        beam_stds: List[Optional[float]] = []
        for i in range(n):
            c = counts[i]
            if c <= 0:
                beam_means.append(None)
                beam_stds.append(None)
                continue
            mean = sums[i] / c
            var = max(0.0, sumsq[i] / c - mean * mean)
            beam_means.append(mean)
            beam_stds.append(math.sqrt(var))

        valid_beams = sum(1 for c in counts if c > 0)
        std_samples = [s for s in beam_stds if s is not None]
        report.add("PASS", f"有效束数 {valid_beams}/{n}（从未收到有效测距的束 {n - valid_beams}）")
        report.add("INFO", f"实测有效距离范围 min={dmin if dmin is None else round(dmin, 3)} m "
                           f"max={dmax if dmax is None else round(dmax, 3)} m")
        report.add("INFO", f"无效点比例 = {invalid / total if total else 0.0:.4f}"
                           f"（{invalid}/{total}）")
        report.add("INFO", f"单束噪声（标准差）均值 = {_mean(std_samples):.4f} m，"
                           f"最大 = {max(std_samples) if std_samples else 0.0:.4f} m")
        if dmin is not None and (dmin < 0.10 or dmin > 0.30):
            report.add("WARN", f"实测最小有效距离 {dmin:.3f} m 与真机基准 0.15 m 偏差较大，"
                               f"请回填 AGENTS 回填表的 min_range。")
        if dmax is not None and abs(dmax - 8.0) > 1.5:
            report.add("WARN", f"实测最大有效距离 {dmax:.3f} m 与真机基准 8.0 m 偏差较大，"
                               f"请回填 AGENTS 回填表的 max_range。")
        report.add_raw("\n提示：把上面的 min/max 有效距离、angle_min/max、频率填入 AGENTS.md 第 10 节回填表；"
                       "并据此核对 launch 的 laser_x/y/z 与 laser_roll/pitch/yaw、以及 cartographer_real.lua 的 min/max_range。")
        if args.out:
            report.write_markdown(args.out)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ------------------------------------------------------------------ imu-calib
def cmd_imu_calib(args) -> int:
    rclpy.init()
    node = DiagnosticsNode()
    report = Report("IMU 静止标定报告 (imu-calib)")
    msgs: List[Imu] = []
    arrivals: List[float] = []

    def cb(msg):
        msgs.append(msg)
        arrivals.append(time.monotonic())

    try:
        node.create_subscription(Imu, args.imu_topic, cb, qos_profile_sensor_data)
        print(f"[INFO] 采集 {float(args.seconds):.0f}s 的 {args.imu_topic}，请保持小车静止…")
        _spin_for(node, float(args.seconds))

        if not msgs:
            report.add("FAIL", f"{args.imu_topic} 未收到任何 Imu（检查驱动/话题名）")
            if args.out:
                report.write_markdown(args.out)
            return 1

        gx = [m.angular_velocity.x for m in msgs]
        gy = [m.angular_velocity.y for m in msgs]
        gz = [m.angular_velocity.z for m in msgs]
        ax = [m.linear_acceleration.x for m in msgs]
        ay = [m.linear_acceleration.y for m in msgs]
        az = [m.linear_acceleration.z for m in msgs]

        report.add("PASS", f"{args.imu_topic}: 收到 {len(msgs)} 帧，频率 ~{_rate(arrivals):.2f} Hz，"
                           f"frame_id={msgs[0].header.frame_id!r}")
        report.add("INFO", f"gyro 零偏/均值 = ({_mean(gx):+.5f}, {_mean(gy):+.5f}, {_mean(gz):+.5f}) rad/s")
        report.add("INFO", f"gyro 噪声/标准差 = ({_std(gx):.5f}, {_std(gy):.5f}, {_std(gz):.5f}) rad/s")
        report.add("INFO", f"accel 均值 = ({_mean(ax):+.3f}, {_mean(ay):+.3f}, {_mean(az):+.3f}) m/s^2")

        # yaw 首尾漂移：用四元数分别算首尾的 yaw，再除以时长（度/分钟）。
        q_first = msgs[0].orientation
        q_last = msgs[-1].orientation
        zero_first = (q_first.x == 0.0 and q_first.y == 0.0 and q_first.z == 0.0 and q_first.w == 0.0)
        zero_last = (q_last.x == 0.0 and q_last.y == 0.0 and q_last.z == 0.0 and q_last.w == 0.0)
        dt = arrivals[-1] - arrivals[0]
        if zero_first or zero_last:
            report.add("WARN", "IMU 未提供有效姿态四元数（全零）：无法计算 yaw 漂移；"
                               "可改用角速度积分自行估算，或确认驱动是否输出 orientation。")
        elif dt <= 1e-6:
            report.add("WARN", "采集时长过短，无法计算 yaw 漂移。")
        else:
            yaw_first = _yaw_from_quaternion(q_first.x, q_first.y, q_first.z, q_first.w)
            yaw_last = _yaw_from_quaternion(q_last.x, q_last.y, q_last.z, q_last.w)
            drift_rad = _normalize_angle(yaw_last - yaw_first)
            drift_deg_per_min = math.degrees(drift_rad) / (dt / 60.0)
            report.add("INFO", f"yaw 首尾漂移 = {math.degrees(drift_rad):+.3f} ° / {dt:.1f}s "
                               f"（≈ {drift_deg_per_min:+.3f} °/min）")
        report.add_raw("\n提示：把 gyro 零偏、噪声标准差、yaw 漂移填入 AGENTS.md 回填表；"
                       "并核对 IMU 安装朝向（原地转 90° 看 yaw 增/减、抬车头看 pitch 符号）。")
        if args.out:
            report.write_markdown(args.out)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ------------------------------------------------------------------ odom-check
def cmd_odom_check(args) -> int:
    rclpy.init()
    node = DiagnosticsNode()
    report = Report("里程计一致性检查报告 (odom-check)")

    odom_samples: List[Tuple[float, float, float, float]] = []  # (t, x, y, yaw)
    tf_samples: List[Tuple[float, float, float, float]] = []
    arrivals: List[float] = []

    def cb(msg):
        t = time.monotonic()
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        odom_samples.append((t, p.x, p.y, _yaw_from_quaternion(q.x, q.y, q.z, q.w)))
        arrivals.append(t)

    def tick():
        try:
            tf = node.tf_buffer.lookup_transform(args.odom_frame, args.base_frame, Time())
        except Exception:  # noqa: BLE001
            return
        tr = tf.transform.translation
        q = tf.transform.rotation
        tf_samples.append((time.monotonic(), tr.x, tr.y,
                           _yaw_from_quaternion(q.x, q.y, q.z, q.w)))

    try:
        node.create_subscription(Odometry, args.odom_topic, cb, qos_profile_sensor_data)
        print(f"[INFO] 采集 {float(args.seconds):.0f}s：同时对比 /odom.pose 与 TF "
              f"{args.odom_frame}->{args.base_frame}。可**手推**小车或低速行驶。")
        _spin_for(node, float(args.seconds), tick=tick)

        if not odom_samples:
            report.add("FAIL", f"{args.odom_topic} 未收到任何 Odometry")
            if args.out:
                report.write_markdown(args.out)
            return 1
        report.add("PASS", f"{args.odom_topic}: 收到 {len(odom_samples)} 帧，频率 ~{_rate(arrivals):.2f} Hz")

        if not tf_samples:
            report.add("FAIL", f"未采到 TF {args.odom_frame}->{args.base_frame}（驱动可能只发 /odom 不发 TF；"
                               f"可置 publish_odom_tf:=true 用 odom_tf_broadcaster 补上）")
        else:
            # 对每个 odom 样本，找时间最近的 TF 样本，比较位置与 yaw。
            pos_diffs: List[float] = []
            yaw_diffs: List[float] = []
            for t, x, y, yaw in odom_samples:
                nearest = min(tf_samples, key=lambda s: abs(s[0] - t))
                _, tx, ty, tyaw = nearest
                pos_diffs.append(math.hypot(tx - x, ty - y))
                yaw_diffs.append(abs(_normalize_angle(tyaw - yaw)))
            report.add("INFO", f"/odom.pose 与 TF 位置差：max={max(pos_diffs):.4f} m "
                               f"mean={_mean(pos_diffs):.4f} m（样本 {len(pos_diffs)}）")
            report.add("INFO", f"/odom.pose 与 TF yaw 差：max={math.degrees(max(yaw_diffs)):.3f} °")
            if max(pos_diffs) > 0.05 or math.degrees(max(yaw_diffs)) > 5.0:
                report.add("WARN", "两者不一致（>0.05 m 或 >5°）：里程计与 TF 可能不同步，"
                                   "暂不建议开启 use_odometry。")

        # 用位姿差分估计实测速度（手推/低速行驶时才有意义）。
        if len(odom_samples) >= 2:
            t0, x0, y0, _ = odom_samples[0]
            t1, x1, y1, _ = odom_samples[-1]
            dt = t1 - t0
            if dt > 1e-6:
                dist = math.hypot(x1 - x0, y1 - y0)
                report.add("INFO", f"位姿差分估计：净位移 {dist:.4f} m / {dt:.1f}s "
                                   f"≈ {dist / dt:.4f} m/s")
        report.add_raw("\n提示：本项需**手推**小车或**低速行驶**才有意义；若位置差/yaw 差偏大，"
                       "请先排查 TF 与里程计不同步，再决定 use_odometry 开关。")
        if args.out:
            report.write_markdown(args.out)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ------------------------------------------------------------------ CLI 入口
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scan-topic", default="scan")
    p.add_argument("--imu-topic", default="imu")
    p.add_argument("--odom-topic", default="odom")
    p.add_argument("--map-topic", default="map")
    p.add_argument("--cmd-vel-topic", default="cmd_vel")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--laser-frame", default="laser")
    p.add_argument("--imu-frame", default="imu_link")
    p.add_argument("--odom-frame", default="odom")
    p.add_argument("--out", default="", help="非空时把报告写成 markdown 到该路径")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="real_diagnostics",
        description="真机诊断 / 标定 CLI（话题、TF、雷达、IMU、里程计）",
    )
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("discover", help="列出所有话题与类型，并给出建议话题名")
    _add_common(d)

    c = sub.add_parser("check", help="链路检查：类型/QoS/频率/字段/TF/clock")
    _add_common(c)
    c.add_argument("--seconds", type=float, default=10.0)

    sc = sub.add_parser("scan-calib", help="雷达静止标定")
    _add_common(sc)
    sc.add_argument("--seconds", type=float, default=60.0)

    ic = sub.add_parser("imu-calib", help="IMU 静止标定")
    _add_common(ic)
    ic.add_argument("--seconds", type=float, default=60.0)

    oc = sub.add_parser("odom-check", help="里程计与 TF 一致性检查")
    _add_common(oc)
    oc.add_argument("--seconds", type=float, default=30.0)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        if args.command == "discover":
            return cmd_discover(args)
        if args.command == "check":
            return cmd_check(args)
        if args.command == "scan-calib":
            return cmd_scan_calib(args)
        if args.command == "imu-calib":
            return cmd_imu_calib(args)
        if args.command == "odom-check":
            return cmd_odom_check(args)
        parser.print_help()
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断（Ctrl-C），退出")
        return 0


if __name__ == "__main__":
    sys.exit(main())
