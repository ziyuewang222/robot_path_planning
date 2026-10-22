#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1 验收探针脚本（ROS2 Foxy / rclpy）。

用途：把 Cartographer 的 /map 作为唯一地图源、TF 只有唯一链路
`map -> odom -> base_footprint -> base_link -> velodyne` 这一验收目标，
做成可重复运行的探针。分成 4 个子命令：

    python3 step1_check.py wait   [--timeout 360]   # 轮询等待 /clock /scan /odom 有数据
    python3 step1_check.py drive  [--seconds 15]    # 发布 /cmd_vel 让小车动一小段
    python3 step1_check.py assert [--map-timeout 30]# 跑 A/B/C/D 断言并把证据打印出来
    python3 step1_check.py e      [--timeout 40]    # E：尽力而为地拉起 frontier_explorer 消费端

注意：Foxy 的 `ros2 topic echo` 有块缓冲问题，输出常为空；本脚本用 rclpy 直接读，可靠。
运行前请确保已 `source /opt/ros/foxy/setup.bash` 与 `source install/setup.bash`，
并在沙箱外执行（否则 Gazebo GLX / ~/.ros 日志会失败）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

LEGACY_ROOT = "/home/wzy/robot_path_planning/regional_legacy"


def _qos_transient(depth: int = 10) -> QoSProfile:
    """transient_local 订阅：用于 /map /tf_static 这类会 latching 的话题。"""
    qos = QoSProfile(depth=depth)
    qos.history = QoSHistoryPolicy.KEEP_LAST
    qos.reliability = QoSReliabilityPolicy.RELIABLE
    qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
    return qos


def _qos_volatile(depth: int = 10) -> QoSProfile:
    qos = QoSProfile(depth=depth)
    qos.history = QoSHistoryPolicy.KEEP_LAST
    qos.reliability = QoSReliabilityPolicy.RELIABLE
    qos.durability = QoSDurabilityPolicy.VOLATILE
    return qos


# ------------------------------------------------------------------ 子命令 wait
class WaitNode(Node):
    def __init__(self) -> None:
        super().__init__("step1_wait_probe")
        self.clock_ok = False
        self.scan_ok = False
        self.odom_ok = False
        self.scan_info = None
        self.odom_info = None
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)

    def _on_clock(self, msg: Clock) -> None:
        self.clock_ok = True

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan_ok = True
        self.scan_info = f"ranges={len(msg.ranges)} frame={msg.header.frame_id}"

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_ok = True
        p = msg.pose.pose.position
        self.odom_info = (
            f"frame={msg.header.frame_id} child={msg.child_frame_id} "
            f"pos=({p.x:.3f},{p.y:.3f},{p.z:.3f})"
        )

    def ready(self) -> bool:
        return self.clock_ok and self.scan_ok and self.odom_ok


def cmd_wait(args) -> int:
    rclpy.init()
    node = WaitNode()
    t0 = time.time()
    last_report = 0.0
    while time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        elapsed = time.time() - t0
        if elapsed - last_report >= 10.0:
            last_report = elapsed
            print(
                f"[wait] t={elapsed:6.1f}s clock={node.clock_ok} "
                f"scan={node.scan_ok} odom={node.odom_ok}"
            )
        if node.ready():
            break
    ok = node.ready()
    print("=== [wait] 结果 ===")
    print(f"  /clock 有数据: {node.clock_ok}")
    print(f"  /scan  有数据: {node.scan_ok}  {node.scan_info}")
    print(f"  /odom  有数据: {node.odom_ok}  {node.odom_info}")
    print(f"  ready={ok}  (用时 {time.time() - t0:.1f}s)")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


# ----------------------------------------------------------------- 子命令 drive
def cmd_drive(args) -> int:
    rclpy.init()
    node = Node("step1_drive_probe")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    print(
        f"[drive] 发布 /cmd_vel linear.x={args.speed} angular.z={args.angular} "
        f"持续 {args.seconds}s"
    )
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        tw = Twist()
        tw.linear.x = args.speed
        tw.angular.z = args.angular
        pub.publish(tw)
        rclpy.spin_once(node, timeout_sec=0.1)
    print("[drive] 结束")
    node.destroy_node()
    rclpy.shutdown()
    return 0


# ---------------------------------------------------------------- 子命令 assert
class AssertNode(Node):
    """A/B/C/D 断言节点。"""

    def __init__(self) -> None:
        super().__init__("step1_assert_probe", parameter_overrides=[])
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])

        self.maps = []            # 收到的 /map 消息
        self.tf_dynamic = []      # /tf 的 (parent, child)
        self.tf_static = []       # /tf_static 的 (parent, child)
        self.map_pub_nodes = None
        self.tf_pub_nodes = None
        self.tf_static_pub_nodes = None

        # /map：同时挂 volatile 与 transient_local 两个订阅，命中其一即可
        self.create_subscription(OccupancyGrid, "/map", self._on_map, _qos_volatile(1))
        self.create_subscription(OccupancyGrid, "/map", self._on_map, _qos_transient(1))
        self.create_subscription(TFMessage, "/tf", self._on_tf, 100)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf_static, _qos_transient(100))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.maps.append(msg)

    def _on_tf(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self.tf_dynamic.append((t.header.frame_id, t.child_frame_id))

    def _on_tf_static(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self.tf_static.append((t.header.frame_id, t.child_frame_id))

    def collect_graph_info(self) -> None:
        """查询 /map /tf /tf_static 的发布者节点名列表。"""
        self.map_pub_nodes = [i.node_name for i in self.get_publishers_info_by_topic("/map")]
        self.tf_pub_nodes = [i.node_name for i in self.get_publishers_info_by_topic("/tf")]
        self.tf_static_pub_nodes = [
            i.node_name for i in self.get_publishers_info_by_topic("/tf_static")
        ]

    def spin_until_map(self, timeout: float, want: int = 3) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout and len(self.maps) < want:
            rclpy.spin_once(self, timeout_sec=0.2)

    def spin_seconds(self, seconds: float) -> None:
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.2)


def _check_a(node: AssertNode) -> bool:
    print("\n=== A. /map 语义断言 ===")
    if not node.maps:
        print("  [FAIL] 未收到任何 /map 消息")
        return False
    msg = node.maps[-1]
    info = msg.info
    data = list(msg.data)
    counts = Counter(data)
    vals = sorted(counts.keys())
    allowed = {0, 100, -1}
    print(f"  收到 /map 条数: {len(node.maps)}")
    print(f"  width={info.width} height={info.height} resolution={info.resolution:.4f}")
    print(f"  origin=({info.origin.position.x:.2f},{info.origin.position.y:.2f})")
    print(f"  取值集合: {vals}")
    print(f"  取值计数: {dict(counts)}")

    ok = True
    bad = [v for v in vals if v not in allowed]
    if bad:
        print(f"  [FAIL] 存在非 {{0,100,-1}} 取值: {bad}")
        ok = False
    else:
        print("  [PASS] 取值只属于 {0,100,-1}")
    if info.width > 0 and info.height > 0 and info.resolution > 0.0:
        print("  [PASS] width/height/resolution 合理 (>0)")
    else:
        print("  [FAIL] width/height/resolution 不合理")
        ok = False
    n_free = counts.get(0, 0)
    n_occ = counts.get(100, 0)
    if n_free + n_occ > 0:
        print(f"  [PASS] 至少有非未知栅格: free(0)={n_free} occupied(100)={n_occ}")
    else:
        print("  [FAIL] 全是未知(-1)，没有已探明栅格")
        ok = False
    return ok


def _check_b(node: AssertNode) -> bool:
    print("\n=== B. TF 唯一性断言 ===")
    node.collect_graph_info()
    print(f"  /tf        publishers = {sorted(set(node.tf_pub_nodes))}")
    print(f"  /tf_static publishers = {sorted(set(node.tf_static_pub_nodes))}")
    print(f"  /map       publishers = {sorted(set(node.map_pub_nodes))}")

    dyn = set(node.tf_dynamic)
    sta = set(node.tf_static)
    print(f"  /tf 里出现过的 (parent,child) 去重集合: {sorted(dyn)}")
    print(f"  /tf_static 里出现过的 (parent,child) 去重集合: {sorted(sta)}")

    ok = True
    # (1) map->odom 的动态发布者应是 cartographer_node
    if ("map", "odom") in dyn:
        print("  [PASS] /tf 上存在动态 map->odom")
    else:
        print("  [FAIL] /tf 上未观察到 map->odom")
        ok = False
    if "cartographer_node" in set(node.tf_pub_nodes):
        print("  [PASS] /tf 发布者包含 cartographer_node")
    else:
        print(f"  [FAIL] /tf 发布者不含 cartographer_node: {sorted(set(node.tf_pub_nodes))}")
        ok = False
    # (2) 静态里不应再有 map->odom（gazebo.launch 已删除）
    if ("map", "odom") in sta:
        print("  [FAIL] /tf_static 仍存在 map->odom（静态冲突未消除）")
        ok = False
    else:
        print("  [PASS] /tf_static 无 map->odom（静态冲突已消除）")
    # (3) /map 发布者应是 cartographer_occupancy_grid_node
    if "cartographer_occupancy_grid_node" in set(node.map_pub_nodes):
        print("  [PASS] /map 发布者含 cartographer_occupancy_grid_node")
    else:
        print(f"  [FAIL] /map 发布者异常: {sorted(set(node.map_pub_nodes))}")
        ok = False
    return ok


TF_CHAIN = [
    ("map", "odom"),
    ("odom", "base_footprint"),
    ("base_footprint", "base_link"),
    ("map", "base_link"),
    ("base_link", "velodyne"),
]


def _check_c(node: AssertNode) -> bool:
    print("\n=== C. TF 链路可解（仿真时刻） ===")
    # 先给 listener 一点时间灌入缓存
    node.spin_seconds(2.0)
    ok = True
    for parent, child in TF_CHAIN:
        try:
            tr = node.tf_buffer.lookup_transform(parent, child, Time())
            t = tr.transform.translation
            print(
                f"  [PASS] {parent} -> {child}: "
                f"translation=({t.x:.4f},{t.y:.4f},{t.z:.4f})"
            )
        except Exception as exc:  # noqa: BLE001 - 探针脚本需要打印原始异常
            print(f"  [FAIL] {parent} -> {child}: {type(exc).__name__}: {exc}")
            ok = False
    return ok


def _check_d(node: AssertNode) -> bool:
    print("\n=== D. 冗余检查（/tf_static 上 base_footprint->base_link 是否双权威） ===")
    print(f"  [报告] /tf_static 发布者 = {sorted(set(node.tf_static_pub_nodes))}")
    has_bfl = ("base_footprint", "base_link") in set(node.tf_static)
    print(f"  [报告] /tf_static 上存在 base_footprint->base_link: {has_bfl}")
    pubs = sorted(set(node.tf_static_pub_nodes))
    rsp = "robot_state_publisher" in pubs
    stp = any("static_transform_publisher" in p for p in pubs)
    print(f"  [报告] robot_state_publisher 在 /tf_static 发布者中: {rsp}")
    print(f"  [报告] static_transform_publisher 在 /tf_static 发布者中: {stp}")
    if has_bfl and rsp and stp:
        print("  [报告] => 存在同值双权威（URDF 固定关节 + 静态 TF），同值无害，不判失败")
    else:
        print("  [报告] => 未检出明显双权威")
    # D 只报告不判失败
    return True


def cmd_assert(args) -> int:
    rclpy.init()
    node = AssertNode()
    print("[assert] 等待 /map ...")
    node.spin_until_map(args.map_timeout, want=3)
    node.spin_seconds(2.0)  # 收集 /tf /tf_static
    a = _check_a(node)
    b = _check_b(node)
    d = _check_d(node)
    c = _check_c(node)
    node.destroy_node()
    rclpy.shutdown()

    print("\n==================== 断言汇总 ====================")
    print(f"  A /map 语义      : {'PASS' if a else 'FAIL'}")
    print(f"  B TF 唯一性      : {'PASS' if b else 'FAIL'}")
    print(f"  C TF 链路可解    : {'PASS' if c else 'FAIL'}")
    print(f"  D 冗余(仅报告)   : {'INFO' if d else 'INFO'}")
    print("=================================================")
    return 0 if (a and b and c) else 1


# ------------------------------------------------------------------- 子命令 E
def cmd_e(args) -> int:
    """尽力而为：拉起 frontier_explorer_node 消费 /map，检查其是否发目标/路径且无 TF/地图报错。"""
    log_path = "/tmp/step1_frontier.log"
    cmd = [
        "ros2", "run", "regional_explorer", "frontier_explorer_node",
        "--ros-args",
        "-p", f"algorithm_root:={LEGACY_ROOT}",
        "-p", "map_topic:=map",
        "-p", "robot_frame:=base_link",
        "-p", "global_frame:=map",
    ]
    print(f"[e] 启动消费端: {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)

    rclpy.init()
    node = Node("step1_e_probe")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    got_goal = {"n": 0}
    got_path = {"n": 0}
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    node.create_subscription(PoseStamped, "/exploration_goal", lambda m: got_goal.__setitem__("n", got_goal["n"] + 1), 10)
    node.create_subscription(Path, "/exploration_path", lambda m: got_path.__setitem__("n", got_path["n"] + 1), 10)

    t0 = time.time()
    while time.time() - t0 < args.timeout and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.2)
    time_ok = time.time() - t0 >= args.timeout
    node.destroy_node()
    rclpy.shutdown()

    alive = proc.poll() is None
    if alive:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    log_text = ""
    try:
        with open(log_path) as f:
            log_text = f.read()
    except OSError:
        pass

    print(f"[e] frontier 进程存活: {alive}（timeout={time_ok}）")
    print(f"[e] 收到 /exploration_goal 条数: {got_goal['n']}")
    print(f"[e] 收到 /exploration_path 条数: {got_path['n']}")
    bad_lines = [
        ln for ln in log_text.splitlines()
        if ("Message Filter dropping message" in ln)
        or ("TF" in ln and ("error" in ln.lower() or "fail" in ln.lower() or "exception" in ln.lower()))
        or ("Traceback" in ln)
    ]
    print(f"[e] 日志中可疑行数: {len(bad_lines)}")
    for ln in bad_lines[:20]:
        print(f"    ! {ln}")
    tail = log_text.splitlines()[-15:]
    print("[e] 日志末尾:")
    for ln in tail:
        print(f"    | {ln}")

    ok = (got_goal["n"] > 0 or got_path["n"] > 0) and not bad_lines
    print(f"[e] 结论: {'PASS' if ok else '未通过/需人工判断'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1 验收探针")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_wait = sub.add_parser("wait", help="等待 /clock /scan /odom")
    p_wait.add_argument("--timeout", type=float, default=360.0)
    p_wait.set_defaults(func=cmd_wait)

    p_drive = sub.add_parser("drive", help="发布 /cmd_vel")
    p_drive.add_argument("--seconds", type=float, default=15.0)
    p_drive.add_argument("--speed", type=float, default=0.3)
    p_drive.add_argument("--angular", type=float, default=0.0)
    p_drive.set_defaults(func=cmd_drive)

    p_assert = sub.add_parser("assert", help="跑 A/B/C/D 断言")
    p_assert.add_argument("--map-timeout", type=float, default=30.0)
    p_assert.set_defaults(func=cmd_assert)

    p_e = sub.add_parser("e", help="消费端 frontier_explorer 尽力而为")
    p_e.add_argument("--timeout", type=float, default=40.0)
    p_e.set_defaults(func=cmd_e)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
