#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2 端到端验收探针（ROS2 Foxy / rclpy）。

验证链路：frontier_explorer (exploration_path / exploration_goal) + TF map→base_link
          → path_follower → /cmd_vel → 车朝目标跑并在到达后停。

断言：
  A. 指令产生：/cmd_vel 有发布且不是恒零；统计 linear.x 范围。
  B. 车真的动：机器人（TF map→base_link）起点到终点位移 > 0.3 m。
  C. 朝目标收敛：到当前目标（path 末端或 goal）的距离总体下降（弱断言：后 5 样本均值 < 前 5 样本均值）。
  D. 到达即停：窗口内出现 距离<=goal_tolerance 时，随后 /cmd_vel 是否归零、日志是否出现“已到达目标”；
     否则标“未观测到”。
  E. 循环：出现“已到达”后又观测到新的更远目标并使车恢复行进则记录，否则标“未观测到”。

运行：
  python3 step2_check.py [--seconds 45] [--goal-tolerance 0.25]
需先 source ROS 与 install，并在沙箱外执行。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

DEFAULT_LOGS = ["/tmp/step2_follower.log", "/tmp/step2_frontier.log"]


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class Step2Probe(Node):
    def __init__(self) -> None:
        super().__init__("step2_probe")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])

        self._last_cmd: Optional[Tuple[float, float]] = None
        self._cmd_count = 0
        self._path_pts: List[Tuple[float, float]] = []
        self._goal_xy: Optional[Tuple[float, float]] = None
        self._path_count = 0
        self._goal_count = 0

        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 20)
        self.create_subscription(Path, "/exploration_path", self._on_path, 10)
        self.create_subscription(PoseStamped, "/exploration_goal", self._on_goal, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _on_cmd(self, msg: Twist) -> None:
        self._cmd_count += 1
        self._last_cmd = (float(msg.linear.x), float(msg.angular.z))

    def _on_path(self, msg: Path) -> None:
        self._path_count += 1
        self._path_pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal_count += 1
        self._goal_xy = (msg.pose.position.x, msg.pose.position.y)

    def robot_map_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tr = self._tf_buffer.lookup_transform("map", "base_link", Time())
        except TransformException:
            return None
        t = tr.transform.translation
        q = tr.transform.rotation
        return (t.x, t.y, _yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def target(self) -> Optional[Tuple[float, float]]:
        if self._path_pts:
            return self._path_pts[-1]
        return self._goal_xy


def cmd_line_contains(path: str, needle: str) -> bool:
    try:
        with open(path, errors="ignore") as f:
            return needle in f.read()
    except OSError:
        return False


def read_logs_grep(needles: List[str]) -> List[str]:
    hits: List[str] = []
    for lp in DEFAULT_LOGS:
        try:
            with open(lp, errors="ignore") as f:
                for line in f:
                    if any(n in line for n in needles):
                        hits.append(f"[{os.path.basename(lp)}] {line.strip()}")
        except OSError:
            continue
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--goal-tolerance", type=float, default=0.25)
    ap.add_argument("--interval", type=float, default=0.2)
    args = ap.parse_args()

    rclpy.init()
    node = Step2Probe()

    # 采样序列：t, x, y, cmd_lin, cmd_ang, dist
    samples: List[Tuple[float, float, float, float, float, float]] = []
    t0 = time.time()
    last_report = 0.0
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=args.interval)
        pose = node.robot_map_pose()
        if pose is None:
            continue
        x, y, _yaw = pose
        lin, ang = node._last_cmd if node._last_cmd is not None else (0.0, 0.0)
        tgt = node.target()
        dist = math.hypot(tgt[0] - x, tgt[1] - y) if tgt is not None else float("nan")
        samples.append((time.time() - t0, x, y, lin, ang, dist))
        elapsed = time.time() - t0
        if elapsed - last_report >= 10.0:
            last_report = elapsed
            print(
                f"[probe] t={elapsed:5.1f}s cmd=({lin:.3f},{ang:.3f}) "
                f"pose=({x:.2f},{y:.2f}) dist={dist:.2f} path_cnt={node._path_count}"
            )

    print("\n==================== Step 2 断言汇总 ====================")
    n = len(samples)
    print(f"采样点数: {n}  时长≈{args.seconds}s  /cmd_vel 收到={node._cmd_count}  "
          f"/exploration_path 收到={node._path_count}  /exploration_goal 收到={node._goal_count}")
    if n < 3:
        print("[FAIL] 采样过少，无法断言（TF/上游可能未就绪）")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    lin_vals = [s[3] for s in samples]
    nz = [v for v in lin_vals if abs(v) > 1e-3]
    # ---------------- A ----------------
    a_ok = node._cmd_count > 0 and len(nz) > 0
    print("\n=== A. 指令产生 ===")
    print(f"  /cmd_vel 发布条数={node._cmd_count}  非零 linear.x 样本={len(nz)}/{n}")
    print(f"  linear.x 范围=[{min(lin_vals):.4f}, {max(lin_vals):.4f}]")
    print(f"  [{'PASS' if a_ok else 'FAIL'}] {'有非零指令' if a_ok else '指令恒零或未发布'}")

    # ---------------- B ----------------
    x0, y0 = samples[0][1], samples[0][2]
    x1, y1 = samples[-1][1], samples[-1][2]
    disp = math.hypot(x1 - x0, y1 - y0)
    travel = sum(
        math.hypot(samples[i][1] - samples[i - 1][1], samples[i][2] - samples[i - 1][2])
        for i in range(1, n)
    )
    b_ok = disp > 0.3
    print("\n=== B. 车真的动 ===")
    print(f"  起点=({x0:.3f},{y0:.3f}) 终点=({x1:.3f},{y1:.3f})")
    print(f"  位移={disp:.3f} m  累计路程={travel:.3f} m")
    print(f"  [{'PASS' if b_ok else 'FAIL'}] 位移{' > ' if b_ok else ' <= '}0.3 m")

    # ---------------- C ----------------
    dists = [s[5] for s in samples if not math.isnan(s[5])]
    c_ok = False
    first5 = last5 = float("nan")
    if len(dists) >= 10:
        first5 = sum(dists[:5]) / 5.0
        last5 = sum(dists[-5:]) / 5.0
        c_ok = last5 < first5
    print("\n=== C. 朝目标收敛（弱断言） ===")
    print(f"  距离样本数={len(dists)}  前5均值={first5:.3f}  后5均值={last5:.3f}")
    print(f"  [{'PASS' if c_ok else 'FAIL'}] 后5均值{' < ' if c_ok else ' >= '}前5均值")
    print(f"  距离序列(每~1s抽一个): {[round(d,2) for d in dists[::5]]}")

    # ---------------- D ----------------
    tol = args.goal_tolerance
    print("\n=== D. 到达即停 ===")
    reached_idx = next((i for i, s in enumerate(samples) if s[5] <= tol), None)
    d_result = "未观测到"
    if reached_idx is not None:
        after = samples[reached_idx + 1:]
        after_lin = [abs(s[3]) for s in after]
        stopped = len(after) > 0 and max(after_lin) < 1e-3
        log_hit = read_logs_grep(["已到达目标", "到达目标"])
        print(f"  首次 dist<={tol} 于样本 #{reached_idx} (dist={samples[reached_idx][5]:.3f})")
        print(f"  之后 linear.x 最大值={max(after_lin) if after_lin else float('nan'):.4f}  样本数={len(after)}")
        print(f"  path_follower 日志“已到达目标”: {bool(log_hit)}")
        for h in log_hit[:3]:
            print(f"    | {h}")
        d_result = "PASS" if (stopped and log_hit) else ("部分" if stopped else "FAIL")
        print(f"  [{d_result}] {'到达后归零' if stopped else '到达后未归零'}")
    else:
        print(f"  窗口内 dist 最小={min(dists):.3f} > tolerance={tol}，{'未观测到到达'}")
    # ---------------- E ----------------
    print("\n=== E. 循环（到达后新目标恢复行进） ===")
    e_result = "未观测到"
    if reached_idx is not None:
        resumed = any(
            samples[i][5] > tol * 2.0 and abs(samples[i][3]) > 1e-3
            for i in range(reached_idx + 1, n)
        )
        log_hit2 = read_logs_grep(["检测到新前沿目标", "恢复行进"])
        e_result = "PASS" if resumed else "未观测到"
        print(f"  到达后出现更远目标且恢复非零指令: {resumed}")
        print(f"  path_follower 日志“检测到新前沿目标”: {bool(log_hit2)}")
    else:
        print("  窗口内未到达，无法观测循环")
    print(f"  [{e_result}]")

    print("\n==================== 结论 ====================")
    print(f"  A={'PASS' if a_ok else 'FAIL'}  B={'PASS' if b_ok else 'FAIL'}  "
          f"C={'PASS' if c_ok else 'FAIL'}  D={d_result}  E={e_result}")
    ok = a_ok and b_ok and c_ok
    print(f"  核心链路(A/B/C)={'达成' if ok else '未达成'}；D/E 为端到端循环指标")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
