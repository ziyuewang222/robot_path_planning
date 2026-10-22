#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时可视化节点：把 Gazebo 里小车的运动与实时建图/选点画到同一张 matplotlib 图上。

设计目标（与用户既有可视化脚本 regional_legacy/real_work/compare_light.py 风格一致）：
  一条命令同时看到
    (1) Gazebo 里小车的真实运动（gzclient 窗口），
    (2) 本节点给出的 2D 视图：实时栅格地图 + 机器人位姿/轨迹 + 探索目标点 + A* 路径。

输入：
  - nav_msgs/msg/OccupancyGrid（默认话题 map）：Cartographer 的实时地图，
    分辨率固定 0.05 m/像素，取值 -1（未知）/ 0..100（连续占用概率），>=50 视为障碍。
  - geometry_msgs/msg/PoseStamped（默认话题 exploration_goal）：探索器选出的目标点。
  - nav_msgs/msg/Path（默认话题 exploration_path）：探索器给出的初步 A* 路径。
  - geometry_msgs/msg/Twist（默认话题 cmd_vel）：路径跟随控制器输出的速度指令（仅用于状态栏显示）。
  - std_msgs/msg/Bool（默认话题 exploration_done，latched/TRANSIENT_LOCAL）：探索完成标志。
  - TF: <global_frame> -> <robot_frame>（默认 map -> base_link）：取机器人当前位姿 (x, y, yaw)。

参数（名称 | 默认值 | 含义）：
  - map_topic           | "map"                 | 订阅的栅格地图话题
  - goal_topic          | "exploration_goal"    | 目标点话题
  - path_topic          | "exploration_path"    | 路径话题
  - cmd_vel_topic       | "cmd_vel"             | 速度指令话题（只读，用于状态显示）
  - done_topic          | "exploration_done"    | 探索完成标志话题（latched）
  - global_frame        | "map"                 | TF 全局坐标系
  - robot_frame         | "base_link"           | TF 机器人坐标系
  - occupied_threshold  | 50                    | >= 该值判为障碍（黑）
  - refresh_hz          | 5.0                   | 重绘频率（Hz）
  - trail_len           | 200                   | 机器人轨迹保留点数（越界丢最早的点）
  - save_once           | ""                    | 非空时进入“只渲染一帧并存图”模式：收到第一帧 /map 后
                          |                       渲染一张 PNG 到该路径、打印确认信息后退出（无显示环境自检用）
  - use_sim_time        | 由 launch 传 True      | 仿真时间（rclpy 内建参数，本节点不再重复声明）

窗口关闭 / 退出：
  - matplotlib 窗口关闭或 Ctrl-C 时优雅退出（不会崩栈）。
  - import matplotlib 失败时用 get_logger().error 打印清晰提示后正常退出。

用法示例（无显示环境自检）：
  MPLBACKEND=Agg python3 -m bringup.exploration_viz --ros-args -p save_once:=/tmp/viz_smoke.png
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

# ---- matplotlib 为可选依赖：导入失败不让节点崩栈，而是给出清晰提示后正常退出 ----
try:
    import matplotlib

    # 无显示环境（未设 DISPLAY 且未显式指定后端）时退回 Agg，
    # 保证仅需出图的 save_once 模式无需 X/GLX 也能渲染 PNG。
    if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
        try:
            matplotlib.use("Agg")
        except Exception:  # noqa: BLE001
            pass

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # 尽量启用系统中文字体，避免标题里的中文显示成方框（找不到则仍走默认字体，仅告警）。
    try:
        matplotlib.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "WenQuanYi Zen Hei",
            "AR PL UMing CN",
            "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001
        pass

    _MATPLOTLIB_OK = True
    _MATPLOTLIB_ERR = ""
except Exception as _exc:  # noqa: BLE001
    _MATPLOTLIB_OK = False
    _MATPLOTLIB_ERR = f"{type(_exc).__name__}: {_exc}"


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """从四元数提取绕 Z 轴的偏航角（rad），避免引入 tf_transformations 依赖。"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ExplorationVisualizer(Node):
    """订阅地图/目标/路径/速度指令/TF，在单个 matplotlib axes 上实时重绘。"""

    def __init__(self) -> None:
        super().__init__("exploration_viz")

        # ---------------------------------------------------------- 参数声明
        self.declare_parameter("map_topic", "map")
        self.declare_parameter("goal_topic", "exploration_goal")
        self.declare_parameter("path_topic", "exploration_path")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("done_topic", "exploration_done")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("refresh_hz", 5.0)
        self.declare_parameter("trail_len", 200)
        self.declare_parameter("save_once", "")

        self._map_topic = self.get_parameter("map_topic").value
        self._goal_topic = self.get_parameter("goal_topic").value
        self._path_topic = self.get_parameter("path_topic").value
        self._cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self._done_topic = self.get_parameter("done_topic").value
        self._global_frame = self.get_parameter("global_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self._refresh_hz = float(self.get_parameter("refresh_hz").value)
        self._trail_len = max(1, int(self.get_parameter("trail_len").value))
        self._save_once = str(self.get_parameter("save_once").value)

        # ---------------------------------------------------------- 运行状态
        self._map_msg: Optional[OccupancyGrid] = None
        # 最近一次收到 /map 的节点时钟时刻（用于状态栏与"上游是否停摆"告警）。
        self._last_map_time: Optional[Time] = None
        # 累计重绘帧数（每渲染一帧 +1，用于直观确认画面在实时刷新）。
        self._frame_count = 0
        self._goal_msg: Optional[PoseStamped] = None
        self._path_msg: Optional[Path] = None
        self._cmd_vel = Twist()  # 最近一帧速度指令（默认全 0）
        self._done = False
        # 机器人位姿 (x, y, yaw) 与轨迹点（浅红细线）。
        self._robot_pose: Optional[Tuple[float, float, float]] = None
        self._trail: List[Tuple[float, float]] = []
        # 地图 extent (xmin, xmax, ymin, ymax) 与已知栅格数，供状态栏与坐标范围使用。
        self._map_extent: Optional[Tuple[float, float, float, float]] = None
        self._known_cells = 0
        # 单帧出图模式的进度标记与退出标记。
        self._saved = False
        self._finished = False
        self._window_closed = False

        # ---------------------------------------------------------- TF
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---------------------------------------------------------- 收发
        self.create_subscription(OccupancyGrid, self._map_topic, self._on_map, 1)
        self.create_subscription(PoseStamped, self._goal_topic, self._on_goal, 10)
        self.create_subscription(Path, self._path_topic, self._on_path, 10)
        self.create_subscription(Twist, self._cmd_vel_topic, self._on_cmd_vel, 10)
        # 完成标志订阅：QoS 必须与发布端匹配（TRANSIENT_LOCAL/latched）。
        self.create_subscription(
            Bool,
            self._done_topic,
            self._on_done,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        # ---------------------------------------------------------- 画布
        # save_once（只出图）模式强制切 Agg 后端：即便当前有显示环境也不会弹窗。
        # 此时尚未创建任何 figure，切换后端安全；失败也不影响（仅告警级别忽略）。
        if self._save_once:
            try:
                matplotlib.use("Agg")
            except Exception:  # noqa: BLE001
                pass
        self._build_figure()

        # ---------------------------------------------------------- 定时重绘
        period = 1.0 / max(self._refresh_hz, 1e-3)
        self.create_timer(period, self._on_refresh)

        self.get_logger().info(
            f"exploration_viz 启动 | map={self._map_topic} | goal={self._goal_topic} | "
            f"path={self._path_topic} | cmd_vel={self._cmd_vel_topic} | done={self._done_topic} | "
            f"tf={self._global_frame}->{self._robot_frame} | refresh={self._refresh_hz}Hz | "
            f"trail={self._trail_len} | save_once={self._save_once or '(实时模式)'}"
        )

    # ------------------------------------------------------------------ 画布构建
    def _build_figure(self) -> None:
        """创建单个 axes（figsize 10x8），并预建所有会被 set_data 复用的图元。"""
        plt.ion()  # 交互模式：与 compare_light.py 风格一致
        self._fig, self._ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        # 窗口关闭事件：置位标记，由定时器优雅退出。
        self._fig.canvas.mpl_connect("close_event", self._on_close_event)

        self._ax.set_aspect("equal")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_title("exploration_viz 启动中…")

        # 栅格地图三类色：0=未知(灰 0.75) 1=自由(白 1.0) 2=障碍(黑 0.0)。
        self._cmap = ListedColormap(["0.75", "1.0", "0.0"])
        # 初始占位 2x2，随后每次 update_map 用 set_data + set_extent 复用，不新建 figure。
        self._image = self._ax.imshow(
            np.zeros((2, 2), dtype=np.int16),
            cmap=self._cmap,
            origin="lower",  # 关键：与 OccupancyGrid（左下角为原点）方向一致
            vmin=0,
            vmax=2,
            interpolation="nearest",
            extent=[0.0, 1.0, 0.0, 1.0],
        )
        # 机器人轨迹（浅红细线）、位置（红点）、朝向（红短线）。
        (self._trail_line,) = self._ax.plot([], [], "-", color="lightcoral", lw=1.0, alpha=0.8)
        (self._robot_mark,) = self._ax.plot([], [], "o", color="red", ms=8)
        (self._robot_arrow,) = self._ax.plot([], [], "-", color="red", lw=2.0)
        # 目标点（绿色五角星）。
        (self._goal_mark,) = self._ax.plot([], [], marker="*", ms=16, color="g", linestyle="None")
        # 路径（蓝色实线）。
        (self._path_line,) = self._ax.plot([], [], "-", color="blue", lw=1.5)

    # ------------------------------------------------------------------ 回调
    def _on_map(self, msg: OccupancyGrid) -> None:
        """缓存最新一帧栅格地图，并记录到达时刻（供上游停摆告警使用）。"""
        self._map_msg = msg
        self._last_map_time = self.get_clock().now()

    def _on_goal(self, msg: PoseStamped) -> None:
        """缓存最新一帧目标点。"""
        self._goal_msg = msg

    def _on_path(self, msg: Path) -> None:
        """缓存最新一帧路径。"""
        self._path_msg = msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        """缓存最新一帧速度指令（仅用于状态栏）。"""
        self._cmd_vel = msg

    def _on_done(self, msg: Bool) -> None:
        """记录探索完成标志。"""
        if msg.data:
            self._done = True

    def _on_close_event(self, _event) -> None:
        """matplotlib 窗口被关闭：置位标记，由定时器触发退出。"""
        self._window_closed = True

    # ---------------------------------------------------------------- 工具函数
    def _lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """查询机器人当前位姿 (x, y, yaw)，TF 未就绪时返回 None。"""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame, self._robot_frame, Time()
            )
        except TransformException:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = _yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
        return (translation.x, translation.y, yaw)

    def _map_age(self) -> Optional[float]:
        """最近一次 /map 距现在多少秒（用节点时钟）；从未收到则返回 None。"""
        if self._last_map_time is None:
            return None
        return (self.get_clock().now() - self._last_map_time).nanoseconds * 1e-9

    def _update_map_image(self) -> None:
        """把最新 /map 转成三类色索引并复用 imshow 图元（set_data + set_extent）。"""
        info = self._map_msg.info
        width = int(info.width)
        height = int(info.height)
        res = float(info.resolution)
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)

        try:
            data = np.asarray(self._map_msg.data, dtype=np.int16).reshape(height, width)
        except ValueError:
            self.get_logger().warn("地图 data 长度与 width/height 不匹配，跳过本帧", throttle_duration_sec=5.0)
            return

        # 0=未知(-1 与异常值) / 1=自由(0..threshold-1) / 2=障碍(>=threshold)。
        disp = np.zeros((height, width), dtype=np.int16)
        known = data >= 0
        disp[known & (data < self._occupied_threshold)] = 1
        disp[data >= self._occupied_threshold] = 2

        self._image.set_data(disp)
        # extent 保证 imshow 与物理坐标一致（origin="lower" + 左下角原点）。
        extent = [ox, ox + width * res, oy, oy + height * res]
        self._image.set_extent(extent)
        self._map_extent = (extent[0], extent[1], extent[2], extent[3])
        self._known_cells = int(np.count_nonzero(disp > 0))

    def _update_robot(self) -> None:
        """更新机器人位置/朝向/轨迹。"""
        pose = self._lookup_robot_pose()
        self._robot_pose = pose
        if pose is None:
            self._robot_mark.set_data([], [])
            self._robot_arrow.set_data([], [])
            return

        x, y, yaw = pose
        # 轨迹：越界丢最早的点。
        self._trail.append((x, y))
        while len(self._trail) > self._trail_len:
            self._trail.pop(0)

        self._trail_line.set_data(
            [p[0] for p in self._trail], [p[1] for p in self._trail]
        )
        self._robot_mark.set_data([x], [y])
        # 朝向用一段短线表示（长度 0.3 m）。
        arrow_len = 0.3
        self._robot_arrow.set_data(
            [x, x + arrow_len * math.cos(yaw)],
            [y, y + arrow_len * math.sin(yaw)],
        )

    def _update_goal(self) -> None:
        """更新目标点（绿色五角星）。"""
        if self._goal_msg is None:
            self._goal_mark.set_data([], [])
            return
        self._goal_mark.set_data(
            [self._goal_msg.pose.position.x], [self._goal_msg.pose.position.y]
        )

    def _update_path(self) -> None:
        """更新路径（蓝色实线）。"""
        if self._path_msg is None or len(self._path_msg.poses) == 0:
            self._path_line.set_data([], [])
            return
        xs = [p.pose.position.x for p in self._path_msg.poses]
        ys = [p.pose.position.y for p in self._path_msg.poses]
        self._path_line.set_data(xs, ys)

    def _apply_view_limits(self) -> None:
        """坐标范围覆盖地图 extent、机器人位置与目标点，并留边距。"""
        xs: List[float] = []
        ys: List[float] = []
        if self._map_extent is not None:
            xs += [self._map_extent[0], self._map_extent[1]]
            ys += [self._map_extent[2], self._map_extent[3]]
        if self._robot_pose is not None:
            xs.append(self._robot_pose[0])
            ys.append(self._robot_pose[1])
        if self._goal_msg is not None:
            xs.append(self._goal_msg.pose.position.x)
            ys.append(self._goal_msg.pose.position.y)
        if not xs:
            return
        margin = 0.5
        xmin, xmax = min(xs) - margin, max(xs) + margin
        ymin, ymax = min(ys) - margin, max(ys) + margin
        if xmax - xmin < 1e-6:
            xmax = xmin + 1.0
        if ymax - ymin < 1e-6:
            ymax = ymin + 1.0
        self._ax.set_xlim(xmin, xmax)
        self._ax.set_ylim(ymin, ymax)

    def _update_title(self) -> None:
        """状态栏：到目标距离、cmd_vel、已知栅格数、是否已收到 done。"""
        if self._robot_pose is not None and self._goal_msg is not None:
            dx = self._goal_msg.pose.position.x - self._robot_pose[0]
            dy = self._goal_msg.pose.position.y - self._robot_pose[1]
            dist_str = f"{math.hypot(dx, dy):.2f} m"
        else:
            dist_str = "--"
        lv = self._cmd_vel.linear.x if self._cmd_vel is not None else 0.0
        av = self._cmd_vel.angular.z if self._cmd_vel is not None else 0.0
        age = self._map_age()
        age_str = f"{age:.1f}s 前" if age is not None else "--"
        self._ax.set_title(
            f"到目标 {dist_str} | cmd_vel v={lv:+.2f} w={av:+.2f} | "
            f"已知栅格 {self._known_cells} | frame={self._frame_count} | "
            f"地图更新 {age_str} | done={'是' if self._done else '否'}"
        )

    def _redraw(self) -> None:
        """按当前缓存数据重绘一帧并刷新画布。

        说明：这里在更新数据期间临时关闭交互态（plt.ioff()），收尾再统一
        draw_idle + flush_events 刷新一次，为的是规避 matplotlib 交互模式
        下、更新数据期间的自动重绘重入：Line2D.set_data 的内部是"先 set_xdata
        再 set_ydata"，在交互态下可能于中途触发一次重绘，若此刻新旧 x/y 长度
        不一致（例如路径点数变化）就有概率触发 broadcast_arrays 的 ValueError
        （本项目冒烟自检中实际观测到过该崩溃）。并非所有画家都会必然触发，但
        这样处理可稳定规避。
        """
        was_interactive = plt.isinteractive()
        if was_interactive:
            plt.ioff()
        self._frame_count += 1
        try:
            if self._map_msg is not None:
                self._update_map_image()
            self._update_robot()
            self._update_goal()
            self._update_path()
            self._update_title()
            self._apply_view_limits()
        finally:
            if was_interactive:
                plt.ion()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    # ------------------------------------------------------------------ 定时刷新
    def _on_refresh(self) -> None:
        # 窗口关闭 -> 优雅退出。
        if self._window_closed:
            self.get_logger().info("可视化窗口已关闭，退出节点")
            self._finished = True
            # 交互模式靠 shutdown 打断 rclpy.spin；save_once 模式由 main 轮询退出。
            if rclpy.ok():
                rclpy.shutdown()
            return

        # save_once 模式：必须等到第一帧 /map 再出图。
        if self._save_once and self._map_msg is None:
            return

        self._redraw()

        if self._save_once and not self._saved:
            self._save_frame()

        # 上游停摆告警（仅实时模式）：超过 5s 未收到新 /map 时明确提醒用户
        # "画面定格"不是可视化节点的问题，而是上游（Gazebo/Cartographer）停了。
        if not self._save_once:
            age = self._map_age()
            if age is not None and age > 5.0:
                self.get_logger().warn(
                    f"已有 {age:.1f}s 未收到新的 /map：上游（Gazebo/Cartographer）可能已停止，"
                    f"画面会定格。",
                    throttle_duration_sec=5.0,
                )

    def _save_frame(self) -> None:
        """只渲染一帧并存成 PNG，打印确认信息后置退出标记。"""
        path = self._save_once
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._fig.savefig(path, dpi=120)
            size = os.path.getsize(path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"[save_once] 保存 PNG 失败: {exc}")
            self._finished = True
            return
        message = f"[save_once] 已渲染并保存首帧到 {path}（{size} 字节）"
        self.get_logger().info(message)
        print(message)
        self._saved = True
        self._finished = True

    # ------------------------------------------------------------------ 生命周期
    @property
    def finished(self) -> bool:
        """save_once 模式是否已完成渲染（供 main 轮询退出）。"""
        return self._finished

    def destroy_node(self) -> bool:
        try:
            if getattr(self, "_fig", None) is not None:
                plt.close(self._fig)
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def main(argv=None) -> None:
    rclpy.init(args=argv)

    # matplotlib 不可用：打印清晰提示后正常退出（不崩栈）。
    if not _MATPLOTLIB_OK:
        fallback = Node("exploration_viz")
        fallback.get_logger().error(
            "无法导入 matplotlib（" + _MATPLOTLIB_ERR + "），可视化节点退出。"
            "请安装 python3-matplotlib；无显示环境可用 MPLBACKEND=Agg 配合 save_once:=<png 路径>。"
        )
        fallback.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return

    node: Optional[ExplorationVisualizer] = None
    try:
        node = ExplorationVisualizer()
    except Exception as exc:  # noqa: BLE001
        # 建窗失败（DISPLAY 存在但 X/GLX 不可用等，TkAgg 会抛 TclError）：
        # 用清晰中文提示后正常退出，不抛整段 traceback。
        report = Node("exploration_viz_report")
        report.get_logger().error(
            f"可视化窗口创建失败（DISPLAY/GLX 可能不可用）：{exc}。"
            "可改用 headless:=true viz:=false 跑闭环，"
            "或用 MPLBACKEND=Agg + save_once:=<png 路径> 静默出图。"
        )
        try:
            report.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()
        return

    try:
        if node._save_once:
            # 单帧出图模式：用 spin_once 轮询，渲染完第一帧即退出（确定性退出，不依赖 spin 语义）。
            while rclpy.ok() and not node.finished:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
