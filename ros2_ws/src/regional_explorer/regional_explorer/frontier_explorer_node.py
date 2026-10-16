from __future__ import annotations

"""区域级几何启发式二维栅格自主探索 ROS2 节点。

输入：
  - nav_msgs/msg/OccupancyGrid（默认话题 map）
  - TF: <global_frame> -> <robot_frame>（默认 map -> base_link）

输出：
  - geometry_msgs/msg/PoseStamped（默认话题 exploration_goal）：下一个前沿目标点
  - nav_msgs/msg/Path（默认话题 exploration_path）：到该目标的初步 A* 路径（不做平滑）

算法本体复用 regional_legacy/real_work/light_qifa.py 与 regional_legacy/core/a_star.py，
通过参数 algorithm_root 指定 regional_legacy 目录（也可用环境变量 REGIONAL_LEGACY_ROOT）。
"""

import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


def _load_algorithm_modules(algorithm_root: str):
    """从 regional_legacy 目录动态导入算法模块，保证与离线调参使用同一份代码。"""
    if not algorithm_root:
        raise RuntimeError(
            "未指定 algorithm_root，请设置 ROS 参数 algorithm_root 或环境变量 "
            "REGIONAL_LEGACY_ROOT 指向 regional_legacy 目录"
        )
    if not os.path.isdir(algorithm_root):
        raise RuntimeError(f"algorithm_root 目录不存在: {algorithm_root}")
    if algorithm_root not in sys.path:
        sys.path.insert(0, algorithm_root)

    from real_work.light_qifa import LightHeuristicConfig, decide_frontier
    from core.a_star import AStarPlanner

    return LightHeuristicConfig, decide_frontier, AStarPlanner


class FrontierExplorerNode(Node):
    """订阅栅格地图与 TF，周期性输出前沿目标点与初步路径。"""

    def __init__(self) -> None:
        super().__init__("frontier_explorer")

        self.declare_parameter("algorithm_root", os.environ.get("REGIONAL_LEGACY_ROOT", ""))
        self.declare_parameter("map_topic", "map")
        self.declare_parameter("goal_topic", "exploration_goal")
        self.declare_parameter("path_topic", "exploration_path")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("replan_rate_hz", 1.0)
        self.declare_parameter("occupied_threshold", 50)

        # 纯几何启发式权重，默认取 real_work 调参得到的较优配置。
        self.declare_parameter("bfs_weight", 8.0)
        self.declare_parameter("turn_weight", 0.35)
        self.declare_parameter("obstacle_weight", 1.0)
        self.declare_parameter("obstacle_safe_distance", 6.0)

        algorithm_root = self.get_parameter("algorithm_root").value
        config_cls, self._decide_frontier, planner_cls = _load_algorithm_modules(algorithm_root)
        self._planner = planner_cls()

        self._heuristic_config = config_cls(
            bfs_weight=float(self.get_parameter("bfs_weight").value),
            turn_weight=float(self.get_parameter("turn_weight").value),
            obstacle_weight=float(self.get_parameter("obstacle_weight").value),
            obstacle_safe_distance=float(self.get_parameter("obstacle_safe_distance").value),
        )

        self._global_frame = self.get_parameter("global_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)

        self._grid_msg: Optional[OccupancyGrid] = None
        self._previous_direction: Optional[Tuple[float, float]] = None
        self._finished_logged = False

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_pub = self.create_publisher(
            PoseStamped, self.get_parameter("goal_topic").value, 10
        )
        self._path_pub = self.create_publisher(
            Path, self.get_parameter("path_topic").value, 10
        )
        self.create_subscription(
            OccupancyGrid, self.get_parameter("map_topic").value, self._on_map, 1
        )

        period = 1.0 / max(float(self.get_parameter("replan_rate_hz").value), 1e-3)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"frontier_explorer 启动 | algorithm_root={algorithm_root} | "
            f"map={self.get_parameter('map_topic').value} | "
            f"goal={self.get_parameter('goal_topic').value} | "
            f"path={self.get_parameter('path_topic').value} | "
            f"tf={self._global_frame}->{self._robot_frame}"
        )

    # ------------------------------------------------------------------ 回调
    def _on_map(self, msg: OccupancyGrid) -> None:
        self._grid_msg = msg

    def _on_timer(self) -> None:
        if self._grid_msg is None:
            return

        robot_grid = self._lookup_robot_grid()
        if robot_grid is None:
            return

        explored_map = self._to_explored_map(self._grid_msg)
        decision = self._decide_frontier(
            explored_map,
            robot_grid,
            previous_direction=self._previous_direction,
            config=self._heuristic_config,
        )

        if decision.target is None:
            if not self._finished_logged:
                self.get_logger().info("未找到可达前沿，当前地图已探索完毕")
                self._finished_logged = True
            return
        self._finished_logged = False

        path_indices = self._planner.plan(explored_map, robot_grid, decision.target)
        if path_indices is None:
            self.get_logger().warn("A* 未能规划到该前沿，本轮跳过")
            return

        goal_world = self._grid_to_world(decision.target)
        # 用机器人->目标方向作为朝向
        robot_world = self._grid_to_world((robot_grid[0], robot_grid[1]))
        yaw = math.atan2(goal_world[1] - robot_world[1], goal_world[0] - robot_world[0])

        self._publish_goal(goal_world, yaw)
        self._publish_path([self._grid_to_world(idx) for idx in path_indices])

        if len(path_indices) >= 2:
            first = path_indices[1]
            self._previous_direction = (
                float(first[0] - robot_grid[0]),
                float(first[1] - robot_grid[1]),
            )

    # ------------------------------------------------------------- 坐标转换
    def _lookup_robot_grid(self) -> Optional[Tuple[int, int]]:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame, self._robot_frame, Time()
            )
        except Exception:  # TF 尚未就绪，等下一轮
            return None

        info = self._grid_msg.info
        col = int(math.floor((transform.transform.translation.x - info.origin.position.x) / info.resolution))
        row = int(math.floor((transform.transform.translation.y - info.origin.position.y) / info.resolution))
        if 0 <= row < info.height and 0 <= col < info.width:
            return (row, col)
        self.get_logger().warn(
            f"机器人位置 ({row}, {col}) 不在地图范围内 ({info.height}x{info.width})"
        )
        return None

    def _grid_to_world(self, grid_index: Tuple[int, int]) -> Tuple[float, float]:
        info = self._grid_msg.info
        row, col = grid_index
        return (
            info.origin.position.x + (col + 0.5) * info.resolution,
            info.origin.position.y + (row + 0.5) * info.resolution,
        )

    def _to_explored_map(self, msg: OccupancyGrid) -> np.ndarray:
        """OccupancyGrid -> int8 栅格：-1 未知 / 0 空地 / 1 障碍，形状 (row, col)。"""
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        explored = np.zeros(data.shape, dtype=np.int8)
        explored[data < 0] = -1
        explored[data >= self._occupied_threshold] = 1
        return explored

    # ------------------------------------------------------------- 话题发布
    def _publish_goal(self, world_xy: Tuple[float, float], yaw: float) -> None:
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self._global_frame
        goal.pose.position.x = float(world_xy[0])
        goal.pose.position.y = float(world_xy[1])
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self._goal_pub.publish(goal)

    def _publish_path(self, world_points: List[Tuple[float, float]]) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self._global_frame
        for x, y in world_points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._path_pub.publish(path)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = FrontierExplorerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
