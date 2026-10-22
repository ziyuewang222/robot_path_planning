from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# 到最近障碍的距离变换统一由 core/obstacle_distance.py 提供（单一实现）：
# A* 膨胀与前沿候选过滤必须使用同一套度量，否则会出现"决策放行、A* 阻挡"的重叠。
# 这里保留 _chamfer_distance_transform 的导入，兼容外部（如离线自检脚本）按旧名字引用；
# _distance_to_obstacle 保留为薄包装，避免其它调用点报错。
from core.obstacle_distance import (
    _chamfer_distance_transform,
    distance_to_obstacle as _shared_distance_to_obstacle,
)


@dataclass(frozen=True)
class LightHeuristicConfig:
    """权重和安全距离，距离单位为栅格（真机基准 0.25 m = 5 格）。"""

    bfs_weight: float = 1.0
    turn_weight: float = 0.35
    obstacle_weight: float = 2.0
    obstacle_safe_distance: float = 5.0
    # 与机器人的直线（欧氏）栅格距离下限：必须 >= 该值才纳入候选，避免选中贴在机器人
    # 身上的退化前沿（单位：栅格）。这类"贴脸前沿"会被下游控制端第一帧即判为"已到达"，
    # 导致车不走、地图不增长、上游反复重发同一近目标直到触顶假完成。
    min_frontier_distance: float = 10.0


@dataclass(frozen=True)
class FrontierDecision:
    """纯几何 frontier 决策及其可解释代价。"""

    target: Optional[Tuple[int, int]]
    frontier_mask: np.ndarray
    score_map: np.ndarray
    distance_cost: np.ndarray
    turn_cost: np.ndarray
    obstacle_cost: np.ndarray


def build_frontier_mask(explored_map: np.ndarray) -> np.ndarray:
    """返回与未知区域四邻接的已知自由栅格。"""
    if explored_map.ndim != 2:
        raise ValueError("explored_map 必须是二维数组")

    free_space = explored_map == 0
    unknown = explored_map == -1
    adjacent_unknown = np.zeros_like(unknown, dtype=bool)
    adjacent_unknown[:-1, :] |= unknown[1:, :]
    adjacent_unknown[1:, :] |= unknown[:-1, :]
    adjacent_unknown[:, :-1] |= unknown[:, 1:]
    adjacent_unknown[:, 1:] |= unknown[:, :-1]
    return (free_space & adjacent_unknown).astype(np.uint8)


def _reachable_distances(explored_map: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """只在已知自由栅格上计算 BFS 距离。"""
    h, w = explored_map.shape
    sx, sy = start
    distances = np.full((h, w), -1.0, dtype=np.float32)
    if not (0 <= sx < h and 0 <= sy < w):
        raise ValueError("robot_pos 超出地图范围")
    if explored_map[sx, sy] != 0:
        return distances

    distances[sx, sy] = 0.0
    queue = [(sx, sy)]
    head = 0
    while head < len(queue):
        x, y = queue[head]
        head += 1
        next_distance = distances[x, y] + 1.0
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                if explored_map[nx, ny] == 0 and distances[nx, ny] < 0:
                    distances[nx, ny] = next_distance
                    queue.append((nx, ny))
    return distances


def _turn_cost(
    robot_pos: Sequence[int],
    frontier_coords: np.ndarray,
    previous_direction: Optional[Sequence[float]],
) -> np.ndarray:
    if previous_direction is None or frontier_coords.size == 0:
        return np.zeros(len(frontier_coords), dtype=np.float32)

    previous = np.asarray(previous_direction, dtype=np.float32)
    previous_norm = float(np.linalg.norm(previous))
    if previous.shape != (2,) or previous_norm <= 1e-6:
        return np.zeros(len(frontier_coords), dtype=np.float32)

    previous /= previous_norm
    robot = np.asarray(robot_pos, dtype=np.float32)
    vectors = frontier_coords.astype(np.float32) - robot
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-6
    costs = np.ones(len(frontier_coords), dtype=np.float32)
    if np.any(valid):
        directions = vectors[valid] / norms[valid, None]
        cosine = np.clip(directions @ previous, -1.0, 1.0)
        costs[valid] = (1.0 - cosine) / 2.0
    return costs


def _distance_to_obstacle(obstacle_mask: np.ndarray) -> np.ndarray:
    """薄包装：委托到 core.obstacle_distance.distance_to_obstacle。

    保留旧函数名（及 `_chamfer_distance_transform` 的模块级别名）只为兼容既有调用点；
    实际算法只有 core/obstacle_distance.py 一份，A* 膨胀与这里共用同一套距离度量。
    """
    return _shared_distance_to_obstacle(obstacle_mask)


def _obstacle_cost(
    explored_map: np.ndarray,
    frontier_coords: np.ndarray,
    safe_distance: float,
    distance_to_obstacle: Optional[np.ndarray] = None,
) -> np.ndarray:
    """候选前沿到最近障碍的接近度代价（越近越大）。

    `distance_to_obstacle` 为可选的"到最近障碍距离"数组（单位：栅格）：调用方若已算好
    （如 decide_frontier 的硬过滤已调用过 _distance_to_obstacle），可直接传入复用，避免
    重复计算。本机没有 scipy 时 _distance_to_obstacle 会退回纯 Python 双层循环，在 ~69
    万格的 /map 上开销可观，故复用意义明确。为 None（默认）时才自行计算，保证向后兼容。
    """
    if frontier_coords.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if safe_distance <= 0:
        raise ValueError("obstacle_safe_distance 必须大于 0")

    obstacle_mask = explored_map == 1
    if not np.any(obstacle_mask):
        return np.zeros(len(frontier_coords), dtype=np.float32)

    # 复用调用方已算好的距离数组；未提供（None）时才自行计算（向后兼容）。
    if distance_to_obstacle is None:
        distance_to_obstacle = _distance_to_obstacle(obstacle_mask)
    fx = frontier_coords[:, 0]
    fy = frontier_coords[:, 1]
    distance = distance_to_obstacle[fx, fy].astype(np.float32)
    proximity = np.clip(1.0 - distance / float(safe_distance), 0.0, 1.0)
    return proximity ** 2


def decide_frontier(
    explored_map: np.ndarray,
    robot_pos: Sequence[int],
    previous_direction: Optional[Sequence[float]] = None,
    config: Optional[LightHeuristicConfig] = None,
    exclude_mask: Optional[np.ndarray] = None,
    distance_to_obstacle: Optional[np.ndarray] = None,
) -> FrontierDecision:
    """根据二维探索地图选择一个可达 frontier。

    exclude_mask：可选的前沿排除掩码（形状与 explored_map 相同，True 表示该格不作为候选）。
    用于上游"目标推进不了就放弃、换一个前沿"的临时黑名单过滤。为 None（默认）时行为
    完全不变，离线脚本不传即无影响（向后兼容）。

    distance_to_obstacle：可选的**预计算**距离场（每个格到最近已知障碍的栅格距离，形状与
    explored_map 相同）。ROS2 侧一轮决策里同时要用它做阿克曼轨迹生成与 A* 膨胀，本机没有
    scipy 时纯 Python chamfer 对 700x700 要约 1.2s，因此由调用方算一次后传进来复用；
    传 None（默认）时本函数自己算，离线脚本行为不变。
    """
    if explored_map.ndim != 2:
        raise ValueError("explored_map 必须是二维数组")
    if explored_map.size == 0:
        raise ValueError("explored_map 不能为空")
    if exclude_mask is not None and exclude_mask.shape != explored_map.shape:
        raise ValueError("exclude_mask 形状必须与 explored_map 相同")

    config = config or LightHeuristicConfig()
    frontier_mask = build_frontier_mask(explored_map)
    distances = _reachable_distances(
        explored_map,
        (int(robot_pos[0]), int(robot_pos[1])),
    )
    valid_mask = (frontier_mask == 1) & (distances >= 0)
    # 黑名单过滤：只在调用方显式传入 exclude_mask 时生效；None 时保持原行为不变。
    if exclude_mask is not None:
        valid_mask = valid_mask & ~exclude_mask.astype(bool)
    if config.min_frontier_distance > 0.0:
        # 剔除"贴在机器人身上"的退化前沿：与机器人的直线（欧氏）栅格距离必须 >= 阈值。
        # 用欧氏距离是因为下游 path_follower 的到达判定就是欧氏距离，两者必须同度量。
        rows_idx, cols_idx = np.indices(explored_map.shape)
        euclid_cells = np.hypot(
            rows_idx.astype(np.float32) - float(robot_pos[0]),
            cols_idx.astype(np.float32) - float(robot_pos[1]),
        )
        valid_mask = valid_mask & (euclid_cells >= float(config.min_frontier_distance))
    # 距最近障碍的距离数组（栅格）：优先复用调用方预计算的 distance_to_obstacle，
    # 否则在这里按需计算；供硬过滤与 _obstacle_cost 复用（无 scipy 时开销可观）。
    if config.obstacle_safe_distance > 0.0:
        # 硬约束（把原来的软打分升级为硬过滤）：候选前沿到最近障碍的（欧氏）栅格距离必须
        # >= obstacle_safe_distance（真机基准 0.25 m = 5 格）。避免目标点落在墙边导致车顶墙。
        obstacle_mask = explored_map == 1
        if distance_to_obstacle is None and np.any(obstacle_mask):
            distance_to_obstacle = _distance_to_obstacle(obstacle_mask)
        if distance_to_obstacle is not None:
            valid_mask = valid_mask & (
                distance_to_obstacle >= float(config.obstacle_safe_distance)
            )
    score_map = np.full(explored_map.shape, -np.inf, dtype=np.float32)
    distance_cost = np.zeros(explored_map.shape, dtype=np.float32)
    turn_cost = np.zeros(explored_map.shape, dtype=np.float32)
    obstacle_cost = np.zeros(explored_map.shape, dtype=np.float32)

    if not np.any(valid_mask):
        return FrontierDecision(
            target=None,
            frontier_mask=frontier_mask,
            score_map=score_map,
            distance_cost=distance_cost,
            turn_cost=turn_cost,
            obstacle_cost=obstacle_cost,
        )

    frontier_coords = np.argwhere(valid_mask)
    frontier_distances = distances[valid_mask]
    max_distance = max(float(np.max(frontier_distances)), 1.0)
    frontier_distance_cost = frontier_distances / max_distance
    frontier_turn_cost = _turn_cost(robot_pos, frontier_coords, previous_direction)
    frontier_obstacle_cost = _obstacle_cost(
        explored_map,
        frontier_coords,
        config.obstacle_safe_distance,
        # 复用上面硬过滤已算出的距离数组，避免再算一次（无 scipy 时开销可观）。
        distance_to_obstacle=distance_to_obstacle,
    )

    frontier_scores = (
        -config.bfs_weight * frontier_distance_cost
        -config.turn_weight * frontier_turn_cost
        -config.obstacle_weight * frontier_obstacle_cost
    )
    score_map[valid_mask] = frontier_scores
    distance_cost[valid_mask] = frontier_distance_cost
    turn_cost[valid_mask] = frontier_turn_cost
    obstacle_cost[valid_mask] = frontier_obstacle_cost

    best_index = int(np.argmax(frontier_scores))
    target = tuple(int(value) for value in frontier_coords[best_index])
    return FrontierDecision(
        target=target,
        frontier_mask=frontier_mask,
        score_map=score_map,
        distance_cost=distance_cost,
        turn_cost=turn_cost,
        obstacle_cost=obstacle_cost,
    )


def choose_frontier(
    explored_map: np.ndarray,
    robot_pos: Sequence[int],
    previous_direction: Optional[Sequence[float]] = None,
    config: Optional[LightHeuristicConfig] = None,
) -> Optional[Tuple[int, int]]:
    """简化接口：只返回目标 frontier 坐标。"""
    return decide_frontier(
        explored_map=explored_map,
        robot_pos=robot_pos,
        previous_direction=previous_direction,
        config=config,
    ).target
