from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # 真机 / ROS 运行环境可能没有 scipy，缺少时退回自实现的栅格距离变换
    from scipy.ndimage import distance_transform_edt as _scipy_edt
except ImportError:  # pragma: no cover - 仅在无 scipy 环境触发
    _scipy_edt = None


@dataclass(frozen=True)
class LightHeuristicConfig:
    """权重和安全距离，距离单位为栅格。"""

    bfs_weight: float = 1.0
    turn_weight: float = 0.35
    obstacle_weight: float = 2.0
    obstacle_safe_distance: float = 6.0


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


def _chamfer_distance_transform(obstacle_mask: np.ndarray) -> np.ndarray:
    """栅格到最近障碍的近似欧氏距离（chamfer 3-4 两遍扫描），用于无 scipy 环境。"""
    height, width = obstacle_mask.shape
    distances = np.where(obstacle_mask, 0.0, float(height + width)).astype(np.float32)
    straight = np.float32(1.0)
    diagonal = np.float32(2.0) ** np.float32(0.5)

    for x in range(height):
        for y in range(width):
            best = distances[x, y]
            if x > 0:
                best = min(best, distances[x - 1, y] + straight)
                if y > 0:
                    best = min(best, distances[x - 1, y - 1] + diagonal)
                if y < width - 1:
                    best = min(best, distances[x - 1, y + 1] + diagonal)
            if y > 0:
                best = min(best, distances[x, y - 1] + straight)
            distances[x, y] = best

    for x in range(height - 1, -1, -1):
        for y in range(width - 1, -1, -1):
            best = distances[x, y]
            if x < height - 1:
                best = min(best, distances[x + 1, y] + straight)
                if y > 0:
                    best = min(best, distances[x + 1, y - 1] + diagonal)
                if y < width - 1:
                    best = min(best, distances[x + 1, y + 1] + diagonal)
            if y < width - 1:
                best = min(best, distances[x, y + 1] + straight)
            distances[x, y] = best

    return distances


def _distance_to_obstacle(obstacle_mask: np.ndarray) -> np.ndarray:
    if _scipy_edt is not None:
        return _scipy_edt(~obstacle_mask).astype(np.float32)
    return _chamfer_distance_transform(obstacle_mask)


def _obstacle_cost(
    explored_map: np.ndarray,
    frontier_coords: np.ndarray,
    safe_distance: float,
) -> np.ndarray:
    if frontier_coords.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if safe_distance <= 0:
        raise ValueError("obstacle_safe_distance 必须大于 0")

    obstacle_mask = explored_map == 1
    if not np.any(obstacle_mask):
        return np.zeros(len(frontier_coords), dtype=np.float32)

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
) -> FrontierDecision:
    """根据二维探索地图选择一个可达 frontier。"""
    if explored_map.ndim != 2:
        raise ValueError("explored_map 必须是二维数组")
    if explored_map.size == 0:
        raise ValueError("explored_map 不能为空")

    config = config or LightHeuristicConfig()
    frontier_mask = build_frontier_mask(explored_map)
    distances = _reachable_distances(
        explored_map,
        (int(robot_pos[0]), int(robot_pos[1])),
    )
    valid_mask = (frontier_mask == 1) & (distances >= 0)
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
