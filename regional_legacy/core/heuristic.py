from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from core.cv_extractor import build_semantic_map


def _reachable_distances(explored_map: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    h, w = explored_map.shape
    dist = np.full((h, w), -1, dtype=np.float32)
    if explored_map[start[0], start[1]] == 1:
        return dist

    free_space = (explored_map != 1)
    current_front = np.zeros((h, w), dtype=bool)
    current_front[start[0], start[1]] = True
    visited = np.zeros((h, w), dtype=bool)

    d = 0.0
    while current_front.any():
        dist[current_front] = d
        visited |= current_front

        # Numpy 向量化十字波前平移，替代极其耗时的纯 Python BFS 队列
        next_front = np.zeros_like(current_front)
        next_front[:-1, :] |= current_front[1:, :]
        next_front[1:, :] |= current_front[:-1, :]
        next_front[:, :-1] |= current_front[:, 1:]
        next_front[:, 1:] |= current_front[:, :-1]

        current_front = next_front & free_space & (~visited)
        d += 1.0

    return dist


def score_frontiers(
    explored_map: np.ndarray,
    frontier_mask: np.ndarray,
    robot_pos: Sequence[int],
    semantic_map: Optional[np.ndarray] = None,
) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
    """对全部 frontier 打分，返回得分最高的 frontier 坐标与分数图。
    （已优化：全矩阵向量化运算，消除所有 Python for 循环）
    """
    if semantic_map is None:
        semantic_map = build_semantic_map(explored_map)

    start = (int(robot_pos[0]), int(robot_pos[1]))
    dist = _reachable_distances(explored_map, start)
    
    # 初始化全图分数矩阵
    scores = np.full(explored_map.shape, -np.inf, dtype=np.float32)

    # 提取有效边界掩码（必须是 frontier 且物理可达）
    valid_mask = (frontier_mask == 1) & (dist >= 0)
    if not valid_mask.any():
        return None, scores

    # 向量化分配语义权重
    attr = np.where(semantic_map == 2, 1.0, 100.0)

    # 一次性矩阵相乘计算所有得分
    scores[valid_mask] = -dist[valid_mask] * attr[valid_mask]

    # 直接提取矩阵中的最大值坐标
    eval_scores = np.where(valid_mask, scores, -np.inf)
    best_idx = np.argmax(eval_scores)
    best_x, best_y = np.unravel_index(best_idx, scores.shape)

    return (int(best_x), int(best_y)), scores
