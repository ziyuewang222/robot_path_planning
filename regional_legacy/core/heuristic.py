from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

# 新增导入 TEST_MAP_SIZE 用于绝对归一化
from config import (
    HEURISTIC_BFS_WEIGHT,
    HEURISTIC_FRONTIER_CLASS_COVERAGE_THRESHOLD,
    HEURISTIC_FRONTIER_CLASS_PENALTY,
    HEURISTIC_OBS_WEIGHT,
    HEURISTIC_TURN_WEIGHT,
    SENSOR_RANGE,
    TEST_MAP_SIZE,
)
from core.cv_extractor import build_semantic_map


def _reachable_distances(explored_map: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    h, w = explored_map.shape
    dist = np.full((h, w), -1, dtype=np.float32)
    if explored_map[start[0], start[1]] == 1:
        return dist

    # 【重大修复】：严格限制波前只能在已知空地 (0) 上蔓延，不能穿透迷雾 (-1)
    free_space = explored_map == 0
    current_front = np.zeros((h, w), dtype=bool)
    current_front[start[0], start[1]] = True
    visited = np.zeros((h, w), dtype=bool)

    d = 0.0
    while current_front.any():
        dist[current_front] = d
        visited |= current_front

        next_front = np.zeros_like(current_front)
        next_front[:-1, :] |= current_front[1:, :]
        next_front[1:, :] |= current_front[:-1, :]
        next_front[:, :-1] |= current_front[:, 1:]
        next_front[:, 1:] |= current_front[:, :-1]

        current_front = next_front & free_space & (~visited)
        d += 1.0

    return dist


def _turn_penalty(robot_pos: Sequence[int], frontier_coords: np.ndarray) -> np.ndarray:
    if frontier_coords.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if not hasattr(_turn_penalty, "previous_direction"):
        _turn_penalty.previous_direction = None  # type: ignore[attr-defined]

    prev_direction = _turn_penalty.previous_direction  # type: ignore[attr-defined]
    if prev_direction is None:
        return np.zeros((len(frontier_coords),), dtype=np.float32)

    prev_vec = np.array(prev_direction, dtype=np.float32)
    prev_norm = float(np.linalg.norm(prev_vec))
    if prev_norm <= 1e-6:
        return np.zeros((len(frontier_coords),), dtype=np.float32)
    prev_unit = prev_vec / prev_norm

    penalties = np.zeros((len(frontier_coords),), dtype=np.float32)
    rx, ry = float(robot_pos[0]), float(robot_pos[1])
    for idx, (fx, fy) in enumerate(frontier_coords):
        frontier_vec = np.array([float(fx) - rx, float(fy) - ry], dtype=np.float32)
        frontier_norm = float(np.linalg.norm(frontier_vec))
        if frontier_norm <= 1e-6:
            penalties[idx] = 1.0
            continue
        frontier_unit = frontier_vec / frontier_norm
        cos_theta = float(np.clip(np.dot(frontier_unit, prev_unit), -1.0, 1.0))
        penalties[idx] = (1.0 - cos_theta) / 2.0
    return penalties


def _obstacle_penalty(explored_map: np.ndarray, frontier_coords: np.ndarray) -> np.ndarray:
    if frontier_coords.size == 0:
        return np.zeros((0,), dtype=np.float32)

    obstacle_mask = explored_map == 1
    if not obstacle_mask.any():
        return np.zeros((len(frontier_coords),), dtype=np.float32)

    dist_to_obs = distance_transform_edt(~obstacle_mask)
    f_x = frontier_coords[:, 0]
    f_y = frontier_coords[:, 1]
    f_dist = dist_to_obs[f_x, f_y].astype(np.float32)
    safe_dist = float(SENSOR_RANGE * 0.5)
    if safe_dist <= 1e-6:
        return np.zeros((len(frontier_coords),), dtype=np.float32)
    penalties = np.clip(1.0 - (f_dist / safe_dist), 0.0, 1.0)
    return (penalties ** 2).astype(np.float32)


def score_frontiers(
    explored_map: np.ndarray,
    frontier_mask: np.ndarray,
    robot_pos: Sequence[int],
    semantic_map: Optional[np.ndarray] = None,
) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
    """对全部 frontier 打分，返回得分最高的 frontier 坐标与分数图。"""
    if semantic_map is None:
        semantic_map = build_semantic_map(explored_map)

    start = (int(robot_pos[0]), int(robot_pos[1]))
    dist = _reachable_distances(explored_map, start)
    scores = np.full(explored_map.shape, -np.inf, dtype=np.float32)

    valid_mask = (frontier_mask == 1) & (dist >= 0)
    if not valid_mask.any():
        return None, scores

    frontier_coords = np.argwhere(valid_mask)
    turn_penalties = _turn_penalty(robot_pos, frontier_coords)
    obs_penalties = _obstacle_penalty(explored_map, frontier_coords)

    room_mask = semantic_map == 2
    total_room_cells = int(np.sum(room_mask))
    if total_room_cells > 0:
        explored_room_cells = int(np.sum(room_mask & (explored_map != -1)))
        room_coverage = float(explored_room_cells) / float(total_room_cells)
    else:
        room_coverage = 0.0

    current_class_penalty = 0.0 if room_coverage >= float(HEURISTIC_FRONTIER_CLASS_COVERAGE_THRESHOLD) else float(HEURISTIC_FRONTIER_CLASS_PENALTY)
    semantic_penalties = np.where(room_mask, 0.0, current_class_penalty).astype(np.float32)
    valid_sem_penalties = semantic_penalties[valid_mask]

    global_max_cost = float(TEST_MAP_SIZE * 1.5)
    dist_norm = np.clip(dist / global_max_cost, 0.0, 1.0).astype(np.float32)
    valid_base = dist_norm[valid_mask]

    scores[valid_mask] = (
        -HEURISTIC_BFS_WEIGHT * valid_base
        -valid_sem_penalties
        -HEURISTIC_TURN_WEIGHT * turn_penalties
        -HEURISTIC_OBS_WEIGHT * obs_penalties
    )

    eval_scores = np.where(valid_mask, scores, -np.inf)
    best_idx = np.argmax(eval_scores)
    best_x, best_y = np.unravel_index(best_idx, scores.shape)

    if len(frontier_coords) > 0:
        move_vec = np.array([float(best_x) - float(robot_pos[0]), float(best_y) - float(robot_pos[1])], dtype=np.float32)
        if np.linalg.norm(move_vec) > 1e-6:
            _turn_penalty.previous_direction = move_vec  # type: ignore[attr-defined]

    return (int(best_x), int(best_y)), scores