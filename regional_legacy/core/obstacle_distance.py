"""到最近障碍的栅格距离变换（决策侧与规划侧共用的唯一实现）。

背景（为什么必须共用一份）：
  frontier 候选过滤（`real_work/light_qifa.py`）与 A* 膨胀（`core/a_star.py`）原先各自
  实现了一套"到最近障碍距离"：
    - light_qifa 用 chamfer 3-4（近似欧氏）；
    - a_star 在没有 scipy 时用"8 邻域多轮位移"（即 Chebyshev 距离，方形膨胀）。
  两者度量不一致，导致同一个格子出现"决策侧认为 dist≈5.66 放行、A* 侧认为半径 4
  而阻挡"的重叠，目标格本身被膨胀盖住 → `plan()` 直接返回 None。

  本模块把距离变换收敛为单一份实现，两侧都调用 `distance_to_obstacle()`；再配合
  "`dist < R` 阻挡 / `dist >= R` 放行"的严格边界，保证 `== R` 时两侧判定一致。

上限化（max_distance）的性能依据：
  在线探索的调用方只关心三类结果：
    1) `dist < inflation_radius_cells`（A* 膨胀是否阻挡）；
    2) `dist >= obstacle_safe_distance`（前沿候选是否足够安全）；
    3) 软代价 `clip(1 - dist / safe_distance, 0, 1)`（`dist >= safe` 时恒为 0）。
  也就是说，任何 `dist >= safe_distance` 的值在判定与代价上**完全等价**。因此在线链路
  不需要整张图的精确距离，只算"被 max_distance 截断的距离场"即可：
    - `max_distance=None` 时走原有完整实现（离线训练/评测口径保持不变，硬要求）；
    - 给定 `max_distance` 时用 numpy 移位做"上限化 8 邻域迭代"，值超过上限的一律饱和。
  这样复杂建筑 946×1063 地图上的单轮距离场从纯 Python chamfer 的 ~2.43s 降到毫秒级。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:  # 真机 / 精简运行环境可能没有 scipy，缺少时退回自实现的栅格距离变换
    from scipy.ndimage import distance_transform_edt as _scipy_edt
except ImportError:  # pragma: no cover - 仅在无 scipy 环境触发
    _scipy_edt = None


def _chamfer_distance_transform(obstacle_mask: np.ndarray) -> np.ndarray:
    """栅格到最近障碍的近似欧氏距离（chamfer 3-4 两遍扫描），用于无 scipy 环境。

    算法与数值口径与 `real_work/light_qifa.py` 中原实现完全一致（原样搬迁，未做改动）。
    """
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


def _capped_vectorized_distance(
    obstacle_mask: np.ndarray, max_distance: float
) -> np.ndarray:
    """向量化 8 邻域迭代，返回"被 max_distance 截断的距离场"（无 scipy 时的快速路径）。

    实现要点：
      - 初值：障碍格 0，其余格饱和为 `cap + 1`；
      - 迭代 `ceil(cap)` 轮，每轮用 numpy 移位同时向 8 邻域传播（直连 +1，对角 +√2），
        取最小值，并再次饱和到 `cap + 1`；
      - 步数依据：到最近障碍的欧氏距离为 d 的格子，用 8 邻域路径至多 ceil(d) 步即可到达
        （每步欧氏推进 ≥ 1），因此迭代 ceil(cap) 轮后，所有 `d <= cap` 的格子都已收敛；
        更远的格子保持饱和值 `cap + 1`，满足"足够远"的语义。

    度量与现有 chamfer 3-4 同族（直连 1 / 对角 √2）；对角位移用 √2 而非 1，避免退化成
    Chebyshev 正方形膨胀。对 `d_true <= max_distance` 的格子，与精确欧氏距离的误差在
    上限范围内不超过 1 格；对 `d_true > max_distance` 的格子返回 `cap + 1 >= cap`。
    """
    cap = float(max_distance)
    saturated = np.float32(cap + 1.0)
    distances = np.where(obstacle_mask, np.float32(0.0), saturated).astype(np.float32)
    straight = np.float32(1.0)
    diagonal = np.float32(math.sqrt(2.0))

    iterations = max(int(math.ceil(cap)), 0)
    for _ in range(iterations):
        # 边界填充饱和值：不会引入"从地图外绕过来"的伪短路径。
        padded = np.pad(distances, 1, mode="constant", constant_values=saturated)
        best = np.minimum(distances, padded[:-2, 1:-1] + straight)
        np.minimum(best, padded[2:, 1:-1] + straight, out=best)
        np.minimum(best, padded[1:-1, :-2] + straight, out=best)
        np.minimum(best, padded[1:-1, 2:] + straight, out=best)
        np.minimum(best, padded[:-2, :-2] + diagonal, out=best)
        np.minimum(best, padded[:-2, 2:] + diagonal, out=best)
        np.minimum(best, padded[2:, :-2] + diagonal, out=best)
        np.minimum(best, padded[2:, 2:] + diagonal, out=best)
        np.minimum(best, saturated, out=best)
        # 障碍格始终为 0（邻居传播不能把障碍自身抬高）。
        best[obstacle_mask] = np.float32(0.0)
        distances = best

    return distances


def distance_to_obstacle(
    obstacle_mask: np.ndarray, max_distance: Optional[float] = None
) -> np.ndarray:
    """返回每个格子到最近障碍的栅格距离（单位：格）。障碍格自身为 0。

    有 scipy 时用 distance_transform_edt（精确欧氏）；无 scipy 时用 chamfer 3-4 两遍扫描
    （近似欧氏，与既有 real_work/light_qifa.py 的口径完全一致）。

    决策侧（light_qifa）的候选条件为 `dist >= obstacle_safe_distance`，规划侧
    （a_star 的 `_inflation_block_mask`）的阻挡条件为 `dist < inflation_radius_cells`；
    两者共用本函数，`dist == R` 时一律放行，边界一致。

    :param max_distance: 可选的上限。``None``（默认）时行为与改造前**完全一致**，
        返回整张图的完整距离（离线训练/评测口径不变）。给定时只算被截断的距离场：
          - 有 scipy：精确欧氏后再 `np.minimum(d, max_distance)`；
          - 无 scipy：向量化 8 邻域迭代 + 饱和（见 `_capped_vectorized_distance`）。
        此时 `d_true <= max_distance` 的格子误差 ≤ 1 格（原口径），
        `d_true > max_distance` 的格子返回 `>= max_distance`（"足够远"）。
        调用方只做阈值比较与软代价 clamp，超出 safe 距离的值一律等价，故判定结果不变。
    """
    obstacle_mask = np.asarray(obstacle_mask, dtype=bool)

    # ---- 路径 1：scipy 精确欧氏（有 scipy 时始终优先）----
    if _scipy_edt is not None:
        exact = _scipy_edt(~obstacle_mask).astype(np.float32)
        if max_distance is None:
            # 与原实现逐元素完全一致（硬要求）。
            return exact
        # 仅做截断，不改变任何 `<= max_distance` 的值。
        return np.minimum(exact, np.float32(max_distance))

    # ---- 路径 2：无 scipy + 给定上限 → 向量化 8 邻域上限化距离场 ----
    if max_distance is not None:
        return _capped_vectorized_distance(obstacle_mask, float(max_distance))

    # ---- 路径 3：无 scipy + 无上限 → 原 chamfer 3-4（离线口径，逐元素不变）----
    return _chamfer_distance_transform(obstacle_mask)
