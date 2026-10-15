from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from core.heuristic import score_frontiers
from core.pretreatment import build_frontier_mask


@dataclass
class FrontierDecision:
    chosen_frontier: Optional[Tuple[int, int]]
    score_map: np.ndarray
    frontier_mask: np.ndarray
    semantic_map: Optional[np.ndarray]


def decide_frontier(
    explored_map: np.ndarray,
    robot_pos: Sequence[int],
    semantic_map: Optional[np.ndarray] = None,
) -> FrontierDecision:
    """统一的 frontier 决策入口。

    这个函数只依赖当前观测与机器人位置，便于后续接入 Explore-Bench
    或其他外部 benchmark 环境。
    """
    frontier_mask = build_frontier_mask(explored_map)
    chosen_frontier, score_map = score_frontiers(
        explored_map=explored_map,
        frontier_mask=frontier_mask,
        robot_pos=robot_pos,
        semantic_map=semantic_map,
    )
    return FrontierDecision(
        chosen_frontier=chosen_frontier,
        score_map=score_map,
        frontier_mask=frontier_mask,
        semantic_map=semantic_map,
    )
