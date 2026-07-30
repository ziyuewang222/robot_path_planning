from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from config import MAP_GENERATOR, SENSOR_RANGE
from core.bfs import find_nearest_frontier
from core.cv_extractor import SemanticExtractor
from core.heuristic import score_frontiers, _turn_penalty
from core.pretreatment import build_frontier_mask
from env.grid_env import GridWorldEnv


SelectionFn = Callable[[np.ndarray, np.ndarray, Sequence[int], Optional[np.ndarray]], Tuple[Optional[Tuple[int, int]], np.ndarray]]


@dataclass
class PolicyStepResult:
    physical_steps: float
    done: bool
    info: dict


@dataclass
class PolicyRunResult:
    total_steps: float
    total_price_of_steps: float
    total_area: float
    done: bool
    rounds: int


@dataclass
class DualRunResult:
    bfs: PolicyRunResult
    heuristic: PolicyRunResult


class HeuristicRuntimeState:
    @staticmethod
    def reset() -> None:
        if hasattr(_turn_penalty, "previous_direction"):
            _turn_penalty.previous_direction = None  # type: ignore[attr-defined]


def step_one_policy(env: GridWorldEnv, target_coord) -> PolicyStepResult:
    goal = [int(target_coord[0]), int(target_coord[1])]
    planned_path = env.planner.plan(env.explored_map, env.robot_pos, goal)
    physical_steps = float(max(0, len(planned_path) - 1)) if planned_path is not None else 0.0
    next_explored, next_pos, _reward, _n_steps, done, info = env.step_macro(goal)
    env.explored_map = next_explored
    env.robot_pos = next_pos
    return PolicyStepResult(physical_steps=physical_steps, done=done, info=info)


def run_global_frontier_policy(
    model_path,
    map_size: int,
    seed: int | None = None,
    max_rounds: int = 200,
) -> PolicyRunResult:
    env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, seed=seed, map_generator=MAP_GENERATOR)
    explored_map, _robot_pos = env.reset()
    semantic_extractor = SemanticExtractor(model_path)

    total_steps = 0.0
    total_price_of_steps = 0.0
    total_area = 0.0
    done = False
    rounds = 0

    for step_idx in range(1, max_rounds + 1):
        semantic_map = semantic_extractor.predict(explored_map)
        frontier_mask = build_frontier_mask(explored_map)
        chosen_frontier, _score_map = score_frontiers(
            explored_map=explored_map,
            frontier_mask=frontier_mask,
            robot_pos=env.robot_pos,
            semantic_map=semantic_map,
        )
        if chosen_frontier is None:
            break

        step_result = step_one_policy(env, chosen_frontier)
        total_steps += step_result.physical_steps
        total_price_of_steps += step_result.physical_steps
        total_area += float(step_result.info.get("coverage_gain", step_result.info.get("delta_area", 0.0)))
        explored_map = env.explored_map
        rounds = step_idx
        done = step_result.done
        if done:
            break

    return PolicyRunResult(total_steps, total_price_of_steps, total_area, done, rounds)


def run_dual_comparison(seed: int, map_size: int, semantic_extractor: SemanticExtractor | None = None, max_rounds: int = 200) -> DualRunResult:
    HeuristicRuntimeState.reset()
    if semantic_extractor is None:
        raise ValueError("semantic_extractor 不能为空，请先创建 SemanticExtractor 实例")

    bfs_env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, seed=seed, map_generator=MAP_GENERATOR)
    heuristic_env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, seed=seed, map_generator=MAP_GENERATOR)
    _, bfs_pos = bfs_env.reset()
    _, heuristic_pos = heuristic_env.reset()
    if not (np.array_equal(bfs_env.ground_truth, heuristic_env.ground_truth) and list(bfs_pos) == list(heuristic_pos)):
        raise RuntimeError(f"seed={seed} 时 BFS 和启发式环境没有初始化到同一张地图")

    bfs_total_steps = 0.0
    bfs_total_price_of_steps = 0.0
    bfs_total_area = 0.0
    bfs_done = False
    bfs_round = 0
    bfs_target = None

    heuristic_total_steps = 0.0
    heuristic_total_price_of_steps = 0.0
    heuristic_total_area = 0.0
    heuristic_done = False
    heuristic_round = 0
    heuristic_target = None

    for step_idx in range(1, max_rounds + 1):
        if not bfs_done:
            bfs_frontier = build_frontier_mask(bfs_env.explored_map)
            bfs_target = find_nearest_frontier(bfs_env.explored_map, bfs_frontier, bfs_env.robot_pos)
            if bfs_target is None:
                bfs_done = True
            else:
                step_result = step_one_policy(bfs_env, bfs_target)
                bfs_total_steps += step_result.physical_steps
                bfs_total_price_of_steps += step_result.physical_steps
                bfs_total_area += float(step_result.info.get("delta_area", 0.0))
                bfs_round = step_idx
                bfs_done = step_result.done

        if not heuristic_done:
            heuristic_frontier = build_frontier_mask(heuristic_env.explored_map)
            semantic_map = semantic_extractor.predict(heuristic_env.explored_map)
            heuristic_target, _score_map = score_frontiers(
                heuristic_env.explored_map,
                heuristic_frontier,
                heuristic_env.robot_pos,
                semantic_map,
            )
            if heuristic_target is None:
                heuristic_done = True
            else:
                step_result = step_one_policy(heuristic_env, heuristic_target)
                heuristic_total_steps += step_result.physical_steps
                heuristic_total_price_of_steps += step_result.physical_steps
                heuristic_total_area += float(step_result.info.get("delta_area", 0.0))
                heuristic_round = step_idx
                heuristic_done = step_result.done

        if bfs_done and heuristic_done:
            break

        if bfs_done and heuristic_done:
            break

    return DualRunResult(
        PolicyRunResult(bfs_total_steps, bfs_total_price_of_steps, bfs_total_area, bfs_done, bfs_round),
        PolicyRunResult(heuristic_total_steps, heuristic_total_price_of_steps, heuristic_total_area, heuristic_done, heuristic_round),
    )
