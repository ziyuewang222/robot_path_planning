# from dataclasses import dataclass
# from typing import Optional, Tuple

# import numpy as np
# import torch

# from bfs import BFSFrontierPlanner
# from config import (
#     HEURISTIC_COST_WEIGHT,
#     HEURISTIC_GAIN_WEIGHT,
#     HEURISTIC_TURN_WEIGHT,
#     TEST_MAP_SIZE,
#     SENSOR_RANGE,
#     max_macro_steps_for,
# )
# from state_pipeline import StateTensorPipeline


# @dataclass
# class RLRunResult:
#     total_steps: int
#     total_area: float
#     reached_done: bool
#     rounds: int
#     last_target: Optional[list]


# class HeuristicRuntime:
#     def __init__(
#         self,
#         weight_path: str = None,
#         fusion_weight_path=None,
#         map_size: int = TEST_MAP_SIZE,
#         w_gain: float = HEURISTIC_GAIN_WEIGHT,
#         w_cost: float = HEURISTIC_COST_WEIGHT,
#         w_turn: float = HEURISTIC_TURN_WEIGHT,
#         debug_print: bool = False,
#     ):
#         self.map_size = map_size
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.pipeline = StateTensorPipeline(map_size=map_size, device=self.device)
#         self.max_rounds = max_macro_steps_for(map_size)
        
#         self.w_gain = float(w_gain)
#         self.w_cost = float(w_cost)
#         self.w_turn = float(w_turn)
        
#         self.previous_direction: Optional[Tuple[float, float]] = None
#         self.bfs_planner = BFSFrontierPlanner()
#         self.debug_print = bool(debug_print)
        
#         # 各启发项在 runtime 内按各自独立规则做一次归一化
#         self.global_max_gain = float(np.pi * (SENSOR_RANGE ** 2))
#         # 极端情况下的最大可能步数（地图长+宽的曼哈顿距离的 1.5 倍冗余）
#         self.global_max_cost = float(map_size * 3)

#     def process_state(self, explored_map, robot_pos):
#         return self.pipeline.process(explored_map, robot_pos)

#     def _turn_penalty(self, robot_pos, frontier_list: np.ndarray) -> np.ndarray:
#         """
#         掉头惩罚（U-Turn Penalty）:
#             P_hat(f) = (1 - cos(theta)) / 2
#         """
#         if frontier_list.size == 0:
#             return np.zeros((0,), dtype=np.float32)
#         if self.previous_direction is None:
#             return np.zeros((len(frontier_list),), dtype=np.float32)

#         prev_vec = np.array(self.previous_direction, dtype=np.float32)
#         prev_norm = float(np.linalg.norm(prev_vec))
#         if prev_norm <= 1e-6:
#             return np.zeros((len(frontier_list),), dtype=np.float32)
#         prev_unit = prev_vec / prev_norm

#         penalties = np.zeros((len(frontier_list),), dtype=np.float32)
#         rx, ry = float(robot_pos[0]), float(robot_pos[1])
        
#         for idx, (fy, fx) in enumerate(frontier_list):
#             frontier_vec = np.array([float(fy) - rx, float(fx) - ry], dtype=np.float32)
#             frontier_norm = float(np.linalg.norm(frontier_vec))
#             if frontier_norm <= 1e-6:
#                 penalties[idx] = 1.0  # 目标就在脚下，通常已被过滤，给满惩罚
#                 continue
#             frontier_unit = frontier_vec / frontier_norm
#             cos_theta = float(np.clip(np.dot(frontier_unit, prev_unit), -1.0, 1.0))
#             penalties[idx] = (1.0 - cos_theta) / 2.0
            
#         return penalties

#     def _debug_print_scores(self, frontier_list, gain_norm, cost_norm, turn_norm):
#         if not self.debug_print:
#             return
#         print("\n--- 启发式分项得分（未乘权重）---")
#         for idx, (coord, g, c, t) in enumerate(zip(frontier_list, gain_norm, cost_norm, turn_norm)):
#             print(
#                 f"#{idx:03d} coord={coord.tolist()} | "
#                 f"gain={float(g):.3f} | cost={float(c):.3f} | turn={float(t):.3f}"
#             )

#     @staticmethod
#     def _is_frontier_cell(explored_map: np.ndarray, x: int, y: int) -> bool:
#         if explored_map[x, y] != 0:
#             return False
#         h, w = explored_map.shape
#         for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < h and 0 <= ny < w and explored_map[nx, ny] == -1:
#                 return True
#         return False

#     def _cost_only_bfs_frontier_scores(self, explored_map: np.ndarray, robot_pos):
#         from collections import deque

#         h, w = explored_map.shape
#         sx, sy = int(robot_pos[0]), int(robot_pos[1])
#         if not (0 <= sx < h and 0 <= sy < w) or explored_map[sx, sy] != 0:
#             return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

#         visited = np.zeros((h, w), dtype=bool)
#         queue = deque([(sx, sy, 0)])
#         visited[sx, sy] = True
#         frontier_coords = []
#         frontier_distances = []

#         while queue:
#             x, y, dist = queue.popleft()
#             if (x != sx or y != sy) and self._is_frontier_cell(explored_map, x, y):
#                 frontier_coords.append([x, y])
#                 frontier_distances.append(dist)
#                 continue

#             for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and explored_map[nx, ny] == 0:
#                     visited[nx, ny] = True
#                     queue.append((nx, ny, dist + 1))

#         if not frontier_coords:
#             return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

#         frontier_coords = np.asarray(frontier_coords, dtype=np.int32)
#         frontier_distances = np.asarray(frontier_distances, dtype=np.float32)
        
#         # 修复点：采用全局最大可达代价进行绝对归一化，而非动态最大值 dmax
#         frontier_scores = np.clip(frontier_distances / self.global_max_cost, 0.0, 1.0)
        
#         return frontier_coords, frontier_scores

#     def _select_by_heuristic(self, explored_map, state_tensor, robot_pos):
#         frontier_mask = state_tensor[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
#         frontier_list = np.argwhere(frontier_mask)
#         if len(frontier_list) == 0:
#             return None, None, None

#         # cost-only 退化分支
#         if self.w_gain == 0.0 and self.w_turn == 0.0 and self.w_cost > 0.0:
#             frontier_coords, frontier_scores = self._cost_only_bfs_frontier_scores(explored_map, robot_pos)
#             if frontier_coords.size == 0:
#                 return None, None, None
#             chosen_idx = int(np.argmin(frontier_scores))
#             chosen_coord = frontier_coords[chosen_idx].tolist()
#             utility = (-self.w_cost * frontier_scores).astype(np.float32)
            
#             gain_norm = np.zeros_like(frontier_scores, dtype=np.float32)
#             turn_norm = np.zeros_like(frontier_scores, dtype=np.float32)
#             self._debug_print_scores(frontier_coords, gain_norm, frontier_scores, turn_norm)
#             return chosen_idx, chosen_coord, utility

#         # 提取增益 (Channel 0 为 Visibility-aware gain map)
#         gain_map = state_tensor[StateTensorPipeline.UNKNOWN_FRONTIER_CHANNEL].detach().cpu().numpy()
#         gains = np.array([gain_map[y, x] for y, x in frontier_list], dtype=np.float32)
        
#         # 提取代价与转弯惩罚
#         turn_penalties = self._turn_penalty(robot_pos, frontier_list)
#         frontier_coords, cost_scores = self._cost_only_bfs_frontier_scores(explored_map, robot_pos)
#         cost_lookup = {tuple(coord.tolist()): float(score) for coord, score in zip(frontier_coords, cost_scores)}
#         cost_values = np.array([cost_lookup.get((int(y), int(x)), 1.0) for y, x in frontier_list], dtype=np.float32)

#         # 修复点：将动态归一化替换为基于全局常数的绝对归一化
#         gain_norm = np.clip(gains / self.global_max_gain, 0.0, 1.0).astype(np.float32)
#         cost_norm = np.clip(cost_values, 0.0, 1.0).astype(np.float32)
#         turn_norm = np.clip(turn_penalties, 0.0, 1.0).astype(np.float32)

#         # 最终效用计算
#         utility = self.w_gain * gain_norm - self.w_cost * cost_norm - self.w_turn * turn_norm
        
#         self._debug_print_scores(frontier_list, gain_norm, cost_norm, turn_norm)
#         chosen_idx = int(np.argmax(utility))
#         chosen_coord = frontier_list[chosen_idx].tolist()
#         return chosen_idx, chosen_coord, utility

#     def select_target(self, explored_map, robot_pos):
#         state_tensor = self.process_state(explored_map, robot_pos)
#         _chosen_idx, chosen_coord, _utility = self._select_by_heuristic(explored_map, state_tensor, robot_pos)
#         return state_tensor, chosen_coord

#     def run_locked(self, env, on_step=None, render_hook=None):
#         explored, pos = env.reset()
#         total_steps = 0
#         total_area = 0.0
#         current_target = None
#         rounds = 0
#         done = False
#         try:
#             with torch.no_grad():
#                 for step_idx in range(1, self.max_rounds + 1):
#                     state_tensor = self.process_state(explored, pos)
#                     frontier_mask = state_tensor[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
#                     frontier_list = np.argwhere(frontier_mask)
#                     if len(frontier_list) == 0:
#                         break

#                     chosen_idx, current_target, utility = self._select_by_heuristic(explored, state_tensor, pos)
#                     if chosen_idx is None or current_target is None:
#                         break

#                     next_explored, next_pos, _reward, n_steps, done, info = env.step_macro(current_target)
#                     total_steps += n_steps
#                     total_area += info.get("coverage_gain", 0.0)
#                     rounds = step_idx
#                     if on_step is not None:
#                         on_step(step_idx, current_target, next_pos, n_steps, done, info, True, utility)

#                     # 修复点：移除 np.sign，保留真实的位移向量以计算精确的余弦掉头惩罚
#                     move_vec = (float(next_pos[0] - pos[0]), float(next_pos[1] - pos[1]))
#                     if np.linalg.norm(move_vec) > 1e-6:
#                         self.previous_direction = move_vec

#                     explored, pos = next_explored, next_pos
#                     if done:
#                         break
#         finally:
#             if render_hook is not None:
#                 render_hook.close()
#         return RLRunResult(total_steps, total_area, done, rounds, current_target)

# RLRuntime = HeuristicRuntime

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from bfs import BFSFrontierPlanner
from config import (
    HEURISTIC_COST_WEIGHT,
    HEURISTIC_GAIN_WEIGHT,
    HEURISTIC_TURN_WEIGHT,
    TEST_MAP_SIZE,
    SENSOR_RANGE,
    max_macro_steps_for,
)
from state_pipeline import StateTensorPipeline


@dataclass
class RLRunResult:
    total_steps: int
    total_area: float
    reached_done: bool
    rounds: int
    last_target: Optional[list]


class HeuristicRuntime:
    def __init__(
        self,
        weight_path: str = None,
        fusion_weight_path=None,
        map_size: int = TEST_MAP_SIZE,
        w_gain: float = HEURISTIC_GAIN_WEIGHT,
        w_cost: float = HEURISTIC_COST_WEIGHT,
        w_turn: float = HEURISTIC_TURN_WEIGHT,
        debug_print: bool = False,
    ):
        self.map_size = map_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = StateTensorPipeline(map_size=map_size, device=self.device)
        self.max_rounds = max_macro_steps_for(map_size)
        
        self.w_gain = float(w_gain)
        self.w_cost = float(w_cost)
        self.w_turn = float(w_turn)
        
        self.previous_direction: Optional[Tuple[float, float]] = None
        self.bfs_planner = BFSFrontierPlanner()
        self.debug_print = bool(debug_print)
        
        # 修正分母：信息增益基准设定为纯理论 2D 面积极值 (圆面积)
        self.global_max_gain = float(np.pi * (SENSOR_RANGE ** 2))
        
        # 修正分母：将原本过大的 (map_size * 3) 缩小，使距离权重在正常房间尺度内更具梯度差异
        self.global_max_cost = float(map_size * 1.5)

    def process_state(self, explored_map, robot_pos):
        return self.pipeline.process(explored_map, robot_pos)

    def _turn_penalty(self, robot_pos, frontier_list: np.ndarray) -> np.ndarray:
        if frontier_list.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if self.previous_direction is None:
            return np.zeros((len(frontier_list),), dtype=np.float32)

        prev_vec = np.array(self.previous_direction, dtype=np.float32)
        prev_norm = float(np.linalg.norm(prev_vec))
        if prev_norm <= 1e-6:
            return np.zeros((len(frontier_list),), dtype=np.float32)
        prev_unit = prev_vec / prev_norm

        penalties = np.zeros((len(frontier_list),), dtype=np.float32)
        rx, ry = float(robot_pos[0]), float(robot_pos[1])
        
        for idx, (fy, fx) in enumerate(frontier_list):
            frontier_vec = np.array([float(fy) - rx, float(fx) - ry], dtype=np.float32)
            frontier_norm = float(np.linalg.norm(frontier_vec))
            if frontier_norm <= 1e-6:
                penalties[idx] = 1.0
                continue
            frontier_unit = frontier_vec / frontier_norm
            cos_theta = float(np.clip(np.dot(frontier_unit, prev_unit), -1.0, 1.0))
            penalties[idx] = (1.0 - cos_theta) / 2.0
            
        return penalties

    def _debug_print_scores(self, frontier_list, gain_norm, cost_norm, turn_norm):
        if not self.debug_print:
            return
        print("\n--- 启发式分项得分（未乘权重）---")
        for idx, (coord, g, c, t) in enumerate(zip(frontier_list, gain_norm, cost_norm, turn_norm)):
            print(
                f"#{idx:03d} coord={coord.tolist()} | "
                f"gain={float(g):.3f} | cost={float(c):.3f} | turn={float(t):.3f}"
            )

    @staticmethod
    def _is_frontier_cell(explored_map: np.ndarray, x: int, y: int) -> bool:
        if explored_map[x, y] != 0:
            return False
        h, w = explored_map.shape
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and explored_map[nx, ny] == -1:
                return True
        return False

    def _cost_only_bfs_frontier_scores(self, explored_map: np.ndarray, robot_pos):
        from collections import deque

        h, w = explored_map.shape
        sx, sy = int(robot_pos[0]), int(robot_pos[1])
        if not (0 <= sx < h and 0 <= sy < w) or explored_map[sx, sy] != 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        visited = np.zeros((h, w), dtype=bool)
        queue = deque([(sx, sy, 0)])
        visited[sx, sy] = True
        frontier_coords = []
        frontier_distances = []

        while queue:
            x, y, dist = queue.popleft()
            if (x != sx or y != sy) and self._is_frontier_cell(explored_map, x, y):
                frontier_coords.append([x, y])
                frontier_distances.append(dist)
                continue

            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and explored_map[nx, ny] == 0:
                    visited[nx, ny] = True
                    queue.append((nx, ny, dist + 1))

        if not frontier_coords:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        frontier_coords = np.asarray(frontier_coords, dtype=np.int32)
        frontier_distances = np.asarray(frontier_distances, dtype=np.float32)
        
        frontier_scores = np.clip(frontier_distances / self.global_max_cost, 0.0, 1.0)
        return frontier_coords, frontier_scores

    def _select_by_heuristic(self, explored_map, state_tensor, robot_pos):
        frontier_mask = state_tensor[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
        frontier_list = np.argwhere(frontier_mask)
        if len(frontier_list) == 0:
            return None, None, None

        if self.w_gain == 0.0 and self.w_turn == 0.0 and self.w_cost > 0.0:
            frontier_coords, frontier_scores = self._cost_only_bfs_frontier_scores(explored_map, robot_pos)
            if frontier_coords.size == 0:
                return None, None, None
            chosen_idx = int(np.argmin(frontier_scores))
            chosen_coord = frontier_coords[chosen_idx].tolist()
            utility = (-self.w_cost * frontier_scores).astype(np.float32)
            
            gain_norm = np.zeros_like(frontier_scores, dtype=np.float32)
            turn_norm = np.zeros_like(frontier_scores, dtype=np.float32)
            self._debug_print_scores(frontier_coords, gain_norm, frontier_scores, turn_norm)
            return chosen_idx, chosen_coord, utility

        # Gain 读取绝对面积进行归一化
        gain_map = state_tensor[StateTensorPipeline.UNKNOWN_FRONTIER_CHANNEL].detach().cpu().numpy()
        gains = np.array([gain_map[y, x] for y, x in frontier_list], dtype=np.float32)
        
        turn_penalties = self._turn_penalty(robot_pos, frontier_list)
        frontier_coords, cost_scores = self._cost_only_bfs_frontier_scores(explored_map, robot_pos)
        cost_lookup = {tuple(coord.tolist()): float(score) for coord, score in zip(frontier_coords, cost_scores)}
        cost_values = np.array([cost_lookup.get((int(y), int(x)), 1.0) for y, x in frontier_list], dtype=np.float32)

        gain_norm = np.clip(gains / self.global_max_gain, 0.0, 1.0).astype(np.float32)
        cost_norm = np.clip(cost_values, 0.0, 1.0).astype(np.float32)
        turn_norm = np.clip(turn_penalties, 0.0, 1.0).astype(np.float32)

        utility = self.w_gain * gain_norm - self.w_cost * cost_norm - self.w_turn * turn_norm
        
        self._debug_print_scores(frontier_list, gain_norm, cost_norm, turn_norm)
        chosen_idx = int(np.argmax(utility))
        chosen_coord = frontier_list[chosen_idx].tolist()
        return chosen_idx, chosen_coord, utility

    def select_target(self, explored_map, robot_pos):
        state_tensor = self.process_state(explored_map, robot_pos)
        _chosen_idx, chosen_coord, _utility = self._select_by_heuristic(explored_map, state_tensor, robot_pos)
        return state_tensor, chosen_coord

    def run_locked(self, env, on_step=None, render_hook=None):
        explored, pos = env.reset()
        total_steps = 0
        total_area = 0.0
        current_target = None
        rounds = 0
        done = False
        try:
            with torch.no_grad():
                for step_idx in range(1, self.max_rounds + 1):
                    state_tensor = self.process_state(explored, pos)
                    frontier_mask = state_tensor[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
                    frontier_list = np.argwhere(frontier_mask)
                    if len(frontier_list) == 0:
                        break

                    chosen_idx, current_target, utility = self._select_by_heuristic(explored, state_tensor, pos)
                    if chosen_idx is None or current_target is None:
                        break

                    next_explored, next_pos, _reward, n_steps, done, info = env.step_macro(current_target)
                    total_steps += n_steps
                    total_area += info.get("coverage_gain", 0.0)
                    rounds = step_idx
                    if on_step is not None:
                        on_step(step_idx, current_target, next_pos, n_steps, done, info, True, utility)

                    move_vec = (float(next_pos[0] - pos[0]), float(next_pos[1] - pos[1]))
                    if np.linalg.norm(move_vec) > 1e-6:
                        self.previous_direction = move_vec

                    explored, pos = next_explored, next_pos
                    if done:
                        break
        finally:
            if render_hook is not None:
                render_hook.close()
        return RLRunResult(total_steps, total_area, done, rounds, current_target)

RLRuntime = HeuristicRuntime