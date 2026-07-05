from dataclasses import dataclass
from typing import Optional

import torch

from config import TEST_MAP_SIZE, max_macro_steps_for
from models import SpatialActionDecision, SpatialFCN
from state_pipeline import StateTensorPipeline


@dataclass
class RLRunResult:
    total_steps: int
    total_area: float
    reached_done: bool
    rounds: int
    last_target: Optional[list]


class RLRuntime:
    """只保留 RL 推理运行逻辑，不持有环境实例。"""

    def __init__(self, weight_path: str, fusion_weight_path: str, map_size: int = TEST_MAP_SIZE):
        self.map_size = map_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = StateTensorPipeline(map_size=map_size, sigma=3.0, device=self.device)
        self.policy_net = SpatialFCN().to(self.device)
        state_dict = torch.load(weight_path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.load_fusion_weights(fusion_weight_path)
        self.policy_net.eval()
        self.decision = SpatialActionDecision(self.policy_net, epsilon=0.0, device=self.device)
        self.max_rounds = max_macro_steps_for(map_size)

    def process_state(self, explored_map, robot_pos):
        return self.pipeline.process(explored_map, robot_pos)

    def select_target(self, explored_map, robot_pos):
        state_tensor = self.process_state(explored_map, robot_pos)
        chosen_coord, _q_masked, _q_raw = self.decision.select_action(state_tensor, explore=False)
        return state_tensor, chosen_coord

    def build_model_info(self, state_tensor):
        if not hasattr(self.policy_net, "forward_with_inspection"):
            return None
        return self.policy_net.forward_with_inspection(state_tensor.unsqueeze(0).to(self.device))

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
                    if current_target is None:
                        current_target, _q_masked, _q_raw = self.decision.select_action(state_tensor, explore=False)
                        if current_target is None:
                            break

                    model_info = self.build_model_info(state_tensor)
                    if render_hook is not None:
                        render_hook.update(step_idx, explored, pos, state_tensor, model_info, current_target)

                    next_explored, next_pos, _reward, n_steps, done, info = env.step_macro(current_target)
                    total_steps += n_steps
                    total_area += info["delta_area"]
                    reached_target = (next_pos[0] == current_target[0] and next_pos[1] == current_target[1])
                    rounds = step_idx

                    if on_step is not None:
                        on_step(step_idx, current_target, next_pos, n_steps, done, info, reached_target, model_info)

                    explored, pos = next_explored, next_pos
                    if reached_target or (info["blocked"] and not reached_target):
                        current_target = None
                    if done:
                        break
        finally:
            if render_hook is not None:
                render_hook.close()

        return RLRunResult(total_steps, total_area, done, rounds, current_target)
