import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from config import SENSOR_RANGE, MAP_GENERATOR
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline


@dataclass
class EpisodeStats:
    episode: int
    train_seed: int
    repeat_idx: int
    reward: float = 0.0
    loss_sum: float = 0.0
    update_cnt: int = 0
    steps: int = 0
    free_square: int = 0
    delta_area_sum: float = 0.0
    collision_count: int = 0
    blocked_count: int = 0
    macro_steps: int = 0
    end_reason: str = "达到本回合最大宏步数限制"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / self.update_cnt if self.update_cnt > 0 else 0.0

    def to_log_line(self, epsilon: float, repeat_episodes: int, **extra_fields) -> Dict[str, Any]:
        payload = {
            "episode": self.episode,
            "train_seed": self.train_seed,
            "train_seed_repeat_idx": self.repeat_idx,
            "train_seed_repeat_episodes": repeat_episodes,
            "reward": self.reward,
            "avg_loss": self.avg_loss,
            "epsilon": epsilon,
            "steps": self.steps,
            "free_square": self.free_square,
            "delta_area_sum": self.delta_area_sum,
            "collision_count": self.collision_count,
            "blocked_count": self.blocked_count,
            "macro_steps": self.macro_steps,
            "end_reason": self.end_reason,
        }
        payload.update(self.extra)
        payload.update(extra_fields)
        return payload


def _make_reset_payload(env: GridWorldEnv):
    explored, pos = env.reset()
    return explored, tuple(pos)


def _worker_loop(remote, map_size: int, sensor_range: int, map_generator: str):
    env: Optional[GridWorldEnv] = None
    pipeline = StateTensorPipeline(map_size=map_size, device="cpu")
    try:
        while True:
            command, data = remote.recv()
            if command == "reset":
                env = GridWorldEnv(map_size=map_size, sensor_range=sensor_range, seed=data, map_generator=map_generator)
                explored, pos = _make_reset_payload(env)
                state_tensor = pipeline.process(explored, pos)
                frontier = state_tensor[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
                frontier_list = np.argwhere(frontier)
                heuristic_matrix = env.extract_heuristic_matrix(frontier_list)
                remote.send((state_tensor.numpy(), explored, tuple(pos), frontier_list, heuristic_matrix))
            elif command == "act":
                if env is None:
                    raise RuntimeError("worker 在 reset 前收到了 act 命令")
                omega = np.asarray(data, dtype=np.float32)
                explored = env.explored_map
                pos = env.robot_pos
                state_tensor = pipeline.process(explored, pos)
                frontier = state_tensor[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
                frontier_list = np.argwhere(frontier)
                heuristic_matrix = env.extract_heuristic_matrix(frontier_list)
                if len(frontier_list) == 0:
                    remote.send((state_tensor.numpy(), explored.copy(), tuple(pos), frontier_list, heuristic_matrix, None, None, None))
                    continue
                heuristic_matrix = np.clip(heuristic_matrix, 0.0, 1.0)
                scores = heuristic_matrix @ omega.reshape(-1, 1)
                action_idx = int(np.argmax(scores))
                target_coord = frontier_list[action_idx].tolist()
                next_explored, next_pos, reward, tau_steps, done, info = env.step_macro(target_coord)
                next_state = pipeline.process(next_explored, next_pos)
                next_frontier = next_state[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
                next_frontier_list = np.argwhere(next_frontier)
                next_heuristic_matrix = env.extract_heuristic_matrix(next_frontier_list)
                remote.send((next_state.numpy(), next_explored, tuple(next_pos), next_frontier_list, next_heuristic_matrix, reward, tau_steps, done, info, action_idx, heuristic_matrix[action_idx]))
            elif command == "close":
                remote.close()
                break
            else:
                raise ValueError(f"未知 worker 命令: {command}")
    except KeyboardInterrupt:
        pass


class SubprocGridWorldVecEnv:
    def __init__(self, num_envs: int, map_size: int, sensor_range: int = SENSOR_RANGE, map_generator: str = MAP_GENERATOR, start_method: str = "spawn"):
        if num_envs < 1:
            raise ValueError("num_envs 必须 >= 1")
        self.num_envs = num_envs
        self.closed = False
        ctx = mp.get_context(start_method)
        self.remotes, worker_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.processes = []
        for worker_remote in worker_remotes:
            process = ctx.Process(target=_worker_loop, args=(worker_remote, map_size, sensor_range, map_generator), daemon=True)
            process.start()
            self.processes.append(process)
            worker_remote.close()

    def reset_at(self, env_idx: int, seed: int):
        self.remotes[env_idx].send(("reset", seed))
        return self.remotes[env_idx].recv()

    def step_with_omegas(self, omegas: Sequence[Optional[Sequence[float]]]):
        if len(omegas) != self.num_envs:
            raise ValueError("omegas 数量必须等于 num_envs")
        active_indices = []
        for idx, omega in enumerate(omegas):
            if omega is None:
                continue
            self.remotes[idx].send(("act", list(map(float, omega))))
            active_indices.append(idx)
        results = {}
        for idx in active_indices:
            results[idx] = self.remotes[idx].recv()
        return results

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def default_num_envs() -> int:
    cpu_count = mp.cpu_count()
    if cpu_count <= 2:
        return 1
    if cpu_count >= 24:
        return 8
    if cpu_count >= 16:
        return 8
    if cpu_count >= 12:
        return 6
    return max(1, min(4, cpu_count - 2))


def format_episode_line(stats: EpisodeStats, max_episodes: int, epsilon: float, repeat_episodes: int, prefix: str = "Episode", env_idx: Optional[int] = None, num_envs: Optional[int] = None, active_envs: Optional[int] = None, global_macro_step: Optional[int] = None, update_step_count: Optional[int] = None, replay_size: Optional[int] = None) -> str:
    parts = []
    if env_idx is not None and num_envs is not None:
        parts.append(f"env={env_idx:02d}/{num_envs:02d}")
    if active_envs is not None:
        parts.append(f"active={active_envs}")
    if global_macro_step is not None:
        parts.append(f"global_macro_step={global_macro_step}")
    if update_step_count is not None:
        parts.append(f"update={update_step_count}")
    if replay_size is not None:
        parts.append(f"replay={replay_size}")
    prefix_text = f"[Parallel-DQN {' '.join(parts)}] " if parts else ""
    return f"{prefix_text}{prefix} [{stats.episode:02d}/{max_episodes:02d}] | seed: {stats.train_seed} | repeat_idx: {stats.repeat_idx}/{repeat_episodes} | 累积物理回报: {stats.reward:7.2f} | 探索率(eps): {epsilon:.3f} | 平均损失: {stats.avg_loss:.4f} | steps: {stats.steps:4d} | free_square: {stats.free_square:4d} | delta_area_sum: {stats.delta_area_sum:7.1f} | collision: {stats.collision_count:2d} | blocked: {stats.blocked_count:2d} | 结束原因: {stats.end_reason}"


def initial_free_square(explored_map: np.ndarray) -> int:
    return int((explored_map == 0).sum())
