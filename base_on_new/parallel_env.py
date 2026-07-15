"""
Parallel-DQN 多进程训练说明

本文件提供 GridWorld 环境的多进程并行采样封装，供 train.py 和 derived_train.py 使用。
当前训练框架仍然是原来的 Spatial SMDP-DQN，不是 PPO/A3C；这里做的是：

1. 多个子进程分别运行 GridWorldEnv.reset / GridWorldEnv.step_macro。
2. 主进程集中运行 StateTensorPipeline、policy_net 批量动作选择、replay buffer 和 Q update。
3. 子进程不持有 CUDA 模型，避免 PyTorch CUDA 多进程问题。

运行方式
--------
普通训练：
    cd /home/wzy/robot_path_planning/base_on_new
    python train.py

派生训练：
    cd /home/wzy/robot_path_planning/base_on_new
    python derived_train.py

如何确认正在并行训练
------------------
训练启动时终端会打印类似：
    [Parallel-DQN] 即将启动 12 个 GridWorld 子进程；主进程负责 batch action selection / replay buffer / Q update；worker 只负责 reset 和 step_macro。
    [Parallel-DQN] 子进程已启动: num_envs=12, worker_pids=[12345, 12346, ...]

每个 episode 完成时会打印类似：
    [Parallel-DQN env=03/12 active=12 global_macro_step=184 update=11 replay=184] Episode [12/1000] | ...

字段含义：
    env=03/12             第 3 号环境 / 总共 12 个并行环境，编号从 00 开始。
    active=12             当前仍在运行的环境数量。训练快结束时可能小于 num_envs。
    global_macro_step=184 所有环境累计产生的宏动作数总和。
    update=23             主进程已经执行的 Q 网络更新次数。
    replay=184            replay buffer 当前 transition 数量。

如果你看到不同 env_id 交替输出，例如 env=00/08、env=05/08、env=02/08，说明多个环境正在并行采样。

并行环境数量怎么调
----------------
默认并行环境数量由本文件底部的 default_num_envs() 决定。
针对当前机器 Intel Core i7-14650HX：
    lscpu 显示在线 CPU 为 24，每核 1 线程。
    这说明系统能同时调度较多 CPU worker，但主进程还要负责 GPU 推理、Q update、replay buffer、日志和状态处理。
    因此默认不建议直接开满 24 个环境，而是保留一部分 CPU 给主进程和系统。

当前默认：
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

在你的 i7-14650HX 上，默认会使用：
    num_envs = 8

为什么不是 24：
    1. 主进程需要做 batch action selection 和 Q update，不能没有 CPU 余量。
    2. StateTensorPipeline 仍在主进程运行，里面有 BFS / gain map 等 CPU 计算。
    3. Pipe 通信、日志保存、checkpoint 保存也需要 CPU。
    4. 同步 vectorized env 会等待最慢 worker，worker 太多不一定线性加速。

当前报错显示 8GB GPU 在 batch_size=64 时会 CUDA out of memory，因此默认先采用更稳妥的 8 个环境。
如果 CPU 仍有大量空闲，并且显存稳定，可以再试：
    return 10
    return 12

建议调参顺序：
    8 -> 10 -> 12
若 CUDA out of memory、CPU 满载、系统卡顿、episode 输出反而变慢，就回退到上一个稳定值。

训练参数修改位置
--------------
普通训练参数在 base_on_new/train.py 的 train_main() 开头附近：
    num_envs = default_num_envs()
    batch_size = 16
    update_every = 16
    soft_update_every = 4
    replay_buffer_size = max(replay_buffer_size_for(map_size), 30000)

派生训练参数在 base_on_new/derived_train.py 的 derived_train_main() 开头附近，参数名相同。

参数含义与建议
------------
1. num_envs
    并行 GridWorld 环境数量。
    针对当前 i7-14650HX + 8GB GPU，默认推荐 8。
    推荐：
        当前默认：8
        CPU/GPU 都稳定后：10 或 12
        不建议起步直接 16-24，因为主进程和系统也需要 CPU，且会增加状态批量推理的显存压力。

2. batch_size
    每次 Q update 从 replay buffer 采样多少条 transition。
    当前默认改为 16，因为 8GB GPU + map_size=200 + batch_size=64 已经触发 CUDA out of memory。
    推荐：
        当前默认 / 稳妥：16
        稳定后可试：24
        显存仍有余量再试：32
        如果仍然 OOM：8

3. update_every
    每累计多少个全局 macro step 执行一次 Q update。
    注意 global_step 是所有并行环境共享累加，不是单个环境的 step。
    当前默认改为 16，因为 num_envs=12 时采样速度更快，如果仍按 8 更新，Q update 会太频繁。
    推荐：
        8  = 更新更频繁，可能更稳但更慢
        16 = 当前默认平衡
        24 = Q update 仍然偏慢时尝试
        32 = Q update 特别慢时尝试

4. soft_update_every
    每执行多少次 Q update 后，才 soft update 一次 target network。
    推荐：
        1 或 2 = 更接近旧训练逻辑，但更慢
        4      = 默认平衡
        8 或 16 = 更快，但 target network 更新更慢

5. replay_buffer_size
    replay buffer 容量。并行采样会更快填满 buffer，太小会导致经验覆盖太快。
    当前默认下限改为 30000，兼顾并行采样速度和内存占用。
    推荐：
        显存/内存压力大：20000
        当前默认：30000
        长时间训练且内存充足：50000 - 100000

推荐初始组合
----------
当前 i7-14650HX + 8GB GPU 推荐默认组合：
    num_envs = 8
    batch_size = 16
    update_every = 16
    soft_update_every = 4
    replay_buffer_size >= 30000

如果仍然 CUDA out of memory：
    num_envs = 6
    batch_size = 8
    update_every = 24
    soft_update_every = 4
    replay_buffer_size >= 20000

如果稳定且 CPU/GPU 仍有余量：
    num_envs = 10 或 12
    batch_size = 24 或 32
    update_every = 16 或 24
    soft_update_every = 4 或 8
    replay_buffer_size >= 30000

如果训练很慢应该怎么判断
----------------------
1. CPU 满载，GPU 利用率低：
    可能是 step_macro / StateTensorPipeline / Pipe 通信成为瓶颈。
    尝试降低 num_envs，或后续把 CPU 状态特征计算下放到 worker。

2. GPU 满载，CPU 等待，或者出现 CUDA out of memory：
    可能是动作选择或 Q update 成为瓶颈。
    已知 batch_size=64 在 8GB GPU + map_size=200 下会 OOM。
    OOM 时优先降低 batch_size：16 -> 8。
    如果不 OOM 但 Q update 慢，尝试增大 update_every 到 24/32，或增大 soft_update_every 到 8。
    如果显存足够，可以把 batch_size 从 16 逐步增到 24/32。

3. replay 很快填满且指标震荡：
    增大 replay_buffer_size。

4. 不同 env_id 不交替输出：
    检查 num_envs 是否为 1，或是否有 worker 卡住。

地图复用和 episode 数量
--------------------
地图复用配置来自 base_on_new/train_reference_eval.py：
    TRAIN_MAP_REPEAT_EPISODES
    train_seed_for_episode(...)

如果当前是 1000 episode / 333 张地图，说明每张地图大约重复 3 次，前期可能太分散。
建议前期减少地图数量、增加每张地图重复训练次数，例如：
    总 episode: 3000 - 5000
    每张地图重复: 30 - 50

中后期再逐步降低重复次数、增加随机地图数量，用于泛化。

实现边界
--------
当前实现是同步 vectorized env：主进程会等待本轮所有 active worker 返回结果。
这比异步采样更简单稳定。若某个地图的 step_macro 特别慢，可能拖慢这一轮；后续需要更极致性能时，再考虑异步 worker。
"""

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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

    def add_loss(self, loss_sum: float, update_cnt: int) -> None:
        if update_cnt > 0:
            self.loss_sum += loss_sum
            self.update_cnt += update_cnt

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
                env = GridWorldEnv(
                    map_size=map_size,
                    sensor_range=sensor_range,
                    seed=data,
                    map_generator=map_generator,
                )
                explored, pos = _make_reset_payload(env)
                state_tensor = pipeline.process(explored, pos)
                remote.send((state_tensor.numpy(), explored, tuple(pos)))
            elif command == "step":
                if env is None:
                    raise RuntimeError("worker 在 reset 前收到了 step 命令")
                next_explored, next_pos, reward, tau_steps, done, info = env.step_macro(data)
                next_state = pipeline.process(next_explored, next_pos)
                remote.send((next_state.numpy(), next_explored, tuple(next_pos), reward, tau_steps, done, info))
            elif command == "close":
                remote.close()
                break
            else:
                raise ValueError(f"未知 worker 命令: {command}")
    except KeyboardInterrupt:
        pass


class SubprocGridWorldVecEnv:
    def __init__(self, num_envs: int, map_size: int, sensor_range: int, map_generator: str, start_method: str = "spawn"):
        if num_envs < 1:
            raise ValueError("num_envs 必须 >= 1")
        self.num_envs = num_envs
        self.closed = False
        self.waiting = False
        ctx = mp.get_context(start_method)
        self.remotes, worker_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.processes = []
        for worker_remote in worker_remotes:
            process = ctx.Process(
                target=_worker_loop,
                args=(worker_remote, map_size, sensor_range, map_generator),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            worker_remote.close()

    def reset_at(self, env_idx: int, seed: int):
        self.remotes[env_idx].send(("reset", seed))
        return self.remotes[env_idx].recv()

    def reset_many(self, seeds: Sequence[int]):
        if len(seeds) != self.num_envs:
            raise ValueError("seeds 数量必须等于 num_envs")
        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions: Sequence[Optional[Sequence[int]]]):
        if len(actions) != self.num_envs:
            raise ValueError("actions 数量必须等于 num_envs")
        active_indices = []
        for idx, action in enumerate(actions):
            if action is None:
                continue
            self.remotes[idx].send(("step", [int(action[0]), int(action[1])]))
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


def format_episode_line(
    stats: EpisodeStats,
    max_episodes: int,
    epsilon: float,
    repeat_episodes: int,
    prefix: str = "Episode",
    env_idx: Optional[int] = None,
    num_envs: Optional[int] = None,
    active_envs: Optional[int] = None,
    global_macro_step: Optional[int] = None,
    update_step_count: Optional[int] = None,
    replay_size: Optional[int] = None,
) -> str:
    parallel_parts = []
    if env_idx is not None and num_envs is not None:
        parallel_parts.append(f"env={env_idx:02d}/{num_envs:02d}")
    if active_envs is not None:
        parallel_parts.append(f"active={active_envs}")
    if global_macro_step is not None:
        parallel_parts.append(f"global_macro_step={global_macro_step}")
    if update_step_count is not None:
        parallel_parts.append(f"update={update_step_count}")
    if replay_size is not None:
        parallel_parts.append(f"replay={replay_size}")

    parallel_prefix = f"[Parallel-DQN {' '.join(parallel_parts)}] " if parallel_parts else ""
    return (
        f"{parallel_prefix}"
        f"{prefix} [{stats.episode:02d}/{max_episodes:02d}] | seed: {stats.train_seed} | "
        f"repeat_idx: {stats.repeat_idx}/{repeat_episodes} | 累积物理回报: {stats.reward:7.2f} | "
        f"探索率(eps): {epsilon:.3f} | 平均 SmoothL1 损失: {stats.avg_loss:.4f} | "
        f"steps: {stats.steps:4d} | free_square: {stats.free_square:4d} | "
        f"delta_area_sum: {stats.delta_area_sum:7.1f} | collision: {stats.collision_count:2d} | "
        f"blocked: {stats.blocked_count:2d} | 结束原因: {stats.end_reason}"
    )


def initial_free_square(explored_map: np.ndarray) -> int:
    return int((explored_map == 0).sum())
