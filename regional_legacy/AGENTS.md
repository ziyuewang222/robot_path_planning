# regional_legacy — AGENTS.md

> 旧版区域级自主探索管线：2D 栅格全知仿真 + 语义分割（SemanticUNet 3类） + 前沿探索（frontier-based）。
> 新管线开发时请勿直接删除本目录，仅做冻结/只读参考；所有输出只写本目录下 `results/`。

## 目录结构

- `config.py` — 唯一真值配置：地图尺寸、传感器、奖励 shaping、启发式权重、训练/测试路径。改参数只改这里。
- `map_generator.py` + `env/` — 地图与仿真：
  - `map_generator.py`：`build_map()/sample_spawn()` 路由，按 `MAP_GENERATOR` 分发
  - `env/grid_env.py`：`GridWorldEnv`，`-1=未知,0=空地,1=障碍`，Bresenham FOV，`step()/step_macro()`（A* 跟踪 + reward）
  - `env/map_gen_easy_big_room.py`：3大房间简单拓扑（默认训练用）
  - `env/fish_bone.py`：主走廊+两侧房间（与 easy 混合训练）
- `core/` — 算法核心：
  - `core/models.py`：`SemanticUNet(in=1,num_classes=3)`，输入 explored_map，输出房间/走廊/障碍语义
  - `core/cv_extractor.py`：`SemanticExtractor` / `build_semantic_map()`，测试时加载一次 checkpoint 连续推理
  - `core/explorer_policy.py`：`decide_frontier()` 统一入口（frontier_mask + score + 语义）
  - `core/pretreatment.py`：`build_frontier_mask()`，前沿提取
  - `core/heuristic.py`：`score_frontiers()`，BFS距离+转角+障碍+语义惩罚融合
  - `core/a_star.py`、`core/bfs.py`、`core/separating_runtime.py`：底层规划器
- 训练：
  - `hybrid_train.py`（当前主用）：12 worker 并行在线监督混合学习，`easy_big_room`/`fish_bone` 各半，masked-known-area CE loss
  - `train_parallel.py` + `parallel_worker.py`：旧并行 RL 训练（`LightweightBFSPlanner` + `extract_perfect_labels`）
- `standard_test/` — Explore-Bench 兼容评测：`run_standard_test.py`、`benchmark_loader.py`、`benchmark_env_adapter.py`、`maps/*.pgm`（12张固定房间/走廊地图）、`map_view.py`/`room_map_viewer.py`
- `tools/` — 离线评估：`eval_policy.py`、`eval_mask.py`、`eval_visualize.py`、`compare_single.py`、`compare_batch.py`
- `real_work/` — 真机相关：`real_config.py`、`light_qifa.py`、`compare_light.py`
- `result_manager.py` — `create_run_dirs()/save_model_artifacts()`，run 目录管理
- `results/` — 唯一输出：`runs/<exp>_<timestamp>/{checkpoints,logs,metrics,configs}` + 根下 `checkpoints/latest/best` 聚合权重。勿手动删，评测脚本默认读 `LATEST_CHECKPOINT_PATH`。

## 关键数据流

`build_map -> GridWorldEnv.reset -> _update_fov_bresenham -> explored_map(-1/0/1) -> SemanticUNet -> semantic_map(3类) -> build_frontier_mask -> score_frontiers -> decide_frontier -> AStarPlanner/step_macro -> 新观测`，闭环。

奖励 shaping 见 `config.py`：`AREA_REWARD_SCALE / ROOM_COMPLETION_* / STEP_PENALTY / COLLISION_PENALTY`，成功阈值 `EXPLORATION_SUCCESS_THRESHOLD=0.995`。

## 常用命令（在本目录下执行）

- 训练（主）：`python hybrid_train.py`
- 旧训练：`python train_parallel.py`
- 标准评测：`python -m standard_test.run_standard_test`
- 单策略评估：`python tools/eval_policy.py`，可视化：`python tools/eval_visualize.py`
- 切地图：改 `config.py` 中 `MAP_GENERATOR`（`easy_big_room|fish_bone|room|dynamic_room|random`），尺寸改 `TRAIN_MAP_SIZE/TEST_MAP_SIZE/DEFAULT_MAP_SIZE`。

## 给 Agent 的约束

1. 改超参只碰 `config.py`，测试/可视化统一走 `resolve_test_checkpoint_path()`，不要硬编码路径。
2. 训练 loss 保持 masked-known 口径（输入`!=-1`才算，详见 `hybrid_train._masked_ce_loss`），按样本已知格数归一化。
3. `results/` 下权重为生成物，默认读 latest；固定实验请显式改 `TEST_CHECKPOINT_PATH` 为某 run 的 `best_checkpoint.pth`。
4. 新增地图生成器需同时更新 `map_generator.build_map/sample_spawn`、`config.py` 常量、`standard_test/benchmark_env_adapter.py`。
5. Python 约束：`from __future__ import annotations` + `pathlib` 路径，`torch.load(..., weights_only=True)` 兼容旧权重。
