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

---

# 附一、外部雷达建图/仿真成果（同学 胡昊宇，来源 prp.zip）

> 解压后真实源码已并入本仓库 `ros2_ws/src/`：仿真 `gazebo_modele`、建图 `nav_slam`。
> 原始副本（含结题报告 docx）见仓库根 `Pure-tracking-slam-automatic-navigation-system/`。
> 上游开源项目：Pure-tracking-slam-automatic-navigation-system（**ROS2 Humble 分支**）。

## 1. 他做了什么（任务与思路）

他的目标**不是**打通导航闭环，而是**搭一个 ROS2 Humble + Gazebo 的建图/仿真验证平台**，作为其"YOLO + 激光雷达融合"课题的实验载体。两个包分工如下：

- `gazebo_modele`（仿真层）
  - `src/gazebo_modele/urdf/model.urdf`：阿克曼四轮底盘（4 轮 + 前轮两个转向节），Gazebo 用 `libgazebo_ros_ackermann_drive.so` 插件，订阅 `cmd_vel`、发布 `/odom`（`publish_odom_tf=false`）。
  - 传感器：单线 2D 激光雷达 `velodyne`（720 线、360°、0.06–12m），输出 `sensor_msgs/LaserScan` → `/scan`。
  - `src/gazebo_modele/launch/gazebo.launch.py`：起 Gazebo（`world/3d.world`）+ spawn 机器 + `robot_state_publisher` + `joint_state_publisher_gui`，并**静态发** `map→odom`、`odom→base_footprint`、`base_footprint→base_link`。
  - 另有 `world/2d.world`、`world/3d.world` 两个场景，`model.urdf.bak` 为改装前备份。

- `nav_slam`（建图层，**两条并行路线**）
  - 路线 A（真 SLAM）：`src/nav_slam/launch/gmapping.launch.py`（名字叫 gmapping，实际起的是 **Cartographer 2D**）→ `cartographer_node` + `cartographer_occupancy_grid_node`，配 `src/nav_slam/config/cartographer.lua`（tracking_frame=base_link、provide_odom_frame=true、**use_odometry=false**、use_imu_data=false、max_range=12、optimize_every_n_nodes=90、回环权重 1.1e4）。输出标准 `/map`。
  - 路线 B（自研栅格，`src/nav_slam/launch/2dpoints.launch.py`）：三个自研节点 + RViz：
    - `src/nav_slam/nav_slam/points_pub_map.py`：订阅 `/scan`+`/odom`，极坐标→笛卡尔，用里程计位姿变换到 `map` 系，发 `/mapokk`（PointCloud2）。
    - `src/nav_slam/nav_slam/map_pub.py`：订阅 `/mapokk`+`/odom`，构建 **60m×60m、0.1m 的固定 OccupancyGrid**，**每帧全量重建**；三层圆形膨胀语义编码：障碍 `100`、第 1 层 `5`、第 2 层 `-8`、第 3 层 `-120`；发 `combined_grid`。
    - `src/nav_slam/nav_slam/odom_map_tf.py`：订阅 `/odom`，直接把里程计位姿广播为 `map→odom`（**把里程计当全局位姿，无 SLAM 修正**）。

## 2. 他的进度

已完成：
- 上游 ROS1 → Humble 迁移、colcon 构建、两个包编译通过，Gazebo + RViz 打通。
- 传感器由 3D 点云切到 2D 雷达，栅格地图随车实时更新。
- 底盘由差速重构为阿克曼 + `ackermann_drive` 插件 + 转向/驱动 PID 调试。
- Cartographer 2D 建图 + 回环参数调优。
- 三层膨胀语义栅格地图构建器。
- 规划层（8 邻域 A*、障碍三层膨胀代价图、B样条平滑）与控制层（纯追踪）**开发并验证过**。

**未完成（关键缺口）**：因"阿克曼底盘控制语义匹配问题"，**规划与控制节点已下线**，自主导航闭环没有打通。**这份 zip 里没有 A*/B样条/纯追踪节点**，`nav_slam` 只交付了**建图**部分。他把"闭环恢复 + 阿克曼控制器重写"列为后续优先项。

## 3. 与本文档管线的接口差异（联调必读）

| 事项 | 他的实现 | regional_explorer 期望 |
| --- | --- | --- |
| 地图话题 | 路线A `/map`；路线B `combined_grid` | 参数 `map_topic`，默认 `map` |
| 栅格语义 | 路线B：`100/5/-8/-120` | `-1=未知 / 0=空地 / >=50=障碍` |
| 未知判定 | 路线B 无 `-1`，`-8/-120` 会被**误判为未知** | `<0 → 未知` |
| TF | 静态 `map→odom`（gazebo.launch）或 Cartographer 动态 `map→odom` | 需要 `map→base_link` |
| 位姿来源 | 路线B 直接把 `/odom` 当全局位姿（有漂移） | 依赖 TF |
| 控制 | 底盘插件订阅 `cmd_vel`（linear.x / angular.z） | 无控制节点 |

结论：**路线 A（Cartographer 的 `/map`）与我们的节点语义天然兼容，应作为主地图源**；路线 B 的固定网格无累积、无回环、语义自定义，只适合可视化，不宜直接喂给探索算法。

---

# 附二、当前大目标（用户）

> **在仿真里跑通一个"自主寻路 + 建图"闭环：**
> 1. 小车用 2D 雷达扫周围环境；
> 2. 用**简单启发式算法**算出**下一个目标点**和**大致路径**（不追求最优，能跑通即可）；
> 3. 小车沿路径跑到目标点（**暂时不考虑阿克曼转向动力学**，能走就行）；
> 4. 重复 1–3，直到整张地图建完（无前沿可探）。
>
> 本质 = 把 `regional_legacy` 的离线探索算法 + `regional_explorer` 节点，接到同学的 Gazebo 仿真上，**补上"跑"这一环**，形成闭环。

现状对照：
- 已有（我方）：`regional_explorer/frontier_explorer_node.py` 已能"读 `/map` + TF → 输出 `exploration_goal` + `exploration_path`"（即目标点 + 大致路径）。
- 已有（同学方）：Gazebo 仿真 + 2D 雷达 + Cartographer `/map` + 会动的阿克曼底盘。
- **缺**：把 `exploration_path` 变成 `cmd_vel` 的**控制节点**（"跑"这一环），以及探索结束的收尾。

---

# 附三、下一步计划（按依赖顺序）

## Step 0 — 先把仿真基线跑起来（不写代码）
- `colcon build`；补依赖：`gazebo_ros`、`gazebo_ros_pkgs`、`cartographer_ros`、`nav2_bringup`、`joint_state_publisher_gui`。
- `ros2 launch gazebo_modele gazebo.launch.py`，确认：车在 Gazebo 里、`/scan` 有数据、`/odom` 有数据、`map→base_link` TF 通。
- 目的：证明"看得见、动得了"。

## Step 1 — 定地图源：采用 Cartographer 的 `/map`
- 跑 Cartographer 建图 launch，确认 `/map` 以标准 `0/100/-1` 刷新，并被 `regional_explorer` 正确转成 `-1/0/1`。
- 关键冲突：`gazebo.launch.py` 有静态 `map→odom`，Cartographer 也会发 `map→odom`，**两者只能保留一个**，否则 TF 打架。
- 若坚持用自研 `map_pub`，必须先重写栅格语义（`-8/-120` 归类），否则会被误判。

## Step 2 — 打通"目标点+路径 → 跑"（核心缺失，需要写代码）
- 新增一个**控制节点**（先用最简：跟踪下一路径点 + 到达判定，或轻量纯追踪）：订阅 `/exploration_path`（或 `/exploration_goal`）+ TF，输出 `cmd_vel` 给底盘插件。
- **先忽略阿克曼动力学**：直接发 `(linear.x, angular.z)`，插件内部会转成转向角，低速下够用。
- 到达判定：与目标点距离 < 阈值即到达 → 触发下一轮重规划，形成 `扫 → 决策 → 跑 → 再扫` 循环。

## Step 3 — 探索完成判定与收尾
- `frontier_explorer_node.py` 目前"无前沿"时只打一行日志；需落地为：停止移动 + `map_saver` 存图 + 完成标志。
- 加超时/最大步数保护，防止走不完。

## Step 4 — 串成一个 launch 联调
- 一个 launch 把 `gazebo + cartographer + frontier_explorer + controller` 一起拉起，全节点统一 `use_sim_time=true`。
- 反复调：`replan_rate_hz`、`bfs_weight / turn_weight / obstacle_weight`、到达阈值、最大线速度。

## 风险清单
1. `use_sim_time`：所有节点必须 `true`，否则 TF 与话题时间戳对不上。
2. TF 冲突：静态 `map→odom` 与 Cartographer 的 `map→odom` 重复。
3. 地图语义：自研网格的 `-8/-120` 会被 `>=50 / <0` 逻辑误判为未知。
4. `regional_explorer` 用 `sys.path` 动态加载 `regional_legacy`，运行需 `algorithm_root` 参数或 `REGIONAL_LEGACY_ROOT` 环境变量指向本仓库 `regional_legacy/`。
5. `frontier_explorer` 默认 `map_topic=map`、`robot_frame=base_link`，需与实际对齐。
6. 阿克曼 `cmd_vel` 语义：直接给角速度会被插件按阿克曼几何解释，务必低速验证。
7. **`nav_slam/setup.py` 只把 `config/*.lua` 装进 share，没有装 `config/rviz.rviz`**，而 `2dpoints.launch.py` 会去 share 找 `rviz.rviz` → 会 `FileNotFoundError`，跑路线 B 前需先修 setup.py。

---

# 附四、Step 0 实测结果与环境踩坑（2026-10-19）

## 1. 本机真实环境
- Ubuntu 20.04 + **ROS2 Foxy**（不是 Humble）+ Gazebo Classic **11.15.1**。
- 同学代码是 Humble 分支，但全是 `ament_python` + 标准 ROS2 API，**在 Foxy 下 colcon build 全通过**。
- 已安装依赖：`ros-foxy-gazebo-ros-pkgs`、`ros-foxy-cartographer-ros`、`ros-foxy-joint-state-publisher(-gui)`、`ros-foxy-xacro`。
- 注意：`/etc/apt/sources.list.d/ros-fish.list`（USTC 镜像）返回 **403 已失效**，已临时改名为 `ros-fish.list.disabled`，apt 改走官方 `packages.ros.org`。恢复命令：`sudo mv /etc/apt/sources.list.d/ros-fish.list.disabled /etc/apt/sources.list.d/ros-fish.list`。

## 2. 起仿真必须做的规避（否则必挂）
| 现象 | 原因 | 规避 |
| --- | --- | --- |
| `PermissionError: /home/wzy/.ros/log/...` | TRAE 沙箱禁止写 `~/.ros`、`~/.gazebo` | `export HOME=/tmp/roshome`（日志与 gazebo 缓存都落到 /tmp） |
| `X Error ... GLX ... X_GLXCreateContext: BadValue` → Gazebo abort、`/spawn_entity` 服务消失 | 沙箱内拿不到 X，GLX 建不出上下文 | 在**沙箱外**运行（Shell 工具 `dangerouslyDisableSandbox`） |

命令模板：
```bash
cd /home/wzy/robot_path_planning/ros2_ws && source /opt/ros/foxy/setup.bash && source install/setup.bash \
  && export HOME=/tmp/roshome && ros2 launch gazebo_modele gazebo.launch.py
```
参考启动命令（无头、只起物理服务器，适合联调/自动化）：
```bash
gzserver -s libgazebo_ros_init.so -s libgazebo_ros_factory.so \
  install/gazebo_modele/share/gazebo_modele/world/3d.world
```

## 3. 已验证通过的链路
- `colcon build`：3 包全部成功。
- `ros2 launch gazebo_modele gazebo.launch.py` → 车 **成功 spawn**（`Spawn status: Successfully spawned entity [fishbot]`）。
- `/scan`：720 线、360°、min 4.34m / max inf。
- `/odom`：有数据（frame `odom` → `base_link`）。
- `/joint_states`：6 个关节。
- TF：`map → odom → base_footprint → base_link → velodyne` 链完整可查（`map→base_link = (0,0,0.05)`）。
- 注意：Gazebo 启动到 spawn 完成要 **约 3.5 分钟**（在等 gazebo 模型库超时），前面查得太早会误以为失败。

## 4. 尚未通过：**车不会动**（下一步的首要问题）
> **已解决（2026-10-19），根因与修复见"附五"。** 以下为当时的原始记录，保留作背景。
- 持续发 `/cmd_vel`（`linear.x=0.6`）后：`/odom` 位置恒为 `(0,0)`，`/odom.twist.linear.x = NaN`。
- `/joint_states` 中四个轮子关节位置**恒为 0**（轮子根本不转）。
- 仿真时钟正常推进（519.6→524.5），说明不是暂停问题，而是**底盘模型/物理**问题。
- 怀疑方向（按优先级）：
  1. **车轮穿地**：`base_link` 在 z=0.05，轮心在 `base_link` 下 -0.04，轮半径 0.032 → 轮底 ≈ -0.022，插进地面，物理求解发散出 NaN。
  2. `ackermann_drive` 插件的 PID（steering 1500 / velocity 1000）对 0.1kg 车体过刚，导致数值不稳定。
  3. `world/3d.world` 是编辑器导出的（含 30 个重名 `link`、`Untitled222` 等），地面/碰撞体属性可能不规范。

## 5. 其它已知差异/坑
- Foxy 的 `ros2 topic echo` **不支持 `--once`**（那是 Humble+）；用 `timeout N ros2 topic echo <topic>` 读。
- Foxy 下 `ros2 topic echo` 对 `/tf_static` 需显式 `--qos-durability transient_local`；**更可靠的做法是用 rclpy 写探针**（本次验证脚本即如此）。
- `gazebo.launch.py` 里 `joint_state_publisher_gui` 需要显示器，无头环境建议换 `joint_state_publisher`。
- **TF 冲突**：`static_transform_publisher` 发 `base_footprint→base_link`，而 `robot_state_publisher` 也会从 URDF 的 `base_joint` 发同一条 → 重复发布。
- `world/3d.world` 的 `<link name='link'>` 重名会让 Gazebo 报 `Non-unique names detected` / `SDF is not valid`，属**警告级**，不影响 spawn。

---

# 附五、"车不会动" 根因定位与修复（2026-10-19，已解决）

## 1. 根因（实测确认）
`gazebo_ros_ackermann_drive` 插件里配置的 **PID 微分(D)增益非 0** 时，插件在每个控制周期输出 **NaN**，通过 `joint->SetForce()` 施加到关节上，使整车物理状态立刻变 NaN：
`/odom.twist.linear = (NaN, NaN, 0)`、位姿冻结在 `(0,0,0)`、轮子与转向节都纹丝不动（`joint_states` 全 0）。
**注意：NaN 在"零指令(baseline)"下就已经出现**——这不是 PID 增益"过大导致爆炸"，而是 D 项本身产出 NaN（疑似该 Foxy/ignition-math 版本下 PID 内 `lastError`/`dt` 初值问题）。

## 2. 排查过程（关键实验与结论）
1. **排除几何穿地**：把 `base_joint` z 由 0.05 改 0.072（轮底恰好落地），NaN 仍在 → 非主因（但该修正确实必要，保留）。
2. **排除世界文件**：同一机器人放进官方 `empty.world` 复测，仍 NaN → 非 `3d.world` 的锅。
3. **排除物理引擎**：不带插件时，机器人会正常下落着地静止；用 `gz joint -m fishbot -j <wheel_joint> -f 5` 直接加力能让车（狂野地）动起来 → 物理与关节链路正常。
4. **写了一个自研 Gazebo 插件探针**打印插件 Load 时读到的几何量，结果为**正确值**：轮半径 `0.032`、`wheel_separation=0.2`、`wheel_base=0.16`（4 个轮 collision 的 `WorldPose` 也都正常）→ **排除"几何算出 0 导致 0/0=NaN"**。
5. **变量对照实验（决定性）**：只改 PID 增益，在全新 gzserver 里逐项测：

   | 变体 | steering gain | linear gain | baseline | 发 `cmd_vel` 后 |
   |---|---|---|---|---|
   | A 全 0 | `0 0 0` | `0 0 0` | 有限 | 不动（力为 0） |
   | C 仅线性 P | `0 0 0` | `0.01 0 0` | 有限 | **x 前进 2.43m / 5s，twist=0.59** ✅ |
   | E 线性+转向 P，无 D | `0.1 0 0` | `0.01 0 0` | 有限 | 正常前进 ✅ |
   | F 含 D | `0.1 0 0.01` | `0.01 0 0.05` | **NaN** | **NaN，车冻结** ❌ |
   | B 含 D（旧增益的小值版） | `0.1 0 0.01` | `0.01 0 0.05` | **NaN** | **NaN，车冻结** ❌ |

   → **D 增益 ≠ 0 ⇒ NaN**；D=0 ⇒ 正常。

## 3. 最终修复（已写入 `ros2_ws/src/gazebo_modele/urdf/model.urdf`）
```xml
<left_steering_pid_gain>0.1 0 0</left_steering_pid_gain>
<right_steering_pid_gain>0.1 0 0</right_steering_pid_gain>
<linear_velocity_pid_gain>0.01 0 0</linear_velocity_pid_gain>
```
（另：`base_joint` z 由 `0.05 → 0.072`，`gazebo.launch.py` 中对应的 static TF `base_footprint→base_link` 也同步改为 `0.072`。）

## 4. 验证结果（在真实 `world/3d.world` 中，2026-10-19）
- baseline（无指令）：`twist=(0.0017,-0.0013)` **有限值**，不再 NaN。
- 发 `/cmd_vel (linear.x=0.4, angular.z=0.5)` 6 秒：位置 `(0.40, 1.04)`、`yaw` 变化 `2.17 rad`（**既前进又转向**）、`twist.x=0.31`。
- 插件日志正常：`Subscribed to [/cmd_vel]`、`Advertise odometry on [/odom]`；仅 `steering_wheel_joint not found` 为**无害警告**（模型本就没有方向盘关节）。

## 5. 遗留 / 待调优
- 速度跟踪偏低：指令 0.4 实测约 0.31（转向时更明显），指令 0.6 直线实测约 0.59。后续如需更准，可小幅提高 `linear_velocity_pid_gain` 的 P（仍须保持 D=0），或改用 `gazebo_ros_diff_drive`（速度环更稳）。
- 无头环境不要用 `gazebo.launch.py`（会拉 `gzclient`，需 X/GLX）。服务器验证建议：`gzserver -s libgazebo_ros_init.so -s libgazebo_ros_factory.so <world>` + `spawn_entity.py`。
- `gz joint -h` 表明可用 `-f/--force`、`--vel-t` 等直接改关节；调试很有用。
- 复现本结论的临时脚本：`/tmp/pid_test3.sh`、`/tmp/pid_test4.sh`、`/tmp/probe.cc`（自研探针插件）、`/tmp/diag2.py`（线性+角速度探针）。

---

# 附六、Step 1 实测结果（2026-10-19）

> 目标：把 Cartographer 的 `/map` 作为唯一地图源，且 TF 只有唯一链路
> `map → odom → base_footprint → base_link → velodyne`。
> 本次 Step 1 的两处改动：`cartographer.lua` 设 `published_frame="odom"`、`provide_odom_frame=false`；
> `gazebo.launch.py` 删除静态 `map→odom`，并给 `robot_state_publisher`/`joint_state_publisher_gui` 加 `use_sim_time=True`。

## 0. 复验结论速览（仓库原生 launch，本轮 2026-10-19）

- **Step 1 验收目标：✅ 已达成** —— Cartographer 的 `/map` 是**唯一地图源**；TF 唯一链路
  `map → odom → base_footprint → base_link → velodyne` 完整可解，且 `map→odom` 的唯一权威 = `cartographer_node`。
- **复验方式：完全用仓库本体源码**（不再用任何 `/tmp` 副本）。`colcon build --symlink-install` 后，
  `install` 的 `cartographer.lua` / `gmapping.launch.py` 经 `readlink -f` 确认指向 `src/`，且 install 内容已含
  `return options`、已无 Foxy 不支持的键、launch 已用 `occupancy_grid_node`。
- **仓库原生 launch 复验（关键）**：`ros2 launch gazebo_modele gazebo.launch.py` + `ros2 launch nav_slam gmapping.launch.py`
  （**只跑这一条**）能一起正常起来；**无** `lua_istable ... not a table`、**无**引用计数 Fatal、**无** `No executable found`；
  日志出现 `Added submap 1` / `Added trajectory with ID '0'` / `constraint_builder_2d ... score 8x%`。
- **逐项**：A ⚠️（严格 `{0,100,-1}` 口径不满足，实为 Foxy 连续概率，下游兼容）、B ✅、C ✅、D ℹ️（同值双权威，无害）、E ✅。

## 1. 测试方式与命令

- **测试脚本（唯一新增文件，可重复运行）**：`/home/wzy/robot_path_planning/regional_legacy/step1_check.py`
  - 子命令：`wait`（等 /clock /scan /odom）、`drive`（发 /cmd_vel）、`assert`（A/B/C/D 断言）、`e`（消费端 E）。
  - 不用 Foxy 的 `ros2 topic echo`（块缓冲问题），全部用 rclpy 直读。
- **构建（本轮，CodeWriter 修复后重编）**：`colcon build --symlink-install` → 3 包全部成功。
- **install 与源码一致性（本轮 `readlink -f` 实测，指向源码而非旧快照）**：
  ```
  install/nav_slam/share/nav_slam/config/cartographer.lua   -> src/nav_slam/config/cartographer.lua
  install/nav_slam/share/nav_slam/launch/gmapping.launch.py -> src/nav_slam/launch/gmapping.launch.py
  install/gazebo_modele/share/gazebo_modele/launch/gazebo.launch.py -> src/gazebo_modele/launch/gazebo.launch.py
  ```
  install 里的 lua 已含 `return options`、已无 `use_range_data_from_local_slam` / `global_constraint_search_after_n_nodes`；
  install 里的 launch 已改用 `executable='occupancy_grid_node'`。
- **运行环境**：`source /opt/ros/foxy/setup.bash` + `source install/setup.bash`，`export HOME=/tmp/roshome`，
  `export GAZEBO_MODEL_DATABASE_URI=""`，全部命令在**沙箱外**执行。
- **时序**：后台起 `ros2 launch gazebo_modele gazebo.launch.py` → `step1_check.py wait`（本次 spawn 很快，
  未出现 Step 0 说的 3.5 min）→ `step1_check.py drive` 驱动小车 → 起 Cartographer → 约 30~90 s 后 `step1_check.py assert`。

## 2. A–E 各项实测结论

### A. `/map` 语义 —— ⚠️ 严格口径 FAIL（连续概率，实测如实记录）
收到 6 条 `/map`：`width=514 height=405 resolution=0.0500 origin=(-13.12,-8.91)`。
取值集合**不是** `{0,100,-1}`：`-1` 之外是连续的 `0..100`（计数片段）：
```
-1: 111586, 0: 88333, 100: 352, 50: 717, 49: 377, 95: 90, 99: 201, ...（1..99 全部出现）
```
- `[PASS]` width/height/resolution 合理；`[PASS]` 有非未知栅格（free(0)=88333、occupied(100)=352）。
- `[FAIL]`（严格口径）存在大量非 `{0,100,-1}` 取值。

**根因**：Foxy 的 `occupancy_grid_node`（Foxy 里可执行名就叫 `occupancy_grid_node`，不是
`cartographer_occupancy_grid_node`）发布的是 **0..100 连续占用概率 + `-1` 未知**，并非二值 `0/100/-1`；
该节点只有 `use_sim_time` 一个参数，**没有阈值/二值化开关**（`ros2 param list` 实测）。
**影响评估**：`frontier_explorer_node._to_explored_map` 用 `data<0→未知 / data>=50→障碍 / 其余→空地`，
对连续概率**仍然可用**（40 以下判自由、50 以上判障碍）。故 A 的“严格 0/100/-1”只是**规格预期与 Foxy
Cartographer 原生行为的差异**，非阻塞。
**给 CodeWriter 的建议**：若确实要求严格三值，新增一个轻量“占用栅格阈值化”relay 节点
（订 `/map`，把 `>=50→100`、`0<val<50→0`、`<0→-1` 后另发一个 canal），或放宽验收口径为“取值⊂[0,100]∪{-1}”。

### B. TF 唯一性 —— ✅ PASS
```
/tf        publishers = [cartographer_node, robot_state_publisher]
/tf_static publishers = [robot_state_publisher, static_transform_publisher_CCc5lryxXgnK3Bgq, static_transform_publisher_zsnZIYNEXNXdGMCH]
/map       publishers = [cartographer_occupancy_grid_node]
/tf 里 (parent,child) 集合: [..., (map, odom)]
/tf_static 里 (parent,child) 集合: [(base_footprint, base_link), (base_link, velodyne), (odom, base_footprint)]
```
- `[PASS]` `/tf` 上存在动态 `map→odom`，且发布者含 `cartographer_node` → `map→odom` 唯一权威 = Cartographer。
- `[PASS]` `/tf_static` **已无 `map→odom`** → gazebo.launch.py 删静态 `map→odom` 已生效。

### C. TF 链路可解（仿真时刻）—— ✅ PASS
`use_sim_time=True` 下用 tf2 Buffer `lookup_transform(Time())`，5 段全部成功：
```
map -> odom            : translation=(-0.4104,0.3420,-0.0720)
odom -> base_footprint : translation=(0.0000,0.0000,0.0000)
base_footprint -> base_link : translation=(0.0000,0.0000,0.0720)
map -> base_link       : translation=(-0.4104,0.3420, 0.0000)
base_link -> velodyne  : translation=(0.0000,0.0000,0.1500)
```
说明链路 `map→odom→base_footprint→base_link→velodyne` 完整可解。
（附注：`map→odom` 的 z=`-0.072` 是因为 `tracking_frame=base_link` 而 `published_frame=odom`，
与下游 `odom→base_link` 的 z=`+0.072` 沿链路相消，`map→base_link` 的 z=`0`，仅 2D 使用时无害。）

### D. 冗余检查（仅报告，不判失败）—— ℹ️ INFO
```
/tf_static 上存在 base_footprint->base_link: True
发布者含 robot_state_publisher: True ；含 static_transform_publisher: True
```
→ `base_footprint→base_link` 确为**同值双权威**（URDF 固定关节 `base_joint` 经 `robot_state_publisher` +
静态 TF `fake_basel_cmd5`），两处值均为 `(0,0,0.072)`，**同值无害**。

### E. 消费端（frontier_explorer）—— ✅ PASS
`ros2 run regional_explorer frontier_explorer_node --ros-args -p algorithm_root:=/home/wzy/robot_path_planning/regional_legacy \
  -p map_topic:=map -p robot_frame:=base_link -p global_frame:=map`，运行 45 s：
```
frontier 进程存活: True
收到 /exploration_goal  条数: 29
收到 /exploration_path  条数: 29
日志可疑行数: 0
日志末尾: [INFO] frontier_explorer 启动 | algorithm_root=... | map=map | ... | tf=map->base_link
```
→ 消费端能订阅 `/map`、在 `map→base_link` 上取到机器人位姿并正常输出目标点与路径，**无 TF/地图报错**。
（附注：本机无 `scipy`，`real_work/light_qifa.py` 自动退回自实现 chamfer 距离变换，不影响运行。）

## 3. 上一轮暴露的三个缺陷 —— ✅ CodeWriter 已修入仓库本体，本轮复验通过

> 这三个问题**不是**本轮两处改动引入，而是**沿用自原始副本**（`Pure-tracking-slam-...`，按 Humble+ 写的），
> 在 Foxy 下会直接让 Cartographer 起不来。**上一轮**只能在 `/tmp` 副本上绕过验证；
> **本轮 CodeWriter 已把它们修进仓库本体（`src/nav_slam/...`）**，本轮用**仓库原生 `ros2 launch nav_slam gmapping.launch.py`**
> 复验：**无任何 Fatal、不使用任何 `/tmp` 副本**，`cartographer_node` 与 `occupancy_grid_node` 均正常起，并输出 `Added submap`。

1. **`nav_slam/config/cartographer.lua` 末尾缺 `return options`**
   - 现象：`F1019 lua_parameter_dictionary.cc:116] Check failed: lua_istable(L, -1) Topmost item on Lua stack is not a table!`
   - 根因：Cartographer 的 Lua 配置必须 `return options`；缺失导致顶层 chunk 没返回表。
   - 修复（已落地）：在文件末尾追加一行 `return options`（官方 `backpack_2d.lua` 就是这么写的）。
   - 修复前复现命令：`ros2 run cartographer_ros cartographer_node -configuration_directory install/nav_slam/share/nav_slam/config -configuration_basename cartographer.lua`
   - 本轮复验：仓库原生 launch 已打印 `Found '.../install/nav_slam/share/nav_slam/config/cartographer.lua'` 并继续启动，**无** `lua_istable` Fatal。

2. **`cartographer.lua` 含两个 Foxy 不支持的键 → 引用计数校验崩溃**
   - 现象：依次报 `Check failed: 1 == reference_counts_.count(key) Key 'use_range_data_from_local_slam' was used the wrong number of times.`
     和 `Key 'global_constraint_search_after_n_nodes' ...`。
   - 根因：这两个键在 Foxy 的 Cartographer 里根本不会被读取（官方 Foxy 默认配置里没有），
     触发 Cartographer 的“键被使用次数必须恰好为 1”校验。
   - 修复（已落地）：删除 `TRAJECTORY_BUILDER_2D.use_range_data_from_local_slam = true` 与
     `POSE_GRAPH.global_constraint_search_after_n_nodes = 10` 两行。
   - 本轮复验（仓库本体，非 /tmp）：修好后用仓库原生 launch 直接启动成功，输出
     `Added submap 1` / `Added trajectory with ID '0'` / `constraint_builder_2d ... score 8x%`。

3. **`nav_slam/launch/gmapping.launch.py` 的可执行名在 Foxy 下不存在**
   - 现象：`No executable found`（第二个 Node `executable='cartographer_occupancy_grid_node'`）。
   - 根因：Foxy 的 `cartographer_ros` 只有 `cartographer_node` 和 `occupancy_grid_node` 两个可执行
     （`ros2 pkg executables cartographer_ros` 实测）；`cartographer_occupancy_grid_node` 是 Humble+ 才有的名字。
   - 修复（已落地）：把 `gmapping.launch.py` 里 `executable` 改为 `occupancy_grid_node`。
   - 本轮复验：仓库原生 launch 日志 `[INFO] [occupancy_grid_node-2]: process started with pid [...]`，**无** `No executable found`。

## 4. 复现步骤（修复后的仓库口径 → 复现本轮通过结果）

```bash
cd /home/wzy/robot_path_planning/ros2_ws && source /opt/ros/foxy/setup.bash && source install/setup.bash
export HOME=/tmp/roshome && export GAZEBO_MODEL_DATABASE_URI=""
# 起仿真（另开终端/后台）
ros2 launch gazebo_modele gazebo.launch.py
# 起建图（本轮复验：正常启动，无 Fatal）
ros2 launch nav_slam gmapping.launch.py             # 实测无 lua Fatal、无 No executable found；输出 Added submap 1 / Added trajectory with ID '0'
```

## 5. 已知非阻塞项

- **(i) legacy 路线 `2dpoints.launch.py` 里的 `odom_map_tf` 也会广播 `map→odom`，禁止与 Cartographer 同跑。**
  上一轮启动前发现**残留进程污染**：一个**更早会话遗留的旧 `static_transform_publisher`（未清理）**仍在
  `/tf_static` 上发静态 `map→odom`，导致第一次断言里 `/tf_static` 出现 `(map,odom)`、`/tf` 上没有 Cartographer 的
  `map→odom`。**全量 kill 后重启即消失**（最终 `/tf_static` 只剩 `odom→base_footprint`、`base_footprint→base_link`、
  `base_link→velodyne`）。同理，只要 `odom_map_tf`/旧静态 `map→odom` 任一在跑，就会与 Cartographer 的 `map→odom` 打架；
  验收/联调时**只允许 Cartographer 一个 `map→odom` 权威**。
- **(ii) `base_footprint→base_link` 可能双权威（URDF 固定关节 + 静态 TF），同值无害。**
  两处值均为 `(0,0,0.072)`，TF 求值一致，无需处理；若后续要“单权威”，可删 `gazebo.launch.py` 的 `fake_basel_cmd5`
  或从 URDF 移除 `base_joint`，二选一即可。
- **(iii) A 项的 `/map` 三值化**：见上文第 2 节 A，属 Foxy 原生行为，下游已兼容，暂按非阻塞处理。

---

# 附七、Foxy ↔ Humble 双版本差异登记（真机 Humble / 本机开发 Foxy）

## 1. 为什么单开这一节
真机（小车）搭载 ROS2 Humble，本机开发环境只有 ROS2 Foxy。凡是为"在本机 Foxy 下跑通"而改动过的地方，都登记在此，并**保留 Humble 原版写法**，真机部署时按本表逐条核对是否需回改。

## 2. 双版本登记表

| # | 文件 | Humble 原版 | Foxy 现版 | 性质 | 真机是否需回改 |
| --- | --- | --- | --- | --- | --- |
| 1 | `nav_slam/launch/gmapping.launch.py` | `executable='cartographer_occupancy_grid_node'` | `executable='occupancy_grid_node'` | **真·版本差异**（Foxy 的 cartographer_ros 只提供 `cartographer_node` / `occupancy_grid_node`） | **需要回改**：Humble 上可执行名是 `cartographer_occupancy_grid_node`，真机必须改回这一行 |
| 2 | `nav_slam/config/cartographer.lua` | `TRAJECTORY_BUILDER_2D.use_range_data_from_local_slam = true` 与 `POSE_GRAPH.global_constraint_search_after_n_nodes = 10` | 两行已删除 | Foxy 的 cartographer(1.0.9001) 无这两个键 | 待 Humble 机核实 |
| 3 | `gazebo_modele/urdf/model.urdf` | `left/right_steering_pid_gain 1500 0 1`、`linear_velocity_pid_gain 1000 0 1` | `0.1 0 0` / `0.01 0 0`（D 增益必须为 0） | 本机 Gazebo 环境适配（D 增益非 0 → PID 输出 NaN → 整车冻结，见附五） | 待 Humble 机核实 |
| 4 | `gazebo_modele/launch/gazebo.launch.py` | `robot_state_publisher` / `joint_state_publisher_gui` 未设 `use_sim_time` | 各加 `parameters=[{'use_sim_time': True}]` | 通用修复，与版本无关 | 无需（建议两版一致） |
| 5 | `nav_slam/config/cartographer.lua` | 原文件末尾**没有** `return options` | 末尾补 `return options` | **通用 bug**（Humble 同样会 Fatal） | 无需（两版都该有） |
| 6 | `nav_slam/config/cartographer.lua` | `published_frame="base_link"` + `provide_odom_frame=true` | `published_frame="odom"` + `provide_odom_frame=false` | 通用 TF 冲突修复（Step 1） | 无需 |
| 7 | `gazebo_modele/launch/gazebo.launch.py` | 含静态 TF `map→odom` | 已删除 | 通用 TF 冲突修复（Step 1） | 无需 |
| 8 | `gazebo_modele/urdf/model.urdf` | `base_joint` origin `z=0.05` | `z=0.072` | 几何落地修正，与版本无关 | 无需 |
| 9 | `path_follower` 包（Step 2 新增） | —（原本没有） | — | 纯 rclpy，无版本差异；`from tf2_ros import TransformException` 已在 Foxy 实测可导入 | 无需 |
| 10 | 地图保存（Step 3 新增能力） | 真机 Humble 可用 nav2 官方 `map_saver` | 本机无 nav2 → 改为自研纯 Python 写出 map_server 标准 `.pgm`+`.yaml`（`save_occupancy_grid_map`） | 存储格式与 nav2 `map_server` 完全兼容（P5 / negate:0 / occupied_thresh 0.65 / free_thresh 0.196），**与版本无关**；真机如需官方工具可自行替换 | 无需回改 |
| 11 | IMU（2026-10-20 新增能力） | 真机为真实 IMU（型号/话题/安装朝向/静态 yaw 漂移**待补**，真机不在手边） | 仿真用 gazebo_ros 的 `libgazebo_ros_imu_sensor.so` 插件近似，发布 `/imu`（100Hz，`frame_id=imu_link`） | 插件与 `use_imu_data`/`tracking_frame` 配置**两版通用**（Humble 同样有该插件） | 无需回改；但真机需按实际 IMU 填参数，且**若 IMU 不与 tracking frame 共位需相应调整 `tracking_frame`** |

## 3. 环境差异（非代码，仅影响开发手法）

- Foxy 的 `ros2 topic echo --once/--field` 不可用，断言改用 rclpy 探针脚本。
- `ros2 pkg executables cartographer_ros` 在 Foxy 下只有 `cartographer_node` / `occupancy_grid_node`。
- 本机沙箱内跑 launch 会 GLX/X 崩溃 + `/home/wzy/.ros/log` 权限错误 → 需沙箱外执行、`HOME=/tmp/roshome`、`GAZEBO_MODEL_DATABASE_URI=""`。
- 杀进程**不要**用 `pkill -f gzserver`（会匹配到命令自身所在 shell 从而自杀），改用 `pkill -x gzserver` 或 `pkill -f '[g]zserver'`。
- 启动命令与杀进程命令要分成两条 Shell，避免 `pkill -f` 自匹配。
- 本机 Foxy **未安装 nav2**：`ros2 pkg list` 无 nav2/slam 包，`ros2 pkg executables nav2_map_server` 报 "Package not found"，全机只有 ROS1 Noetic 的 `map_saver`（不能订阅 ROS2 话题）。因此存图改为自研实现（见上表第 10 条）。

---

# 附八、Step 2 实测结果（2026-10-19）

## 0. 结论速览
Step 2 **已达成**（隔离条件下）：控制节点 `path_follower_node` 能把 `/exploration_path` / `/exploration_goal` 转成 `/cmd_vel`，小车能朝目标行驶、到达后停车、出现更远的新目标后能恢复行进（A/B/C/D/E 全 PASS）。
完整栈"扫→算→跑→再扫"的闭环**尚未在测试窗口内观测到**，留待 Step 4 联调。

## 1. 被测对象与接口
- 新包 `src/path_follower/`，节点 `path_follower_node`：订阅 `exploration_path`(nav_msgs/Path) 与 `exploration_goal`(PoseStamped)，用 TF `map→base_link` 取位姿，10 Hz 发布 `/cmd_vel`(geometry_msgs/Twist) 给 `gazebo_ros_ackermann_drive`。
- 关键参数：`max_linear_vel=0.3`、`min_linear_vel=0.08`（防目标在正后方时线速度归零致阿克曼卡死）、`slowdown_dist=0.8`（减速段）、`goal_tolerance=0.25`、`lookahead_dist=0.4`、`reached_watchdog_sec=8.0`（到达后无新目标时仅告警一次）。
- 策略：前视点（carrot）追踪；`angular.z=k*err`、`linear.x` 先按航向误差、再按到目标距离限速，最后取下限。

## 2. 第一轮：完整栈（gazebo + cartographer + frontier_explorer + path_follower）
- **为什么这一轮不能判定控制器追踪能力**：起点 `(-0.008,-0.008)` 距上游给出的前沿目标仅约 0.17 m，**已在 `goal_tolerance`(0.25) 内** → 控制器第一帧就正确判定"已到达"并输出零速；这是**正确行为**，不是 bug。
- 实测：`/cmd_vel` 收到 381 条，`linear.x` 恒 0；位移 0.033 m；距离样本前 5 均值 0.166、后 5 均值 0.135（弱收敛断言 PASS）；日志出现「已到达目标 (-0.17, -0.02)，距离 0.168 m，停止行进等待新前沿」→ D 项 PASS；**E 项未观测到**（上游 45 s 内始终发布同一个近目标，未换新前沿）。
- 副产品结论：上游长时间只给同一目标的现象，正是 `reached_watchdog_sec` 看门狗要覆盖的场景；但"探索是否结束"的判定属于 Step 3，本轮不做。

## 3. 第二轮：隔离验证（注入人造远目标，不启上游探索器）
- 方法：gazebo + cartographer + path_follower；机器人起点 `(0.012, 0.010)`、yaw≈0°；按 `/map` 选可达远点，注入 `/exploration_goal = (1.5118, 0.0102)`（正前方 1.5 m，连线上无占用栅格），持续发布。
- **A 指令产生**：`/cmd_vel` 非恒零，`linear.x` 范围 `[0.0000, 0.3000]`，峰值 = `max_linear_vel` → **PASS**
- **B 车真的动**：`start=(0.012,0.010)`、`end=(1.362,0.006)`，净位移 **1.350 m**（另一次复现 1.310 m）→ **PASS**
- **C 朝目标收敛**：距离 `1.500 → 最小 0.148 m`；前段均值 1.084、后段均值 0.152 → **PASS**
- **D 到达即停**：日志「已到达目标 (1.51, 0.01)，距离 0.238 m，停止行进等待新前沿」，其后线速度稳态归零（到达瞬间的 5 个采样残余 0.095 属探针取值滞后假象）→ **PASS**
- **E 到达后恢复行进**：注入距上一到达点 1.355 m 的新目标后，日志「检测到新前沿目标 (2.87, 0.01)，距上一到达点 1.355 m，恢复行进」，`max|linear.x|=0.3` → **PASS**

## 4. 已知限制与待办（均非本轮阻塞）
- (a) **仅给点目标时的收敛性**：当没有 Path、只给点目标且目标有横向偏置（或落在车后）时，一次运行出现"掠过目标 → 绕圈 → 距离发散到 ~11 m、始终不进容差"。真实链路里上游会给 `Path`（carrot 沿路径取），预期更好；**Step 4 联调需重点验证**，必要时改为"沿路径取 carrot + 更强减速/绕圈抑制"。
- (b) **物理卡死**：一轮运行中控制器满速 0.3 但车位移仅 0.016 m，判定为该处物理卡住/贴障碍，非控制器逻辑问题；Step 4 需确认 world 出生点不卡死。
- (c) **完整闭环未验证**：第一轮上游 45 s 未换目标；`reached_watchdog_sec=8s` 目前只是日志告警，"探索结束"判定属 Step 3。
- (d) **上游 `use_sim_time`**：`regional_explorer/launch/frontier_explorer.launch.py` 未设 `use_sim_time`，Step 4 的统一 launch 必须把所有节点统一为 `true`。
- (e) **幽灵进程**：测试中发现上一会话遗留的多个 `static_transform_publisher` / `robot_state_publisher` 与第二个 `cartographer_node`，导致 `map→odom` 出现双权威、TF 树断裂（`two or more unconnected trees`）。**每次测试前必须全量清理进程**。

## 5. 复现方式
- 断言脚本：`regional_legacy/step2_check.py`（`--seconds` 控制采样时长，输出 A–E 断言汇总）。
- 隔离编排脚本与详细报告：`/tmp/iso_run.py`、`/tmp/step2_isolated_report.md`（临时文件，仅供追溯）。
- 完整栈命令序列：`colcon build --symlink-install` → `ros2 launch gazebo_modele gazebo.launch.py` → `ros2 launch nav_slam gmapping.launch.py` → `REGIONAL_LEGACY_ROOT=/home/wzy/robot_path_planning/regional_legacy ros2 launch regional_explorer frontier_explorer.launch.py` → `ros2 launch path_follower path_follower.launch.py`。**禁止**同时启动 `2dpoints.launch.py`。

## 6. 残留阻塞
无硬阻塞。

---

# 附九、Step 3 实测结果（2026-10-19）

## 0. 结论速览
Step 3 **已达成**：探索完成判定（三条件 + 两道门槛 + 只触发一次）、完成后立即停车且不再被新目标驱动、地图以 map_server 标准格式落盘。单元测试 36/36 通过，端到端全项通过，无残留硬阻塞。

## 1. 本轮改动（2 个文件）
- `regional_explorer/regional_explorer/frontier_explorer_node.py`：
  - 新增模块级纯函数 `save_occupancy_grid_map(msg, prefix, occupied_threshold=50) -> (bool, str)`；二进制 `P5` PGM + 同目录 yaml；像素 `-1→205`、`>=threshold→0`、其余 `→255`；**行序翻转**（图像第 r 行取栅格第 `h-1-r` 行）。
  - 新增参数与完成判定（见第 2 节）。
- `path_follower/path_follower/path_follower_node.py`：新增订阅 `done_topic`（`TRANSIENT_LOCAL`）；收到 `True` → `_exploration_done=True`，`_on_control` 开头直接发零 Twist 并 return（**永久停车**，忽略后续目标）。

## 2. 完成判定设计
三条件（任一满足即完成，`_declare_done` 用 `_done` 守卫**只执行一次**）：
1) 连续 `no_frontier_confirm_count` 轮「没有产出可达目标」（无前沿 **或** A* 规划失败）；
2) 超过 `max_exploration_time_sec`（用 `time.monotonic()` 单调钟，避免 `use_sim_time` 下 `/clock` 未就绪导致首轮误超时）；
3) 累计发布目标数 ≥ `max_goal_count`。
两道防误判门槛（在 `_bump_no_target` 内，只作用于条件 1）：`_goal_count == 0`（还没产出过目标）或已知栅格数 < `min_known_cells` → **不计入** streak。机器人位姿查不到同样不计入。
完成后：发 latched `Bool(True)` 到 `done_topic` + 打日志 + 存图，并停止发布目标与路径。

新增参数表（参数名 | 默认值 | 含义）：

| 参数名 | 默认值 | 含义 |
| --- | --- | --- |
| `done_topic` | `"exploration_done"` | 完成标志发布话题 |
| `no_frontier_confirm_count` | `10` | 连续无可达目标多少轮判定完成 |
| `min_known_cells` | `100` | 门槛②：已知栅格数下限 |
| `max_exploration_time_sec` | `600.0` | 条件②：探索时限（单调钟） |
| `max_goal_count` | `200` | 条件③：累计发布目标数上限 |
| `map_save_prefix` | `"maps/exploration_map"` | 存图路径前缀（空串则跳过存图） |

path_follower 侧新增 `done_topic` `"exploration_done"`。

## 3. 存图方案（用户决策）
本机 Foxy 无 nav2_map_server（`map_saver_cli` 不存在），故改为**自研纯 Python 写出 map_server 标准 `.pgm`+`.yaml`**。格式：`P5`、`maxval 255`、`negate: 0`、`occupied_thresh: 0.65`、`free_thresh: 0.196`、`origin` 取 `msg.info.origin`；该格式 Foxy 与真机 Humble 通用（见附七第 10 条）。

## 4. Part 1 单元测试（无 Gazebo）：36/36 通过
覆盖：PGM 头与逐像素回读比对、行序翻转、-1→205、yaml 六字段、尾随 `/`、`w/h<=0`、`len(data)<w*h` 不留半截文件、节点级两道门槛、连调 10 次触发 done、latched 订阅者收到 True、二次 bump 不重复、done 后不再规划/存图各 1 次。
脚本：`/tmp/step3_unit.py`。

## 5. Part 2 端到端（Gazebo + Cartographer）
- `colcon build` 4 包成功；`/map` 350x323 known=52987。
- 强制 `max_goal_count:=2` 触发完成；explorer 日志「探索完成（达到最大目标数 2）：停止发布目标与路径」与「已写出 /tmp/step3_maps/e2e.pgm 与 e2e.yaml」。
- done 后 16 s 内 `/cmd_vel` 采样 159 条全零；done 后再注入远处目标 (100,100) → 采样 99 条仍全零。
- **两种启动顺序**（先 follower / 先 explorer）下 latched 标志都能收到。
- e2e.pgm 350x324：unknown=59455、occupied=1031、free=52914。

## 6. 已知项与待办（非阻塞）
- 端到端是用 `max_goal_count:=2` **强制提前完成**来验证"完成链路"的；真实"跑到无前沿才完成"的场景只做了节点级单元验证，**Step 4 联调时应观察一次自然完成的完整闭环**。
- `map_save_prefix` 默认是相对路径 `maps/exploration_map`（落在启动时的 CWD）；Step 4 的统一 launch 建议传绝对路径。
- `regional_explorer/launch/frontier_explorer.launch.py` 仍未设 `use_sim_time`（Step 4 统一为 true）。

## 7. 复现方式
- 单元：`source /opt/ros/foxy/setup.bash && python3 /tmp/step3_unit.py`
- 端到端：`colcon build --symlink-install` → gazebo → cartographer → explorer（`max_goal_count:=2 map_save_prefix:=/tmp/step3_maps/e2e`）→ `ros2 launch path_follower path_follower.launch.py`
- 详细报告：`/tmp/step3_report.md`（临时文件，仅供追溯）

## 8. 残留阻塞
无。

---

# 附十、硬约束与「真实数据映射」基准（来源：9.15 组会文档，2026-10-19 截图）

## 1. 硬约束（不可违反）
- **分辨率 r = 0.05 m/像素 绝对固定。** 地图**尺寸**可以自适应增长，但像素的物理**尺度不能失真**：
  - 不得修改 `cartographer` 的 `occupancy_grid_node` 的 resolution；
  - 不得对 `/map` 做 resize / 插值 / 重采样；`frontier_explorer_node._to_explored_map()` 必须保持 **1:1** 映射（当前即如此，**禁止**改成缩放）；
  - 一切"自适应尺寸"只能在**以栅格（格）为单位的尺度**上做，不允许换算/改变像素尺度。
- 违反后果（用户原话）：后期接训练集的学习方法会出现很多问题。

## 2. 真实数据映射表（r = 0.05 m/格）
| 物理对象 | 核心参数项 | 真实物理值 | 映射到算法/仿真 |
| --- | --- | --- | --- |
| 激光雷达 LD14P | 有效测距范围 | 0.15 ~ 8.0 m | 最大探测半径 **160 格**，盲区 **3 格** |
| 激光雷达 LD14P | 视场角 / 频率 | 360° 全向 / 6 Hz | 全向观测；决策调度频率 **1 Hz** |
| 阿克曼底盘 R3/Mini | 运动约束模型 | 阿克曼转向（无自转） | 严惩大角度掉头点，避免狭小空间卡死 |
| 阿克曼底盘 R3/Mini | 前轮最大转角 α_max | ≈30° = π/6 (0.52 rad) | 限制规划曲率 κ_max = 1/R_min ≈ **3.57 m⁻¹** |
| 阿克曼底盘 R3/Mini | 最小转弯半径 R_min | ≈0.28 m | 栅格半径 ≈ **5.6 格** |
| 阿克曼底盘 R3/Mini | 车体安全包络半径 | ≈0.25 m | `heuristic` 中 safe_dist = **5 格** |

## 3. 当前实现与该基准的差异（Step 4 对齐清单）
- **雷达量程/频率**：`gazebo_modele/urdf/model.urdf` 的仿真雷达是 `0.06 ~ 12.0 m @10 Hz`（720 线、360°），与基准的 `0.15~8.0 m @6 Hz` 不一致 → 需决定"改 URDF 对齐"还是"注明仿真为近似"。
- **`nav_slam/config/cartographer.lua`**：`min_range 0.12` / `max_range 12.` 与真实雷达 8 m 不一致。
- **safe_dist**：基准要求 **5 格**；运行链路默认 `obstacle_safe_distance = 6.0`（单位=格）→ 应收敛到 5。
- **权重存在三套互不相同的值**：
  - `regional_legacy/config.py`：`HEURISTIC_BFS_WEIGHT=10` / `TURN=1` / `OBS=8`
  - `real_work/light_qifa.py::LightHeuristicConfig` 类默认：`1.0 / 0.35 / 2.0`
  - **运行实际生效**的是节点/launch 默认：`bfs_weight=8.0 / turn_weight=0.35 / obstacle_weight=1.0`
  → 基准明确要求"严惩大角度掉头点"，而 `turn_weight` 仅 0.35，**是最需要先确认的一处**。
- **距离归一化**：运行链路 `light_qifa.decide_frontier()` 用**相对归一化**（`dist / max(dist)`），会随地图尺寸自适应、不产生数值失真，但绝对物理含义随地图变大而漂移；`core/heuristic.py` 里写死的 `TEST_MAP_SIZE*1.5 = 300 格（=15 m）` **不在运行链路上**（仅参考实现）。
- **`config.py::SENSOR_RANGE = 12`（格）** 与基准的 160 格差距很大 → 需确认它只服务于合成训练环境（200×200 网格），而非真机映射。
- **分辨率**：基准 r=0.05 ✅ 与实测 `/map` 的 0.05 一致（见附六）。
- **决策调度频率**：基准 1 Hz ✅ 与 `replan_rate_hz = 1.0` 一致。

---

# 附十一、Step 4 实测结果（2026-10-19）

## 0. 结论速览
- ✅ **一条命令联调达成**：`ros2 launch bringup exploration_bringup.launch.py` 一次拉起 gazebo + cartographer + frontier_explorer + path_follower。
- ✅ **多轮闭环达成**：实测真实跑出「扫 → 算 → 跑 → 到达 → 出新目标 → 再跑」的循环。
- ⚠️ **"自然完成"（跑到没有前沿）未在窗口内观测到**：终验时完成由 `max_exploration_time_sec=600s` **超时**截断，截断时**仍有可达前沿且在产出目标**。完成判定机制本身已在 Step 3 单元级验证通过（见附九）。

## 1. 本轮改动清单
1. **新包 `ros2_ws/src/bringup/`** + `launch/exploration_bringup.launch.py`：一键联调入口，包含 gazebo_modele 与 nav_slam 的 launch，并直接起 `frontier_explorer_node` 与 `path_follower_node`；**全链路 `use_sim_time` 写死 True**（不暴露该参数，避免半仿真）；暴露 `algorithm_root / map_save_prefix / replan_rate_hz / obstacle_safe_distance / min_frontier_distance / max_linear_vel / goal_tolerance / no_frontier_confirm_count / max_exploration_time_sec / max_goal_count / headless`。
2. **新增 `min_frontier_distance`（默认 10 格 = 0.5 m，欧氏栅格距离）**：算法侧 `light_qifa.LightHeuristicConfig` 在 `valid_mask` 里用 `np.indices + np.hypot` 剔除与机器人直线距离过近的前沿；节点侧同度量兜底守卫（不发布、不计数）。四处默认值一致（算法字段 / 节点参数 / 两个 launch）。
3. **`obstacle_safe_distance` 运行默认 6.0 → 5.0**（真机基准 0.25 m = 5 格），同步 `light_qifa.LightHeuristicConfig` 与 `real_config.HEURISTIC_CONFIG`，全仓库源码不再有 6.0 残留。
4. **`gazebo.launch.py` 新增 `headless` 开关**：`headless:=true` 时用 `gzserver` 替代 `gazebo` 包装器、跳过 `joint_state_publisher_gui`，并 **条件清空 `DISPLAY`**（`GroupAction(scoped=False)`，实测生效）。bringup 已转传该参数。
5. **`_goal_count` 改为"只在目标坐标变化时计数"**：修正原来"每帧 1 Hz 重发都自增"导致 `max_goal_count=200` 约 200 s 就触顶、提前判完成的问题。
6. **`regional_legacy/config.py` 真机参数化**（见附十）：`SENSOR_RANGE` 12→160，新增 `GRID_RESOLUTION=0.05`、`SENSOR_BLIND_CELLS=3`、`SAFE_ENVELOPE_CELLS=5`、`MIN_TURN_RADIUS_M/CELLS`、`MAX_STEER_RAD`、`MAX_CURVATURE_M_INV`、`DECISION_RATE_HZ`；并修正 `core/heuristic.py` 里由 `SENSOR_RANGE*0.5` 派生的 `safe_dist`（改用 `SAFE_ENVELOPE_CELLS`）。

## 2. 联调命令（一条命令）
```
ros2 launch bringup exploration_bringup.launch.py \
  algorithm_root:=/home/wzy/robot_path_planning/regional_legacy \
  map_save_prefix:=/tmp/s4d_maps/exploration_map \
  headless:=true
```
（`headless:=true` 适配无显示/GLX 不可用环境；有显示时可省。)

## 3. 三轮失败 → 修复经过（很有价值，务必如实记录）
1. **第 1 轮**：探索器选出的前沿目标**贴在机器人身上**（仅差 ~2 格/10cm，落在控制端 0.25 m 到达容差内）→ 车第一帧即判"已到达"→ 零速 → 地图不涨 → 反复重发同一近目标 → 直到 `max_goal_count=200` 触顶的**假完成**。
2. **第 1 次修复**（引入新 bug）：算法侧把候选前沿按 **BFS 步数** 过滤 `distances >= 10`，而节点兜底守卫按**欧氏距离** `< 10` 拒绝 —— **两层度量不一致**，算法放行的合法前沿被守卫全部拒掉 → 日志刷 113 次"与机器人直线距离过近…跳过本轮"、**0 个目标**。根因是欧氏距离可能小于 BFS 步数（路径要绕）。
3. **第 2 次修复**：把距离条件在算法侧**统一改为欧氏栅格距离**（与节点守卫、与 `path_follower` 的到达判定三处同度量）→ **闭环达成**：20 个不同目标、净位移 4.045 m、`/cmd_vel` 非零占比 ~98%、峰值 0.30 m/s、地图 574×473→694×693、到达 3 次且之后继续出新目标；但完成仍由 `max_goal_count=200`（当时的错误计数方式）在 ~351 s 触发。
4. **终验（第 4 轮）**：计数修正为"坐标变化才计数" + headless 的 DISPLAY 清理改为 `scoped=False` 后重跑，见第 4 节。

## 4. 终验实测数据（本轮重点）
- **启动**：一条命令、未手工 `unset DISPLAY`；gzserver 进程环境里 `DISPLAY=` 已为**空**（不再 `:1`）→ headless 清理**生效**；四部件全部在线（gazebo + robot_state_publisher + 2×静态 TF、cartographer_node + occupancy_grid_node、frontier_explorer、path_follower）。
- **闭环**：不同目标数 **71**；净位移 ≈ **18.09 m**；累计路程 ≈ **63.39 m**；完成前 `/cmd_vel` 非零占比 **0.914**；`linear.x` 峰值 **0.300 m/s**（= vmax）。
- **地图增长**：`/map` 从 `504×354 / known 95721` → 完成时 `977×731 / known ≈ 320272`（单调增长）。
- **到达行为**：**2 次**（距离 0.243 m ≤ 0.25）→ 均归零 + 日志"已到达目标"，随后出现更远新目标并恢复行进（第二次恢复时新目标远 **9.388 m**）。
- **完成原因（未达成自然完成）**：`exploration_done` 在启动后约 **601.7 s** 触发，原因原样为 `探索完成（超过最大探索时长 600.0s）`。截断时不同目标仅 71（`max_goal_count` 未触及）、最后一帧目标距机器人 ≈14.13 m、**仍在持续产出目标** → 被超时截断，非真正无前沿。日志中"未找到可达前沿（第 n/10 轮）"全程**仅出现 1 次**，streak 从未累计到 10。
- **地图落盘**：`exploration_map.pgm` `980×731`（unknown 400428 / free 299226 / occupied 16726），yaml `res 0.05`、`origin [-34.161, -16.951, 0]`、`negate 0`（自研写出，见附九）。
- **稳定性**：`/odom` NaN 行数 0；日志 13686 行中 `[ERROR]=0`、`process has died=0`、`TF 查询失败=0`、`Traceback=0`。

## 5. 达成与未达成
- **达成**：一条命令起全套；多轮「扫→算→跑→到达→新目标」闭环；地图持续增长；到达即停并可继续；`headless` 可在无显示环境跑；稳定性良好。
- **未达成**：**"跑到没有前沿"的自然完成**。原因判断：3d.world 场景较大，600 s 不够；且 `max_exploration_time_sec` 只是安全阀，不该当作完成条件。

## 6. 已知项与待办（非阻塞）
- `max_exploration_time_sec=600` 对当前 world 偏小（600 s 内 71 个不同目标、地图仍在增长）→ 联调/真机建议上调（如 1800~3600），或按场景规模设定。
- `min_frontier_distance=10 格（0.5 m）` 为经验值，可按需调；调大能强制更远的探索步长，调小易回到"贴脸前沿"。
- 权重仍有两套：`config.py` 的 `HEURISTIC_BFS/TURN/OBS = 10/1/8` 与**运行实际生效**的节点默认 `bfs_weight=8.0 / turn_weight=0.35 / obstacle_weight=1.0`；后者距离项相对归一化 ⇒ 等于"偏好最近的前沿"（探索上合理，但"严惩大角度掉头"的意图偏弱）→ 待调参。
- 附十清单里**未对齐**项（用户指示本轮只按真机设定 config.py，故未动）：URDF 仿真雷达 `0.06~12.0 m@10 Hz` vs 真机 `0.15~8.0 m@6 Hz`；`cartographer.lua` 的 `min_range 0.12 / max_range 12.`。
- `config.py` 的 `SENSOR_RANGE=160`（真机值）会让**离线合成训练环境**的 FOV 视距失真（合成图为 200×200，160 格近乎一次揭示全图）；用户已声明训练后置，**恢复训练前必须处理**（建议拆出 `SYNTHETIC_FOV_RANGE` 供 `GridWorldEnv` 使用）。

## 7. 复现方式
- 一条命令见第 2 节；采样脚本与原始报告：`/tmp/s4d_bringup.log`、`/tmp/s4d_trace.csv`、`/tmp/step4d_report.md`、`/tmp/s4d_maps/`（临时文件，仅供追溯）。
- 前几轮报告：`/tmp/step4_report.md`、`/tmp/step4b_report.md`、`/tmp/step4c_report.md`。

## 8. 残留阻塞
无硬阻塞。唯一未闭合项是第 5 节的"自然完成"，需要更长的运行窗口或更小的 world。

---

# 附十二、仿真 IMU 接入 + 治漂移 + 治卡墙实测（2026-10-20）

## 0. 结论速览
- **漂移根治级改善**：连续帧类别翻转占比 **14.1% → 0.08%**；运动期 30s 间隔两帧墙体 IoU **0.142 / 0.024 → 0.947 / 0.903**；最优对齐额外平移量 **米级 → 0.00 m**。
- **cartographer 不再崩溃**（终验按仓库原样配置：`Check failed` / `colocated` / `process has died` / `exit code -6` 均 0 次）。
- **无永久物理卡墙**（偶发 1 次卡死被 1s 倒车自救解除）；闭环正常：到达 19 次 ↔ 新前沿 18 次，净位移 12.81 m / 累计 17.97 m。
- 终验 10/10 项通过；其中 3 条修复路径**运行期未触发**，仅代码核查（见第 5 节）。

## 1. 本轮改动清单
**IMU / SLAM**
- `gazebo_modele/urdf/model.urdf`：新增 `imu_link`（fixed joint 挂 `base_link`，origin `0 0 0.10 rpy=0`）+ `<gazebo reference="imu_link">` 下的 `imu_sensor`（100Hz）+ `libgazebo_ros_imu_sensor.so`（`~/out:=/imu`、`initial_orientation_as_reference=false`、`<frame_name>imu_link</frame_name>`）。
- `nav_slam/config/cartographer.lua`：`TRAJECTORY_BUILDER_2D.use_imu_data` false→**true**；新增 `imu_gravity_time_constant = 10.`；**`tracking_frame` 由 `base_link` 改为 `imu_link`**；`publish_frame_projected_to_2d` false→**true**（`published_frame="odom"`、`provide_odom_frame=false`、`use_odometry=false` 保持不变）。
- `nav_slam/launch/gmapping.launch.py`：remappings 追加 `('imu','/imu')`。

**规划 / 控制**
- `regional_legacy/core/a_star.py`：新增 `inflation_radius_cells`（**库默认 0 = 不膨胀**，保证 6 个离线/训练/评测调用点行为不变）；有 scipy 用 `distance_transform_edt`、无 scipy 用纯 numpy 8 邻域位移膨胀。
- `regional_legacy/real_work/light_qifa.py`：`decide_frontier` 的 `valid_mask` 新增**离障碍硬过滤**（候选前沿到最近障碍 ≥ `obstacle_safe_distance` 格）；并把已算出的距离变换**复用**给 `_obstacle_cost`（无 scipy 时是纯 Python chamfer，避免重复计算）。
- `ros2_ws/src/regional_explorer/regional_explorer/frontier_explorer_node.py`：新增参数 `inflation_radius_cells`（默认 5.0）并**显式传给 `plan()`**；`plan` 失败时用 `inflation_radius_cells=0` **重试**，重试成功则视为"仅膨胀不可达"、**不计入** `no_frontier_streak`。
- `ros2_ws/src/path_follower/.../path_follower_node.py`：**前向安全刹车**（正前方 ±35° 最小测距 < `brake_distance` 时，把线速度压到爬行 `brake_crawl_speed` 但**保留转向**）；刹车持续 > `brake_timeout_sec` 升级倒车自救；`/scan` 超时保护；**卡死检测 + 倒车自救**（含后向安全校验与倒车转向）。
- 两个 launch（`frontier_explorer` / `exploration_bringup`）暴露 `inflation_radius_cells`；`path_follower.launch.py` 暴露刹车与倒车相关新参数。

## 2. 关键踩坑（务必保留）
1. **致命**：Cartographer 启动即崩溃 `Check failed: sensor_to_tracking->translation().norm() < 1e-5 — The IMU frame must be colocated with the tracking frame.`（exit -6）。原因：`tracking_frame=base_link` 而 IMU 在 `base_link` 上方 0.10 m，平移 0.1 ≫ 容差 1e-5。**修法：`tracking_frame = "imu_link"`**（同一 frame 平移恒 0）。若以后把 IMU 改成与 base_link 共位，可改回。
2. **前向刹车不能把线速度与转向一起归零**：阿克曼必须靠线速度才能转向，一起归零 = 永久刹停；而且早期实现把刹车写成"发零速后 early return"，会**绕过**后面的卡死自救 —— "顶墙 + 想走走不动"这一最该自救的场景反而永远不自救。修法：只把 `linear.x` 压到 0.06 爬行、保留 `angular.z`，并让刹车持续超时后升级为倒车自救。
3. **A\* 库默认膨胀 5 会静默污染离线链路**：`plan()` 的 6 个调用点（`env/grid_env.py`、`tools/eval_policy.py`、`real_work/compare_light.py`、`real_work/test_light_qifa.py`、`core/separating_runtime.py`、ROS 节点）都没传该参数，默认 5 等于全体被静默改行为 → 训练/离线指标不可比。修法：**库默认 0，只有部署侧显式传 5**。
4. **膨胀可能把窄门封住**，使 `plan` 返回 None 被误记为"A* 未能规划"→ 连续 10 轮**误判探索完成**。修法：失败时用 `inflation_radius_cells=0` 重试，成功则本轮跳过且**不计数**。

## 3. 新增参数表
（表格：参数 | 默认值 | 含义）
- `frontier_explorer`：`inflation_radius_cells` | 5.0 | 障碍硬膨胀半径（格）=真机安全包络 0.25m
- `path_follower`：`scan_topic`|scan；`enable_safety_brake`|True；`brake_distance`|0.30；`brake_half_angle_deg`|35.0；`brake_crawl_speed`|0.06；`brake_timeout_sec`|4.0；`scan_timeout_sec`|1.0；`stuck_detect_sec`|3.0；`stuck_move_eps`|0.05；`reverse_speed`|0.12；`reverse_sec`|1.0；`enable_stuck_recovery`|True；`stuck_recover_cooldown_sec`|5.0；`reverse_brake_distance`|0.25；`reverse_angular`|0.4
- `a_star.plan`：`inflation_radius_cells` | **0（库默认）**，部署侧传 5

## 4. 终验实测数据（按仓库原样配置，751.5 s）
- 编译：5 包成功。cartographer：崩溃相关 grep 全 0，`Added submap 32`；`/map` 753 帧；TF `map→base_link` 1504 样本连续；`node info /cartographer_node` 订阅含 `/imu`。
- IMU：`/imu` 74932 帧、实测 `span_hz≈99.58`、`frame_id=imu_link`；TF `base_link→imu_link` = (0, 0, 0.100000)。
- 漂移：翻转 **0.08%**（旧 14.1%）；IoU **0.947/0.903**（旧 0.142/0.024）；额外平移 **0.00 m**（旧米级）。
- 卡墙：无 ≥30s 的永久物理卡死；1 次卡死被 1s 倒车自救解除；抽样目标到最近障碍（离线复算 15 帧）**最小值 = 5.00 格，全部达标**。
- 闭环：到达 19 次 ↔ 新前沿 18 次；净位移 12.81 m / 累计 17.97 m；完成方式=600s 超时。
- 稳定性：`/odom` NaN 0/74933；TF 报错 0；节点崩溃 0。

## 5. 残余问题与未触发路径（非阻塞）
- **[低] 一次 ~47s 的"到达等待"停顿**（非物理卡死，自行恢复）：上游探索器连续重发**步长 < 到达门限 0.25m** 的小幅新目标（如 -9.37→-9.57），而控制端 `_reached` 需要"新目标相对上次到达点位移 > 0.25m"才恢复行进 → 车零速等待约 47 s，直到出现位移 0.320 m 的新前沿才恢复。**根因是"探索器小步重发"与"控制端 reached 恢复门限"的耦合**。建议后续评估：把"新目标"判定改为相对**当前机器人位置**，或让探索器避免里程碑式小步目标。
- **[仅代码核查，运行期未触发]**：前向刹车、刹车超时升级自救、倒车后向取消、节点级 `inflation_radius_cells=0` 重试 —— 本轮车全程离正前方障碍 ≥0.30m，四条路径均未进入。**需在更窄的场景（含窄门/贴墙）里再验一次**。

## 6. 复现方式
- 一条命令：`ros2 launch bringup exploration_bringup.launch.py map_save_prefix:=/tmp/final_maps/exploration_map headless:=true`
- 原始报告与产物：`/tmp/final_report.md`、`/tmp/final_bringup.log`、`/tmp/verify_report.md`（上一轮，含 IMU 共位崩溃的原始报错）、`/tmp/final_maps/`

## 7. 残留阻塞
无。唯一未闭合项是第 5 节的 ~47s 到达等待，与四条未触发的修复路径。

---

# 附十三、封闭房间完整建图闭环达成（2026-10-21）

## 0. 结论速览
- ✅ **达成**：小车在封闭矩形房间里，从零开始自主探索，**自然完成**（不是超时截断）。
  - 完成时刻 sim **2147.56 s**，原因字符串原样：`探索完成（未找到可达前沿，地图已探索完毕）`。
  - 存图：`room6_no_frontier.pgm/.yaml`（579×420，res 0.05，origin `(-14.4707, -10.4855)`）。
- ✅ **覆盖度**：内净空 224000 格中，已知 **216480 格 = 96.64%**（自由 95.05%、障碍 1.60%），未知仅 **3.36%**。
  - 剩余未知 **100% 落在 4 根柱子 + 2 个台子的实体 footprint 内部**（物理不可达）；严格按"邻接自由可达区"统计只有 **69 格（0.03%）**。
  - **东侧 x>8 覆盖 48000/48000 = 100%**（此前两轮分别是 99.97% / 99.88% 未知）。
- ✅ **三种智能体分工**：CodeWriter 改代码、CodeReviewer 静态审查、TestEngineer 实测定位与验收。

## 1. 新增/改动清单
**环境（新增）**
- `gazebo_modele/world/room.world`：**封闭矩形房间**，内净空 **x∈[-14,14]（28 m）× y∈[-10,10]（20 m）**，墙厚 0.3 m / 高 2 m / 四角重叠闭合。内部只有**独立凸出物**：4 根 2×2 m 柱子 @`(±7,±5)`、2 个 3×1.2 m 台子 @`(0,±7)`，**无窄门/窄走廊**。车 spawn 在 `(0,0)`。
  - 选这个尺寸的理由：雷达量程 12 m，从原点看东/西墙 14 m（>12 → 开局是迷雾）、南/北墙 10 m（可见）、四角 17.2 m（迷雾）。若房间更小会被一次扫描看全，探索就没意义。
- `gazebo.launch.py`：新增 launch 参数 `world`（默认 `room.world`，用 `PathJoinSubstitution`），`world:=3d.world` 可切回原来的复杂建筑场景；`exploration_bringup.launch.py` 同步透传。

**算法（关键修复）**
- `regional_legacy/core/obstacle_distance.py`（**新建**）：`distance_to_obstacle()` 成为"到最近障碍距离"的**唯一实现**（有 scipy 用 `distance_transform_edt`，否则 chamfer 3-4）。`core/a_star.py` 与 `real_work/light_qifa.py` 都改用它。
- `core/a_star.py`：膨胀判据统一为 **`dist < radius` 阻挡 / `dist >= radius` 放行**，与决策侧候选条件 `dist >= obstacle_safe_distance` 在 `== R` 时边界一致；删除原"无 scipy 时 8 邻域方形膨胀"（Chebyshev）与"起点/终点落膨胀区就静默退回未膨胀路径"的兜底（失败直接返回 None，由调用方显式处理）。
- `real_work/light_qifa.py`：`_distance_to_obstacle` 改为委托共享实现（保留旧名兼容）；距离变换结果复用给 `_obstacle_cost`，避免重复计算。

**探索器 `frontier_explorer_node.py`**
- `plan` 在硬膨胀下失败时，**接受零膨胀路径并正常发布**（节流 info `改用零膨胀路径发布`），不再"只打日志然后 return"。
- 黑名单把所有候选都排除时**清空黑名单**（info `清空黑名单以便继续探索`），本轮仍不计 streak。
- 两种"不该计入探索完成"的情形：**机器人所在格不是自由格**、**候选仅被临时黑名单排除** → 不计 `no_frontier_streak`（此前会被误当"没有前沿"，10 轮后永久提前 done）。
- **目标迟滞** `target_hysteresis_sec=10s`：目标仍可达就沿用，不再中位 5.3~6.35 s 换一次目标。
- 目标/黑名单改存**世界坐标**（`_current_target_world`），不再用旧栅格行列（`/map` 的 origin 会随建图漂移）。
- `max_exploration_time_sec` 默认 600 → **1800**（三处：节点 + 两个 launch）；存图文件名带完成原因后缀 `_no_frontier` / `_timeout` / `_max_goals`。

**控制器 `path_follower_node.py`**
- **后向安全判定改为正后方窄锥形** `reverse_half_angle_deg=30`（只统计 `|angle| > 150°`）。原来是整个后半区 `|angle| > 90°`，会把**与车身平行**的柱面（约 0.2 m）误判成"车尾障碍"。
- **脱困改为两相位三点掉头**：`reverse` 相 `(-escape_speed, +dir*escape_angular)`、`forward` 相 `(+escape_forward_speed, +dir*escape_angular)`（**同号**）；每个控制周期必定输出非零指令，**绝不发零速**。
- `stale_timeout_sec` 参数化（默认 8.0，原来是模块常量 3.0）：因为本机无 scipy 时单轮 `decide_frontier` 要 2.4~4.6 s。

## 2. 关键踩坑（务必保留）
1. **两层过滤的度量必须统一**：决策侧用 chamfer 近似欧氏、A\* 侧用 Chebyshev 方形膨胀 → 对 `dx=dy=4` 的格子，决策算 5.66 放行、A\* 算 4 阻挡，**目标格本身被膨胀盖住** → `plan(inflation=5)` 对所有候选返回 None。实测后果：目标发布永久停止（只打 `该前沿在当前膨胀半径下不可达` 282 次），`/cmd_vel` 在 8 s stale 后恒 0，**零速 1137.7 s**、未知 66.47%、东侧 99.88% 未知、到达 0 次。修法就是收敛成单一份 `distance_to_obstacle` + `dist < R` / `dist >= R` 的严格边界。
2. **"规划失败"的分支不能只打日志**：必须要么换候选、要么接受兜底路径并继续发布。否则同一目标每轮失败 → 上游停止发目标 → 下游 stale 零速 → 死锁。
3. **后向判定不能用整个后半区**：贴柱/贴墙平行停放时，侧方柱面会落进 `|angle|>90°`，被误判成"车尾障碍"，让脱困退化成只发零速；而阿克曼**不能原地转向** → 永久死锁（那次实测 180 次脱困全部打满 10 s 超时）。
4. **`gazebo_ros_ackermann_drive` 的 yaw 符号**：插件先按 `steer = atan(L*w/v)` 反解前轮转角，再由转角产生横摆，等效 `yaw_rate = v*tan(steer)/L == w`。所以**倒车时转角自动反向、yaw_rate 仍等于 w、符号不变** → 三点掉头的两个相位必须用**同号** `angular.z` 才能单向累积；取反号会正负相消（实测 77 次脱困航向净变化中位仅 0.07 rad）。
5. **`escape_angular` 要让反解转角不超 `max_steer`**：`w=0.6, v=0.15` 时 `steer=atan(0.2*0.6/0.15)=0.674 > 0.6` 被截断；取 `w=0.35` 时 `steer=0.437`，可达。
6. **相位时长要大于转向执行器滞后**：`steering_pid_gain` 只有 0.1，实测 2 s 才到目标的约 37%。`shuttle_sec=0.8` 根本来不及收敛（0.8+0.8 s 周期净转仅 0.046 rad ≈ 2.6°）；改 2.5 s 后平均 **11.0°/周期（改善 4.18 倍）**。
7. **PID 只能低 P**：高 P（速度 1000 / 转向 1500）会让 `/odom` **100% NaN、整车冻结**；低 P（0.1 / 0.01、D=0）实测自由空间指令 0.15 m/s → 实测约 0.125 m/s（比值 0.83）。**D 必须为 0**。

## 3. 三轮迭代（失败 → 修复 → 达成）
| 轮次 | 结束方式 | 内净空未知 | 可达但未知 | 东侧未知 | 到达次数 | 不同目标 | 最长停滞 |
|---|---|---|---|---|---|---|---|
| room4（复杂建筑 3d.world） | 超时 2400 s | 38.73% | 38.21% | 99.97% | 2 | 160 | 799.2 s |
| room5（房间，修窄锥形+三点掉头后） | 无 done（人工终止，已确定性死锁） | **66.47%** | 66.47% | 99.88% | 0 | 4 | 1102.0 s |
| **room6（统一度量后）** | **自然完成 2147.56 s** | **3.36%** | **1.75%**（实体内部；可达区仅 0.03%） | **0.00%** | 7 | 112 | — |

room5 比 room4 更差，正是因为"膨胀失败 → 只打日志 return"这条新分支把目标发布彻底掐断；修掉后才达成。

## 4. room6 终验数据
- 运行 2496.4 s 墙钟 / 2490.0 s 仿真；自然完成于 sim 2147.56 s。
- 不同目标 **112 个**、到达 **7 次**、中位换目标 **6.35 s**（< `stale_timeout_sec=8`）。
- 净位移 **10.276 m**、累计路程 **109.699 m**；`/cmd_vel` 非零占比 **83.68%**（完成前有效阶段 **99.25%**），`linear.x` 峰值 **0.300 m/s**。
- `/map` 已知格增长：55027 → 186210 → **220304**。
- `改用零膨胀路径发布` 32 次，均有后续目标与运动；`清空黑名单` 0 次（未触发）。
- 稳定性：`/odom` NaN 0、TF 报错 0、cartographer CHECK 0、运行期崩溃 0。

## 5. 脱困有效性（开环实测）
- 修复前（`shuttle_sec=0.8`、前进相反号）：0.8+0.8 s 周期净转 **0.046 rad ≈ 2.6°**，77 次脱困全打满 10 s 超时，累计浪费 **773.9 s（31%）**。
- 修复后（同号 + `shuttle_sec=2.5` + `escape_angular=0.35`）：3/3 周期净转均为正，平均 **0.1925 rad ≈ 11.0°/周期**（改善 **4.18 倍**），`escape_timeout_sec=10 s` 约能转 **22°**。
- 残留：前进相需把前轮从约 -0.437 摆到 +0.437，2.5 s 只够刚跨过零点，因此前进相仍可能小幅抵消。**根本瓶颈是 `steering_pid_gain=0.1` 太低**（纯倒车实测 yaw rate 仅为指令的 38.1%，而转角已达目标的 99.5%，说明主要亏损来自线速度而非转角未收敛）。

## 6. 已知限制与后续建议（非阻塞）
- **若想进一步缩短耗时**：把 `steering_pid_gain` / `linear_velocity_pid_gain` 在**低 P 区间内**（例如 1~5）小步上调，并每组都做开环标定；**绝对不要**直接跳到 1000/1500（会 NaN 冻结/翻滚）。
- **本机没有 scipy**：`decide_frontier` 单轮 2.4~4.6 s（纯 Python BFS + chamfer），这也是要靠 `stale_timeout_sec=8` 兜的原因。装 scipy 后两条慢路径都会走 EDT/C 实现，能显著提速。
- `A*` 仍是 **4 邻接 + Manhattan 启发式**（路径是 90° 折角），靠 `lookahead_dist` 平滑；若后续要更顺的轨迹，可考虑 8 邻接 + 后处理。
- 复杂建筑场景（`3d.world`）仍未跑通：该场景有窄门/窄走廊，而当前控制器**没有真正的局部绕障**（只有前向刹车 + 脱困）。要在那里跑通，需要加基于 `/scan` 的局部避障（自由空间法/沿墙绕行）。

## 7. 复现方式
```
ros2 launch bringup exploration_bringup.launch.py \
  world:=room.world \
  map_save_prefix:=/tmp/room6_maps/room6 \
  max_exploration_time_sec:=3000 \
  headless:=true
```
预期：约 35~42 分钟后出现 `探索完成（未找到可达前沿，地图已探索完毕）`，并生成 `<prefix>_no_frontier.pgm/.yaml`。
原始报告：`/tmp/room6_report.md`、`/tmp/escape_report.md`、`/tmp/shuttle_report.md`；前几轮：`/tmp/room4_report.md`、`/tmp/room5_report.md`。

## 8. 残留阻塞
无。完整建图闭环已在封闭房间场景达成；唯一未闭合项是第 6 节的"复杂建筑场景仍需局部避障"。

---

# 附十四、真机接入前必须在真机上调查/确认的清单

## 0. 使用说明
本清单用于**真机到手后、接终端做第一次细致调试**时逐项落实。原则：
- **仿真结论不能直接搬到真机**：仿真里 IMU 是"真值"、雷达无噪声、底盘参数是估的。凡下表标注"待实测/待确认"的，必须真机实测后回填。
- 每完成一项就在 `[ ]` 里打 `x`，并把实测值填进第 10 节的回填表。
- 本清单之外不要急着改算法参数；先把"接口 + 物理参数 + 标定"三件事弄清楚。

---

## 1. 硬件与型号调查（先读说明书/铭牌）
- [ ] **雷达**：型号（此前按 LD14P 记录）、厂商、**说明书里的准确量程/角分辨率/扫描频率/盲区/接口（串口还是网口）**、供电电压、驱动/固件版本、需要哪个 ROS2 驱动包、默认波特率/端口
- [ ] **IMU**：型号、厂商、**输出频率**、**输出内容**（是否含姿态四元数 / 仅角速度+线加速度）、接口（I2C/SPI/串口/USB）、是否需要外部标定、**安装朝向标记**
- [ ] **底盘/底盘驱动板**：型号（此前记 R3/Mini）、通信协议（串口/CAN）、协议文档、是否需要专用驱动节点、**是否提供轮式里程计**
- [ ] **编码器**：是否有、分辨率（每转脉冲数）、是否被底盘驱动板积分成里程计
- [ ] **电机与舵机/差速**：这是**阿克曼**还是差速底盘？（决定 `MAX_STEER_RAD`/`MIN_TURN_RADIUS` 是否有意义）
- [ ] **电源**：电池电压/容量、计算平台（树莓派/工控机）型号与 ROS2 版本（**确认是不是 Foxy**）、主控与底盘通信的物理连线
- [ ] **急停**：有没有物理急停按钮？没有的话第一次跑车必须**架空轮子**

## 2. 数据接口调查（最关键，接线后第一件事）
接线后逐条确认，把**实际**结果记下来：
- [ ] `ros2 topic list`：实际有哪些话题？特别是雷达、IMU、底盘里程计/控制的话题**真实名字**
- [ ] 每个话题的 **消息类型** 与 **发布频率**（`ros2 topic hz`）
- [ ] 每条消息的 **`header.frame_id`** 到底是什么（不要假设是 `laser`/`imu_link`/`base_link`）
- [ ] **QoS**（`ros2 topic info -v`）：是 reliable 还是 best_effort？传感器常用 best_effort，订阅端要匹配
- [ ] `ros2 run tf2_tools view_frames`：**实际 TF 树**长什么样？有哪些 frame？谁是根？
- [ ] 底盘驱动是否**自己发布** `odom→base_link`（或 `odom→base_footprint`）的 TF？还是只发 `/odom` 话题？
- [ ] 控制入口：接受 `/cmd_vel`（Twist）还是自定义消息（如 AckermannDrive）？**速度单位与符号约定**（正=前进？角速度正=左转？）
- [ ] 时间源：真机**没有 `/clock`**，全链路必须 `use_sim_time=false`

## 3. 关键物理参数实测（卷尺 + 直尺，十分钟能测完）
- [ ] **轴距 L**（前轮中心到后轮中心，米）——直接决定 `MIN_TURN_RADIUS = L / tan(MAX_STEER)`
- [ ] **轮距**（左右轮中心距）
- [ ] **轮径**（或轮胎直径）
- [ ] **车体外形尺寸**（长×宽）与**最大安全包络半径**——决定 `SAFE_ENVELOPE_CELLS`（现按 0.25 m = 5 格）
- [ ] **前轮最大转角**（左右各多少度，实测打满方向看角度或读驱动反馈）——对应 `MAX_STEER_RAD`
- [ ] **雷达安装位置**：相对车体中心/`base_link` 的 `x, y, z`（米）与**朝向 rpy**（雷达 0° 是否指车头？有无装反/装偏）
- [ ] **IMU 安装位置与朝向**：相对 `base_link` 的 `x, y, z` 与 rpy。**重点确认 z 轴是否朝上、与车头是否同向**
- [ ] 车体最低离地间隙（判断能否过门槛/地毯）

## 4. 传感器标定与实测（用数据说话）
- [ ] **雷达静态噪声**：车静止、空旷场地，连续 60 s 采集 `/scan`，给出同一角度上测距的**标准差**
- [ ] **雷达实际有效量程**：对着墙从近到远，找出"可靠最大距离"与"最小盲区"（此前基准写 0.15~8.0 m）
- [ ] **雷达视场角**：`angle_min`/`angle_max` 是否真是 360°？有无被车体遮挡的扇区
- [ ] **IMU 静止零偏**：静止 60 s，记录 gyro 三轴均值（零偏）与标准差（噪声）
- [ ] **IMU yaw 漂移**：静止 5 分钟，积分 yaw 漂移多少度/分钟
- [ ] **IMU 朝向正确性**：把车头抬起（绕 y 轴）看 pitch 符号、绕 x 轴看 roll 符号是否符合右手系；**原地顺时针转 90°，看 yaw 是增还是减**
- [ ] **IMU 是否给出绝对姿态**：上电时 yaw 是否为 0、是否以世界系为参考

## 5. 里程计可信度实测（决定 `use_odometry` 能不能开）
- [ ] **直线 3~5 m**：实际前进 5 m（卷尺量），`/odom` 报多少？误差百分比
- [ ] **原地转 360°**：`/odom` 报的累计角度误差多少度
- [ ] **`/odom` 与 TF 是否一致**：让车动起来，对比 `/odom.pose` 与 TF `odom→base_link`，**是否同步**（仿真里曾出现 `/odom` 位姿冻结的问题）
- [ ] 结论：`use_odometry` 该开还是该关（**若里程计误差 >5% 或与 TF 不同步，就先关**）

## 6. 需要在真机上改的配置（调查完再动手）
按依赖顺序，**先接口、后参数**：
- [ ] **新增"真机 launch"**：不能复用 `exploration_bringup.launch.py`（它 include 了 Gazebo，且 `use_sim_time` 写死 `True`）。需要一份只起 驱动 + Cartographer + explorer + follower 的 launch，`use_sim_time=False`
- [ ] **`cartographer.lua`**：按实测改 `TRAJECTORY_BUILDER_2D.min_range` / `max_range`（真机基准 0.15 / 8.0；仿真里是 0.12 / 12，**必须改**）；按第 5 节结论决定 `use_odometry`；`tracking_frame` 必须与 IMU 共位（仿真里因为 IMU 装在 base_link 上方 0.10 m，把 `tracking_frame` 改成了 `imu_link`——**真机要按实际安装位置重算**，否则会触发 `Check failed: The IMU frame must be colocated with the tracking frame`）
- [ ] **`config.py` 的 `SENSOR_RANGE`**：现为真机值 160，会让**离线合成训练环境**的 FOV 失真（合成图 200×200）。**恢复训练前必须拆出独立的 `SYNTHETIC_FOV_RANGE`**
- [ ] **URDF/机器人描述**：把仿真 URDF 里的**估测参数**换成第 3 节的实测值（轴距、轮径、轮距、最大转角、雷达/IMU 外参）
- [ ] **底盘驱动节点**：确认它吃 `/cmd_vel`（Twist）还是 Ackermann 消息；如果是后者，需要一层转换
- [ ] **雷达驱动**：装厂商/社区驱动，把 `frame_id`、`angle_min/max`、`range_min/max`、`scan_time` 配上，频率对齐（真机 6 Hz vs 仿真 10 Hz）
- [ ] **IMU 驱动**：把输出 remap 成 `imu` 且 `frame_id` 与 TF 树里的 IMU link 一致，频率 ≥ 100 Hz（Cartographer 需要）
- [ ] **PID/速度限制**：真机不要延用仿真的低 P；先设保守限速（如 0.15 m/s）、开前向刹车与卡死自救
- [ ] **存图路径**：`map_save_prefix` 改成真机可写目录（如 `~/.ros/maps/...`）

## 7. 首次上电/首跑的顺序（安全第一）
按顺序执行，**任一步异常就停下**：
1. [ ] 只接电源与底盘，**架空轮子**，手发 `/cmd_vel` 看轮子转向与符号是否正确（正=前进）
2. [ ] 接雷达，`ros2 topic hz /scan` 确认频率；用 `rviz2` 看 `/scan` 点云方向是否正确（车头朝前）
3. [ ] 接 IMU，静止看零偏；手动转车看 yaw 方向
4. [ ] **手推车**（不开电机）跑 Cartographer，看 `map→base_link` 是否跟手推轨迹一致、建图是否闭合（这一步能排除大部分 TF/外参问题）
5. [ ] 低速自动跑（限速 0.10~0.15 m/s），人在旁边随时急停
6. [ ] 逐步放开限速与探索参数

## 8. 与仿真已知差异（真机上要特别小心这几条）
| 项 | 仿真现状 | 真机基准/注意 |
|---|---|---|
| 雷达量程 | 0.06~12.0 m @10 Hz | 0.15~8.0 m @6 Hz，**`max_range` 必须从 12 改到约 8** |
| IMU | Gazebo 插件，**几乎无噪声的真值** | 有零偏/漂移/噪声；**仿真结果不能当调参结论** |
| 里程计 | ackermann 插件 `/odom` **位姿冻结/与 TF 不同步**，实际用 `use_odometry=false` | 若底盘有可靠里程计，建议开启以抑制长直走廊的平移漂移 |
| 底盘 PID | 只能低 P（转向 0.1 / 速度 0.01），**高 P 会 `/odom` 全 NaN 冻结** | 真机 PID 在底盘驱动板里，需按说明书整定 |
| `use_sim_time` | 全链路写死 `True` | 必须 `False` |
| 网关/TF | 静态恒等 `odom→base_footprint` | 真机应由底盘或 SLAM 提供，**不要同时有两份** |
| 探索耗时 | RTF 3 倍速下墙钟约 33 min | 真机是实时，建议先用**小场地/小房间**验证闭环 |

## 9. 调查产出要求
调查完成后，需要形成三份东西：
1. **回填表**（第 10 节）填满；
2. 一份**真机 launch**（第 6 节第一条）；
3. 一份**改动清单**（改了哪些参数、为什么、实测依据是什么），写回 AGENTS.md 新的附节。

## 10. 回填表（2026-10-21 真机勘察回填）

> 说明：本表区分「已实测/已从固件或 URDF 确认」与「待实测」。标 **待实测** 的项需要在车轮架空或安全场地条件下补测。
> 本轮**未启动底盘驱动、未发布 `/cmd_vel`、未运行 Cartographer/探索/控制**；仅在真车上做了只读串口监视（`wheeltec_keyboard.py --monitor`，不发任何字节）。

| 项目 | 符号/参数 | 实测值 | 依据/备注 |
|---|---|---|---|
| 轴距 | `L` (m) | **0.144** | mini_akm URDF 前后轮中心 x 差 0.14325；固件 `Akm_axlespacing=0.144`；卷尺复核 |
| 轮距 | (m) | **0.162** | 固件 `Akm_wheelspacing=0.162`；URDF 轮中心 y 约 0.1595/0.1642 |
| 轮径 | (m) | **≈0.064（半径 0.032）** | URDF 后轮中心 z≈0.0352、前轮 z≈0.030，取 ~0.032；卷尺实测 |
| 车体长×宽 | (m) | **待实测** | mini 级车体，需卷尺 |
| 前轮最大转角 | `MAX_STEER_RAD` (rad) | **28° = 0.489** | `wheeltec_keyboard.py --max-angle 28`；固件 `MINI_AKM_MAX_ANGLE`；打满复核 |
| 最小转弯半径 | `MIN_TURN_RADIUS_M` (m) | **0.350** | 固件 `MINI_AKM_MIN_TURN_RADIUS`；`L/tan(28°)=0.271` 为几何下限，以固件 0.350 为准 |
| 安全包络半径 | `SAFE_ENVELOPE_CELLS` (格) | **0.25 m = 5 格（保守沿用）** | 项目基线；实际 mini 车体更小，待实测车体后复核 |
| 雷达型号 | — | **LD14（LD14P）** | `/etc/udev/rules.d/ld14_lidar.rules`；`/root/ld14_ws` 日志 `FOUND LiDAR_LD14`；**不是** `wheeltec_param.yaml` 里的 LS M10P |
| 雷达量程 | `min_range`/`max_range` (m) | **驱动 0.15 / 30.0；物理有效 8.0（基线）** | `/root/ld14_ws/src/ld14/src/main.cpp`；AGENTS 基准 0.15~8.0；待实测 |
| 雷达频率 | (Hz) | **待实测（基线 6 Hz）** | `slbf.h SCAN_FRE=2300` 非直接 Hz；插雷达后 `ros2 topic hz /scan` |
| 雷达 FOV | `angle_min`/`angle_max` | **360°** | LD14 全向；驱动 `angle_min=raw-π`、`angle_max=raw-π`；0° 方向需 RViz/实物确认 |
| 雷达外参 | `xyz` / `rpy` | **xyz=(0.051343, -1.2576e-05, 0.094524)；物理安装 rpy=(0,0,π)；逻辑 `laser` 帧 rpy=(0,0,0)** | `robot_model.yaml mini_akm base_to_laser`；URDF `laser_joint` 的 rpy=3.1416；驱动已做 -π 角度偏移；**首次运行必须确认扫描 0° 指车头** |
| 雷达静态噪声 | std (m) | **待实测** | 插雷达、静止 60 s |
| IMU 型号 | — | **STM32 底盘板载（`imu_mode: stm32`）** | `wheeltec_param.yaml`；无独立 IMU 驱动 |
| IMU 频率 | (Hz) | **≈20 Hz** | 底盘回传 20.2 Hz（只读监视实测）；驱动 `Publish_ImuSensor` 在串口接收循环内；**低于基线 100 Hz**，需评估 Cartographer 表现 |
| IMU 零偏 | gyro x/y/z (rad/s) | **待实测** | 静止 60 s |
| IMU yaw 漂移 | (°/min) | **待实测** | 静止 5 min |
| IMU 外参 | `xyz` / `rpy` | **xyz=0, rpy=0（`gyro_link` 与 `base_footprint` 共位）** | `robot_model.yaml base_to_gyro`；物理 IMU 在控制板，URDF `controller_link≈(-0.000357,-0.04684,0.0515)`，2D 忽略 |
| IMU link 与 base_link 是否共位 | 是/否 | **否；`gyro_link` 与 `base_footprint` 共位，`base_link` 高 0.019 m** | `robot_model.yaml base_to_link z=0.019`；Cartographer 应取 `tracking_frame="gyro_link"` |
| 里程计直线误差 | (%) | **待实测** | 5 m 手推/低速 |
| 里程计转角误差 | (°) | **待实测** | 360° |
| `/odom` 与 TF 是否同步 | 是/否 | **驱动只发 `/odom`，不发 TF；需 EKF 或 `odom_tf_broadcaster`** | `wheeltec_robot.cpp Publish_Odom()` 无 `sendTransform`；EKF 配置 `publish_tf: true`、`odom_combined→base_footprint` |
| 底盘控制接口 | 话题/消息类型 | **`/cmd_vel` `geometry_msgs/Twist`；固件阿克曼，Vx(m/s)、Vz(rad/s)** | `base_serial.launch.py akm_cmd_vel=none`；`Cmd_Vel_Callback` 直接写 0x7B 帧；不能原地转，R≥0.35 m |
| 底盘速度上限 | (m/s) | **0.5（键盘脚本上限）；首次建议 0.15** | `wheeltec_keyboard.py --max-speed 500 mm/s`；固件 |
| 串口（底盘） | 设备 | **`/dev/wheeltec_base`→`ttyACM1`；`/dev/wheeltec_controller` 已建别名** | CH343 `1a86:55d4`，serial `5B0B028473`；115200 8N1；11 B 下发 / 24 B 回传 |
| 串口（雷达） | 设备 | **udev 别名 `/dev/LD14`；serial `5A6D013855`；服务 `ld14-lidar.service`** | 当前雷达未插，服务 inactive；插上后 udev 自动启动 |
| 底盘反馈 | 频率/电压 | **20.2 Hz / 12.00 V / 电机使能 / acc_z 9.75** | `wheeltec_keyboard.py --port /dev/wheeltec_base --monitor` 只读实测（未发字节） |
| 丢帧保护 | 安全 | **固件未启用：停发指令会保持最后速度** | 厂商 README/固件源码；必须加驱动看门狗/软件零速兜底，并保证电机使能开关可随时关断 |
| 关键依赖 | 运行 | **`robot_localization` / `joint_state_publisher` / `turtlesim` 未安装** | apt 尝试时 `robot-localization` 404；厂商全量 launch 暂不可直接用；本栈可绕开 EKF 与 joint_state_publisher |

---

# 附十五、真机 RDK-X5 勘察结果（2026-10-21，只读勘察，未运行底盘）

## 0. 结论速览

- 真机为 **WHEELTEC mini_akm 阿克曼**（前轮转向，不能原地转），不是厂商配置文件里默认的 `mini_mec`。
- 底盘串口 `/dev/wheeltec_base`（CH343，serial `5B0B028473`）在线，回传 **20.2 Hz、12.00 V、电机使能、acc_z≈9.75**。
- 雷达为 **LD14（LD14P）**，独立驱动 `/root/ld14_ws`，发布 `/scan`（frame `laser`）；当前雷达**未插/未上电**，服务 inactive。
- IMU 为底盘 STM32 板载，`/imu/data_raw`（frame `gyro_link`），**约 20 Hz**；不是独立 100 Hz IMU。
- 驱动只发 `/odom`（frame `odom_combined`→`base_footprint`），**不发 TF**；需要 EKF 或自研 `odom_tf_broadcaster` 补 `odom_combined→base_footprint`。
- 厂商配置 `car_mode: mini_mec` 与实车不符，**已改为 `mini_akm`**（只改配置，未运行）；`lidar_type: ls_M10P_uart` 也与实车 LD14 不符，**不要使用 `wheeltec_lidar.launch.py`**。
- 本轮没有启动底盘驱动、没有发布 `/cmd_vel`、没有运行建图/探索/控制；只做了只读串口监视和文件勘察。

## 1. 环境与连接

| 项 | 值 |
|---|---|
| 主控 | D-Robotics RDK X5 V1.0，aarch64 |
| 系统 | Ubuntu 22.04.5 LTS，kernel 6.1.83 |
| ROS | `/opt/ros/humble`（Humble） |
| 资源 | 8 核 / 3.0 GiB RAM / 根分区 57 G（已用 11 G） |
| Python | 3.10.12；numpy 1.26.4；scipy 1.10.1 |
| colcon | 已安装（0.21.3） |
| 主机名/用户 | `ubuntu` / `sunrise` |
| 当前网络 | 手机热点 `OnePlus Ace 3`，`wlan0 = 192.168.242.226/24` |
| SSH | `ssh rdk-t`（`~/.ssh/config` 已指向 192.168.242.226，`id_rsa` 可登录） |

> 校园网 `SJTU` 对客户端做隔离：真车与本机虽然都在 `10.180.0.0/15`，但 IPv4/IPv6 单播互不可达。真机联调必须让两端处于同一可互通局域网（手机热点/路由器/网线），或使用 Tailscale 等 overlay。热点是 DHCP，IP 变化后需更新 `~/.ssh/config` 的 `HostName`。

### 1.1 SSH/网络实测明细

- 校园网阶段：真车 `wlan0=10.180.94.135/15`（SSID `SJTU`），本机 `wlp8s0=10.180.183.221/15`；真车 ping 本机、本机 ping 真车（IPv4 / link-local IPv6 / global IPv6）全部不可达，确认是**客户端隔离**。
- 手机热点阶段：SSID `OnePlus Ace 3`，本机 `192.168.242.234/24`，网关 `192.168.242.112`，真车 `192.168.242.226/24`；`nc -vz 192.168.242.226 22` 成功，banner `SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.13`。
- 本机 `~/.ssh/config`：`Host rdk-t` → `HostName 192.168.242.226`、`User sunrise`、`IdentityFile ~/.ssh/id_rsa`、`StrictHostKeyChecking accept-new`、`ServerAliveInterval 30`、`ServerAliveCountMax 3`；`ssh rdk-t` 免密登录成功。
- `~/.ssh/known_hosts` 里仍有旧地址 `192.168.1.105`、`192.168.105.226` 的 host key；新地址 `192.168.242.226` 已追加，不影响登录。
- 热点下 `192.168.242.0/24` 走 `wlp8s0` 直连，未被 Mihomo TUN 劫持；但 Mihomo 的 fake-ip DNS 会拦截 `.local` 与外部域名，不要用 `getent hosts` 判断真车地址。
- 真车当前主机名是 `ubuntu`（不是 `rdk-t`）；`rdk-t` 只是本机 SSH 别名。

## 2. 底盘与串口（安全关键）

- 车型：**mini_akm（阿克曼）**，轴距 0.144 m、轮距 0.162 m、前轮最大转角 28°、最小转弯半径 0.350 m、轮半径约 0.032 m（URDF 推算，待卷尺复核）。
- 串口：CH343 USB 串口 `1a86:55d4`，serial `5B0B028473`，设备 `ttyACM1`；已有 `/dev/wheeltec_base`，本轮新增 `/dev/wheeltec_controller` 别名（厂商 `base_serial.launch.py` 使用）。
- 协议：115200 8N1；下发 11 字节（`0x7B` 头、Vx/Vy mm/s、Vz×1000、BCC、`0x7D` 尾）；回传 24 字节，20 Hz。
- 只读实测：`帧=~20.2 Hz，Vx=0，Vy=0，Vz=0，电池=12.00 V，电机=使能(可动)，acc_z=9.74~9.77`。
- **安全红线**：当前固件把丢帧保护注释掉了——停止发指令后底盘会**保持最后速度**。因此：
  - 任何启动脚本退出/Ctrl-C 必须补发零速；
  - 驱动侧需要增加「超过 0.5 s 没收到 `/cmd_vel` 就发零速」的看门狗；
  - 首次实跑必须架空轮子或留出安全空间，并保证电机使能开关/急停可随时关断；
  - 上电后约 10 s 内底盘不响应串口（固件 `CONTROL_DELAY=1000`）。
- 控制语义：`/cmd_vel` 为 `geometry_msgs/Twist`，`linear.x` 为正前进（m/s），`angular.z` 为逆时针（rad/s）；底盘固件按阿克曼反解前轮转角，**不能原地转**，且要求 `R = Vx/Vz ≥ 0.350 m`，否则固件自动缩小 Vz。
  - 真机建议：`max_linear_vel=0.15`、`max_angular_vel≈0.4`（0.15/0.4=0.375 m ≥ 0.35）；若用 0.6 rad/s，理论 R=0.25 m，会被固件限幅。
  - `min_linear_vel` 必须保留（阿克曼没有线速度就无法转向），现有 `0.06~0.08` 合理。

### 2.1 URDF 关节与外参明细（`mini_akm_robot.urdf` 解析）

| 关节 | 类型 | parent → child | origin xyz (m) | origin rpy | 说明 |
|---|---|---|---|---|---|
| `lb_joint` | continuous | `base_link`→`lb_link` | (-0.069657, 0.079750, 0.0162) | 0 | 左后轮 |
| `rb_joint` | continuous | `base_link`→`rb_link` | (-0.069657, -0.079776, 0.0162) | 0 | 右后轮 |
| `front_joint` | fixed | `base_link`→`front_link` | (0.143830, -1.2575e-05, 0.010) | 0 | 前部固定件（**不是前轴**） |
| `controller_joint` | fixed | `base_link`→`controller_link` | (-0.00035718, -0.04684, 0.0515) | 0 | 控制板 |
| `laser_joint` | fixed | `base_link`→`laser_link` | (0.051343, -1.2576e-05, 0.094524) | (0, 0, **3.1416**) | 物理激光安装，rpy=π |
| `camera_joint` | fixed | `base_link`→`camera_link` | (0.113650, 0.00016542, 0.076204) | 0 | 相机 |
| `left_joint` | revolute | `base_link`→`left_link` | (0.073593, 0.051987, 0.022) | 0 | 左转向节，axis (0,0,-1) |
| `lf_point` | continuous | `left_link`→`lf_link` | (0, 0.030126, -0.011) | 0 | 左前轮 |
| `right_joint` | revolute | `base_link`→`right_link` | (0.073593, -0.0520126, 0.022) | 0 | 右转向节，axis (0,0,1) |
| `rf_point` | continuous | `right_link`→`rf_link` | (0, -0.030126, -0.011) | 0 | 右前轮 |

- 链接：`base_link, lb_link, rb_link, front_link, controller_link, laser_link, camera_link, left_link, lf_link, right_link, rf_link`。
- 由关节推算：前轮中心 x≈0.073593、后轮中心 x≈-0.069657 → **轴距 0.14325 m**（固件 0.144）；轮中心 y 后 0.1595 / 前 0.1642 → **轮距约 0.162 m**；轮中心 z 后 0.0352 / 前 0.030（含 base_link z=0.019）→ **轮半径约 0.030~0.035 m**。
- `robot_model.yaml` 的 mini_akm 外参：`base_to_link=[0,0,0.019,0,0,0]`、`base_to_gyro=[0,0,0,0,0,0]`、`base_to_laser=[0.05134,0,0.09452,0,0,0]`、`base_to_camera=[0.11365,0.00017,0.0762,0,0,0]`、`base_to_radar=[0,0,0.3,0,0,0]`。
- **易错点**：`base_to_laser` 是 `base_footprint→laser` 的**逻辑扫描帧**（rpy=0）；URDF 的 `laser_link` 是**物理安装帧**（rpy=π）。驱动已经把角度整体减 π，所以逻辑帧用 rpy=0；不要再给 `laser` 加一次 π。
- **z 偏差（2D 可忽略）**：`robot_model.yaml` 的 `base_to_laser.z=0.09452` 是直接挂 `base_footprint`；URDF `laser_joint.z=0.094524` 挂 `base_link`，而 `base_link` 比 `base_footprint` 高 0.019 m。两者相差 0.019 m；2D Cartographer 忽略 z，若做 3D 点云/相机投影需把逻辑帧 z 改为约 0.1135（或统一挂到 base_link）。

## 3. 雷达 LD14

- 型号：**LD14（LD14P）**，360°；udev 规则 `/etc/udev/rules.d/ld14_lidar.rules`，serial `5A6D013855`，别名 `/dev/LD14`。
- 驱动：独立工作区 `/root/ld14_ws`（已构建），`start_node.sh` → `ros2 launch ld14 ld14.launch.py`；发布 `/scan`，frame **`laser`**。
- 参数：`range_min=0.15`、`range_max=30.0`（驱动代码）；物理有效量程按 AGENTS 基准 **0.15~8.0 m**，Cartographer 应按 8.0 m 截断。
- 角度：驱动 `angle_min=raw-π`、`angle_max=raw-π`；URDF `laser_joint` 的物理安装 rpy=(0,0,π)，而 `robot_model.yaml` 的逻辑 `base_to_laser` rpy=(0,0,0)。**首次运行必须用 RViz/实物确认 `/scan` 的 0° 指车头**，否则建图与避障会反向。
- 服务：`ld14-lidar.service` 由 udev 在雷达插入时触发（无 `[Install]`，不随机启动）；当前雷达未插，服务 `inactive`，`/dev/LD14` 不存在。**插上雷达 USB 后服务会自动启动并开始发布 `/scan`，它不控制底盘。**
- 不要使用厂商 `wheeltec_lidar.launch.py`：它按 `lidar_type: ls_M10P_uart` 去启动 `lslidar_driver` 并找 `/dev/wheeltec_lidar`，与本车 LD14 不符。

### 3.1 LD14 服务与环境（真机解析）

- 服务文件：`/etc/systemd/system/ld14-lidar.service`（root:root, 0600）
  - `[Unit] Description=LD14 lidar ROS 2 node (/scan)`，`After=network.target`
  - `[Service] Type=simple`；`Environment=HOME=/root / ROS_HOME=/root/.ros / ROS_LOG_DIR=/root/.ros/log`
  - `ExecStart=/root/ld14_ws/start_node.sh`
  - `Restart=on-failure`，`RestartSec=3`
  - `StandardOutput/StandardError=append:/root/ld14_ws/lidar.log`
  - **没有 `[Install]`**：不随机启动，只由 udev 在雷达 USB 接入/移除时 `systemctl --no-block restart/stop ld14-lidar.service`。
- `/root/ld14_ws`：已构建的独立工作区（`src/build/install/env.sh`），`start_node.sh` 指向 `lidar_autorestart/start_node.sh`；脚本内容为 `source /root/ld14_ws/env.sh` + `exec ros2 launch ld14 ld14.launch.py`；`env.sh` 只 source `/opt/ros/humble/setup.bash` 并追加 `ld14` 的 `AMENT_PREFIX_PATH`/`LD_LIBRARY_PATH`。
- udev 规则：`/etc/udev/rules.d/ld14_lidar.rules`，匹配 CH343 `1a86:55d4`、serial `5A6D013855`，创建 `/dev/LD14`（0777, dialout），并触发上述服务。
- 驱动源码（`/root/ld14_ws/src/ld14/src/main.cpp`）：
  - `scan.header.frame_id = "laser"`；
  - `scan.range_min = 0.15`、`scan.range_max = 30.0`；
  - `scan.angle_min = raw_angle_min - π`、`scan.angle_max = raw_angle_max - π`；
  - `scan.angle_increment = (angle_max-angle_min)/data.len`；
  - `scan.scan_time` 用相邻两帧时间戳实测、`time_increment` 按点数均分；
  - `slbf.h` 中 `SCAN_FRE = 2300`（不是直接的 Hz 值）。
- 当前状态（2026-10-21）：`systemctl is-active ld14-lidar` = **inactive**；`/dev/LD14` 不存在；`/dev/ttyACM*` 只有底盘 `ttyACM1`；`/root/ld14_ws/lidar.log` 里能看到历史启动记录与 `FOUND LiDAR_LD14`（2026-10-21 17:51:14），说明驱动本身能识别该雷达，但**雷达当前未插/未上电**。

## 4. IMU 与里程计

- IMU：底盘 STM32 板载（`imu_mode: stm32`），驱动发布 `/imu/data_raw`，frame `gyro_link`，与底盘回传同频 **约 20 Hz**；姿态四元数由底盘 `Quaternion_Solution` 计算。
- `gyro_link` 与 `base_footprint` 共位（`base_to_gyro=[0,0,0,0,0,0]`）；`base_link` 在 `base_footprint` 上方 0.019 m。Cartographer 的 `tracking_frame` 应取 **`gyro_link`**（与 IMU 帧完全同帧，满足共位 CHECK）。
- 里程计：驱动发布 `/odom`（header.frame_id=`odom_combined`，child_frame_id=`base_footprint`），20 Hz；`Publish_Odom()` **不发 TF**。厂商 EKF（`robot_localization`）配置为 `publish_tf: true`、`odom_combined→base_footprint`，但本机 **未安装 `robot_localization`**。
- 因此两种接线：
  1. 安装 `ros-humble-robot-localization`，用厂商 `wheeltec_ekf.launch.py`（融合 `/odom` + `/imu/data_raw`，10 Hz）；
  2. 不装 EKF，用本仓库 `bringup/odom_tf_broadcaster` 把 `/odom` 位姿广播为 `odom_combined→base_footprint`（简单、少依赖；Cartographer 自带 IMU + 激光融合）。
- `/imu/data_raw` 仅 20 Hz，低于 AGENTS 基准的 100 Hz；Cartographer 仍可用，但高动态转向时姿态约束会变弱，需实跑评估。

## 5. 话题、TF 与建议接线

### 5.1 话题（来自驱动源码/配置，运行期需复核）

| 话题 | 类型 | frame | 说明 |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | `laser` | LD14 驱动；360°；0.15~30 m（代码），有效 8 m |
| `/odom` | `nav_msgs/Odometry` | `odom_combined`→`base_footprint` | 20 Hz；驱动只发话题，不发 TF |
| `/imu/data_raw` | `sensor_msgs/Imu` | `gyro_link` | 20 Hz；板载 STM32 |
| `/cmd_vel` | `geometry_msgs/Twist` | — | 驱动订阅；阿克曼固件 |
| `/PowerVoltage` | `std_msgs/Float32` | — | 12.00 V（实测） |
| `/ultrasonic_data_A..F` | `sensor_msgs/Range` | — | 驱动预留；本车 `ranger_avoid_flag=false` |
| `/robot_charging_flag` 等 | 回充相关 | — | 未使用 |

### 5.1.1 驱动发布/订阅/服务全表（`wheeltec_robot.cpp` 解析）

| 方向 | 名称 | 类型 | 说明 |
|---|---|---|---|
| Pub | `odom` | `nav_msgs/Odometry` | 20 Hz；frame `odom_combined`、child `base_footprint`；pose 仅 yaw，twist 为底盘反馈 |
| Pub | `imu/data_raw` | `sensor_msgs/Imu` | 20 Hz；frame `gyro_link`；姿态四元数由底盘解算 |
| Pub | `PowerVoltage` | `std_msgs/Float32` | 每 11 帧发一次（`Count_Voltage_Pub>10`） |
| Pub | `robot_charging_flag` | `std_msgs/Bool` | 回充状态 |
| Pub | `robot_charging_current` | `std_msgs/Float32` | 回充电流 |
| Pub | `robot_red_flag` | `std_msgs/UInt8` | 红外对接状态 |
| Pub | `robot_charging_mode` | `std_msgs/Bool` | 回充模式 |
| Pub | `ultrasonic_data_A..F` | `sensor_msgs/Range` | 6 路超声波；本车 `ranger_avoid_flag=false`，未使用 |
| Sub | `cmd_vel` 或 `cmd_vel_avoid` | `geometry_msgs/Twist` | 由 `ultrasonic_avoid` 参数决定；默认 0 → **`/cmd_vel`** |
| Sub | `robot_recharge_flag` | `std_msgs/Int8` | 回充请求 |
| Sub | `set_rgb_color` | `std_msgs/UInt8MultiArray` | RGB 灯 |
| Service | `/set_charge` | `turtlesim/srv/Spawn` | 回充位姿设置；**因此构建 `turn_on_wheeltec_robot` 需要 `turtlesim`** |

> 以上名称来自源码/launch 解析，**尚未在运行态逐一确认**（本轮未启动底盘节点）。首次实跑时用 `ros2 topic list -t` / `ros2 node info` 复核。

### 5.2 TF 链（建议）

```
map → odom_combined            (Cartographer，published_frame="odom_combined")
odom_combined → base_footprint (EKF 或 bringup/odom_tf_broadcaster；驱动不发布)
base_footprint → laser         (静态：xyz≈(0.051343,-1.2576e-05,0.094524)，rpy=0)
base_footprint → gyro_link     (静态：恒等)
base_footprint → base_link     (可选：z=0.019；只有用 base_link 时才需要)
```

- 本栈建议把 `robot_frame` 直接设为 **`base_footprint`**，省掉 `base_footprint→base_link`，避免多一层 TF 和双权威。
- `laser` 是**逻辑扫描帧**（驱动已做 -π 角度偏移，rpy=0）；URDF 的 `laser_link` 是物理安装帧（rpy=π）。不要同时对 `laser_link` 和 `laser` 各加一次 180°。
- 厂商 `robot_mode_description.launch.py` 会发 `base_footprint→laser/camera_link/base_link/gyro_link/radar`；若使用本仓库的 `real_bringup.launch.py` 静态 TF，必须把对应的 `publish_*_tf` 关掉，避免双权威。

### 5.3 Cartographer 真机配置（待创建文件，未运行）

真机 `cartographer` 配置必须使用厂商帧，而不是仿真里的 `imu_link`/`odom`：

```lua
tracking_frame = "gyro_link"
published_frame = "odom_combined"
odom_frame = "odom_combined"
provide_odom_frame = false
publish_frame_projected_to_2d = true
use_odometry = false          -- 先关；里程计实测可信后再开
use_imu_data = true
TRAJECTORY_BUILDER_2D.min_range = 0.15
TRAJECTORY_BUILDER_2D.max_range = 8.0
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 8.0
```

> 注意：Foxy 与 Humble 都支持上述键；Humble 的 occupancy grid 可执行名是 `cartographer_occupancy_grid_node`，Foxy 是 `occupancy_grid_node`。真机 `use_sim_time=false`。

### 5.4 启动件选择（后续执行时）

- **底盘驱动**：只用 `turn_on_wheeltec_robot/launch/base_serial.launch.py`（启动 `wheeltec_robot_node`），参数 `usart_port_name=/dev/wheeltec_controller`、`serial_baud_rate=115200`、`odom_frame_id=odom_combined`、`robot_frame_id=base_footprint`、`gyro_frame_id=gyro_link`、`car_mode=mini_akm`、`akm_cmd_vel=none`。
- **不要直接跑** `turn_on_wheeltec_robot.launch.py`：它会连带启动 `wheeltec_ekf`（缺 `robot_localization`）和 `joint_state_publisher`（缺包）。
- **雷达**：用 `ld14-lidar.service`（`/root/ld14_ws`），不要用 `wheeltec_lidar.launch.py`。
- **TF/里程计**：用本仓库 `bringup/odom_tf_broadcaster`（`odom_combined→base_footprint`）+ `real_bringup.launch.py` 的静态 `base_footprint→laser/gyro_link`；或安装 EKF 后由 EKF 发 TF。
- **探索/控制**：`frontier_explorer`/`path_follower` 的 `robot_frame=base_footprint`、`global_frame=map`、`scan_topic=scan`、`cmd_vel_topic=cmd_vel`、`use_sim_time=false`；`max_linear_vel=0.15`、`max_angular_vel≈0.4`、`inflation_radius_cells=5`、`obstacle_safe_distance=5`、`goal_tolerance=0.25`。

## 6. 厂商配置的坑（务必保留）

1. **`car_mode` 默认 `mini_mec` 与实车不符**：实车为阿克曼 mini_akm。已把 `~/wheeltec_ws/src/turn_on_wheeltec_robot/config/wheeltec_param.yaml` 的 `car_mode` 改为 `mini_akm`（只改配置，未运行）。若以后重解压厂商 zip，需重新改。
2. **`lidar_type: ls_M10P_uart` 与实车 LD14 不符**：厂商 `wheeltec_lidar.launch.py` 会找 `/dev/wheeltec_lidar` 并启动 `lslidar_driver`，本车没有该设备；实际用 `/root/ld14_ws` 的 LD14 服务。
3. **全量 launch 依赖缺失**：`turn_on_wheeltec_robot.launch.py` 需要 `joint_state_publisher`、`robot_localization`；本机 `ros2 pkg list` 均无，apt 安装 `ros-humble-robot-localization` 时遇到 404（索引过期），需先 `sudo apt-get update`。本栈可绕开这两个依赖。
4. **驱动不发 TF**：`Publish_Odom()` 只发 `/odom` 话题；不要指望 `odom_combined→base_footprint` 由驱动给出。
5. **驱动无 `/cmd_vel` 超时**：`Cmd_Vel_Callback` 收到才发帧；固件丢帧保护又被注释，停发即保持最后速度。后续必须补看门狗/零速兜底。
6. **IMU 只有 20 Hz**：低于常见 100 Hz 基线；Cartographer 可用但需评估。
7. **激光安装角**：URDF 物理 `laser_link` rpy=π，逻辑 `laser` 帧 rpy=0，驱动做 -π 偏移；首次运行必须确认 0° 指向车头。

### 6.1 厂商 YAML / 驱动参数原始值（勘察时）

| 来源 | 键 | 勘察值 | 备注 |
|---|---|---|---|
| `wheeltec_param.yaml` | `car_mode` | `mini_mec` → **已改 `mini_akm`** | 实车阿克曼；重解压 zip 后需重改 |
| `wheeltec_param.yaml` | `ranger_avoid_flag` | `false` | 超声波避障未启用 |
| `wheeltec_param.yaml` | `lidar_type` | `ls_M10P_uart` | **与实车 LD14 不符** |
| `wheeltec_param.yaml` | `x10.lidar_port` | `/dev/wheeltec_lidar` | 本机不存在此别名 |
| `wheeltec_param.yaml` | `x10.angle_disable_min/max` | `[9000]` / `[24000]` | 仅 LS 系列使用 |
| `wheeltec_param.yaml` | `imu_mode` | `stm32` | 板载 IMU |
| `wheeltec_param.yaml` | `camera_mode` | `astra` | 未使用 |
| `base_serial.launch.py` | `usart_port_name` | `/dev/wheeltec_controller` | 本轮已建别名 |
| `base_serial.launch.py` | `serial_baud_rate` | `115200` | 驱动声明默认 `114200`，launch 覆盖 |
| `base_serial.launch.py` | `robot_frame_id` | `base_footprint` | |
| `base_serial.launch.py` | `odom_frame_id` | `odom_combined` | |
| `base_serial.launch.py` | `gyro_frame_id` | `gyro_link`（驱动默认） | |
| `base_serial.launch.py` | `cmd_vel` / `akm_cmd_vel` | `cmd_vel` / `none` | 订阅 Twist；不走 Ackermann 消息 |
| `base_serial.launch.py` | `odom_x/y/z_scale` | `1.0` | |
| `wheeltec_robot.cpp` | 默认 `car_mode` | `mini_mec` | launch 用 YAML 覆盖；现已 YAML=mini_akm |
| `wheeltec_robot.cpp` | 默认 `usart_port_name` | `/dev/wheeltec_controller` | |
| `wheeltec_robot.cpp` | 默认 `serial_baud_rate` | `114200` | **launch 必须显式传 115200** |
| `robot_model.yaml` | mini_akm 外参 | 见 2.1 节 | |
| `ekf.yaml` | `publish_tf` / frames | `true` / `odom_combined→base_footprint` | 依赖 `robot_localization`（未安装） |

## 7. 本轮对真车做的改动（均不涉及运动）

- `~/wheeltec_ws/src/turn_on_wheeltec_robot/config/wheeltec_param.yaml`：`car_mode: mini_mec` → `car_mode: mini_akm`。
- `/etc/udev/rules.d/99-wheeltec-controller.rules`（新增）：把 CH343 serial `5B0B028473` 映射为 `/dev/wheeltec_controller`（与已有 `/dev/wheeltec_base` 同指 `ttyACM1`）。
- 只读运行 `wheeltec_tools/wheeltec_keyboard.py --port /dev/wheeltec_base --monitor`（不发字节），确认 20.2 Hz / 12.00 V / 电机使能 / acc_z≈9.75。
- **未**启动 `wheeltec_robot_node`、**未**发布 `/cmd_vel`、**未**启动 Cartographer/探索/控制、**未**启动 LD14 服务。
- 尝试 `apt-get install ros-humble-robot-localization ros-humble-joint-state-publisher ros-humble-turtlesim`；因 `robot-localization` 404 中止，`dpkg -l` 确认三者均未安装。

## 8. 仍需实测/确认（禁止在未确认前实跑）

- [ ] 雷达实际 `ros2 topic hz /scan`、有效量程、静态噪声、0° 方向（插雷达后，仅雷达服务，不动底盘）。
- [ ] IMU 静止零偏、yaw 漂移、20 Hz 下的姿态可用性。
- [ ] 里程计 5 m 直线误差、360° 转角误差、`/odom` 与 TF 同步性（手推/架空轮子）。
- [ ] 车体长×宽、轮径、实际最大转角（卷尺/打满）。
- [ ] `use_odometry` 是否可开（里程计误差 <5% 且与 TF 同步）。
- [ ] `max_linear_vel=0.15`、`max_angular_vel=0.4`、脱困参数在真机上的表现（架空轮子→低速场地）。
- [ ] 丢帧看门狗/零速兜底的实际实现与验证。

> 首次实跑顺序（仅记录，未执行）：架空轮子 → 手发 0.05 m/s×1 s 零速 → 低速 0.15 m/s → 手推建图 → 自动探索。任何一步异常立即断开电机使能。

---

## 9. Foxy / Humble 双版本要点（真机 Humble / 本机 Foxy）

| 项 | 本机 Foxy | 真机 Humble | 备注 |
|---|---|---|---|
| occupancy grid 可执行 | `occupancy_grid_node` | `cartographer_occupancy_grid_node` | `real_bringup.launch.py` 用 `occupancy_grid_executable` 参数覆盖 |
| `use_sim_time` | 仿真链路 True | **False**（无 `/clock`） | 真机 launch 全部节点 False |
| 车型/帧 | 仿真 `base_link`/`imu_link`/`odom` | `base_footprint`/`gyro_link`/`odom_combined` | 真机 Cartographer 配置必须用后者 |
| 雷达 | 仿真 720 线 0.06~12 m @10 Hz | LD14 360° 0.15~8 m（有效）@待测 | 真机不要用 `wheeltec_lidar.launch.py` |
| IMU | Gazebo 真值 100 Hz | STM32 20 Hz | 真机姿态约束更弱 |
| 底盘 | `gazebo_ros_ackermann_drive` | WHEELTEC STM32 串口固件 | 真机固件丢帧保护关闭，需看门狗 |
| 控制 | 仿真 `cmd_vel` Twist | 真机 `cmd_vel` Twist（阿克曼固件） | 语义一致；真机限速 0.15 |

---

## 10. 勘察数据来源、精度与未部署清单

### 10.1 数据来源分级

- **实测（本轮）**：
  - `wheeltec_keyboard.py --port /dev/wheeltec_base --monitor`：20.2 Hz、12.00 V、电机使能、acc_z≈9.75；**0 字节下发**。
  - `ls -l /dev/wheeltec_base /dev/wheeltec_controller`：别名指向 `ttyACM1`。
  - `lsusb`：只有 CH343 `1a86:55d4`（serial `5B0B028473`），没有第二个 USB 串口（LD14 不在线）。
  - `systemctl is-active ld14-lidar`：inactive；`/dev/LD14` 不存在。
  - `dpkg -l | grep ...`：`robot_localization` / `joint_state_publisher` / `turtlesim` 均未安装。
  - SSH/网络：见 1.1 节。
- **源码/URDF/launch 解析**：
  - `mini_akm_robot.urdf` 关节 origin（2.1 节）；
  - `robot_model.yaml` mini_akm 外参；
  - `wheeltec_param.yaml` / `base_serial.launch.py` / `wheeltec_robot.cpp`（6.1、5.1.1 节）；
  - `/root/ld14_ws/src/ld14/src/main.cpp`、`slbf.h`（3.1 节）；
  - `/etc/systemd/system/ld14-lidar.service`、`/etc/udev/rules.d/ld14_lidar.rules`。
- **推断值（需实测复核）**：
  - 轮半径 ≈0.032 m（由 URDF 轮中心高度推算）；
  - 轴距 0.14325 / 轮距 0.1595~0.1642（由 URDF 关节推算，与固件 0.144/0.162 基本一致）；
  - 最小转弯半径 0.35 m、最大转角 28°（厂商脚本/固件参数，未打满实测）。
- **完全未测（运行前必须补）**：
  - `/scan` 实际频率、有效量程、静态噪声、0° 方向；
  - IMU 零偏与 yaw 漂移；
  - `/odom` 直线/转角误差、与 TF 同步性；
  - 车体长×宽、实际最大转角、实际轮径；
  - `use_odometry` 是否可开；
  - 丢帧看门狗实际行为。

### 10.2 已部署/未部署

- **已存在（真机原厂/历史配置）**：
  - `/opt/ros/humble`、`cartographer_ros`、`nav2_*`、`robot_state_publisher`；
  - `~/RDK X5_wheeltec_ros2_src_20260206.zip`（完整厂商源码，871 MB，未全量解压）；
  - `/root/ld14_ws`（LD14 驱动，已构建）；
  - `/dev/wheeltec_base`（历史 udev 别名）、`/etc/udev/rules.d/ld14_lidar.rules`；
  - `~/wheeltec_tools/`（键盘/监视脚本、SSH/底盘说明）。
- **本轮新建（仅源码/配置）**：
  - `~/wheeltec_ws/src/` 下已选择性解压：`turn_on_wheeltec_robot`、`wheeltec_robot_urdf`、`wheeltec_robot_msg`、`interfaces`、`depend`、`wheeltec_lidar_ros2`、`wheeltec_imu`、`wheeltec_robot_slam/wheeltec_cartographer`、`wheeltec_robot_keyboard`、`wheeltec_path_follow`；
  - `/dev/wheeltec_controller` udev 别名；
  - `wheeltec_param.yaml` 的 `car_mode: mini_akm`。
- **尚未部署/构建**：
  - `~/wheeltec_ws` **尚未 colcon build**；
  - 本仓库 `ros2_ws` 的 `bringup/nav_slam/regional_explorer/path_follower` **尚未同步到真机**；
  - `regional_legacy` **尚未同步到真机**（`algorithm_root` 尚无真机路径）；
  - `robot_localization` / `joint_state_publisher` / `turtlesim` 未安装；
  - 任何底盘驱动/Cartographer/探索/控制节点均未启动。

### 10.3 安全声明

本轮严格遵守「只勘察、不运行」：未启动 `wheeltec_robot_node`，未发布 `/cmd_vel`，未启动 Cartographer/探索/控制，未启动 LD14 服务；唯一对串口的操作是 `--monitor` 只读监视（不写字节）。真车当前没有悬空支架，**后续任何首次运动测试必须先架空轮子或清场，并保证电机使能开关可随时关断**。

---

# 附十六、仿真对齐真机 + 建图质量攻坚（2026-10-21）

> 注：文件中已存在「附十五 真机 RDK-X5 勘察结果」，本节按顺序接为其后的附十六。

## 0. 结论速览
- ✅ **完整建图闭环达成，且是自然完成**：`探索完成（未找到可达前沿，地图已探索完毕）`，sim **2168.779 s**，**墙钟 1087.46 s = 18.12 分钟**（RTF 1.9966）。
- ✅ **建图质量**：房间内覆盖 **98.25%**（未知仅 1.75%）；**最终漂移 0.0377 m / 0.0976°**；`>0.5m` 单秒突跳 **0 次**；东侧 x>8 与西侧 x<-12 **均 100% 已知**；四柱偏移 **0.087 / 0.034 / 0.167 / 0.180 m**（主分量均单一，无成型重影）。
- ✅ **仿真参数已对齐真机实测**（AGENTS 附十四回填表），所以本轮的结论可以往真机搬。
- 七轮迭代对比见第 7 节：覆盖度 68.83% → **98.25%**，最终漂移 17.13 m → **0.0377 m**。

## 1. 真机参数对齐（仿真 = 真机）
| 项 | 原仿真值 | 对齐后（真机实测） | 依据 |
|---|---|---|---|
| 轴距 `L` | 0.160 | **0.144 m** | 附十四回填表 / 固件 `Akm_axlespacing` |
| 轮距 | 0.200 | **0.162 m** | 固件 `Akm_wheelspacing` |
| 轮径 | 0.064 | 0.064 m（不变） | 卷尺复核 |
| 前轮最大转角 `max_steer` | 0.6 rad | **0.489 rad（28°）** | `Mini_akm_max_angle` |
| 底盘最高速 | 3 m/s | **0.5 m/s** | 固件/键盘脚本上限（控制端仍限 0.3） |
| 雷达 | 10 Hz / 0.06~12.0 m | **6 Hz / 0.15~8.0 m** | LD14 物理有效量程 |
| IMU | 100 Hz | **20 Hz** | 底盘板载 STM32 |
| `cartographer.lua` 量程 | min 0.12 / max 12 / missing 5 | **0.15 / 8 / 4** | 同上 |
**派生**：最小转弯半径 `R_min = L/tan(max_steer) = 0.2706 m`（真机固件记 0.350，含打角死区/轮胎侧偏裕度）。
**注意**：转向关节 `limit` 也从 ±0.6 同步改为 ±0.489，否则与 `max_steer` 不自洽。

## 2. 底盘 PID 开环标定（**推翻了历史误判**）
方法：只起仿真、20 Hz 直发 `/cmd_vel`、与 Gazebo 真值对比；每个 P 值单独重启（PID 是 URDF 编译期参数）。

**线速度环**（转向固定 0.1）：
| P | NaN | 0.30 m/s 实测比值 | 起动到 90% | 可用 |
|---|---|---|---|---|
| 0.01（历史基线） | 0 | 0.953 | **2.115 s** | 可用但瞬态极慢 |
| 0.1 | 0 | 0.999 | 0.205 s | 可用 |
| 0.2 | 0 | 0.999 | 0.115 s | 可用 |
| **0.3（采用）** | 0 | **1.000** | **0.095 s** | **可用（上限）** |
| 0.4 | **812** | — | 制动段数值爆炸 + roll 3.02 rad | **不可用** |
| 0.5 | **1519** | — | 翻滚、残余位移 4.88 m | **不可用** |

**关键纠正**：历史上"实际速度只有指令的 45~57%"**不是稳态跟不上**（P=0.01 稳态就有 0.95），而是**加速瞬态太慢（2.18 s）**。失效点是**制动段**而非匀速——P 过高时制动会让内部 PID 爆掉。

**转向环**（线速度固定 0.3）：
| P | yaw 比值(0.35) | 阶跃过零 | roll | 备注 |
|---|---|---|---|---|
| 0.1（历史基线） | 0.205 | 0.305 s | 0 | 慢、拖尾 |
| 0.5 | 0.420 | 0.195 s | 0 | 单次大过冲 |
| **2.0（采用）** | **0.515** | **0.055 s** | 0 | 最快且不振荡 |
| 5 / 10 / 50 | 0.546 / 0.563 / 0.584 | 0.075 / 0.085 / 0.145 s | 0 | 振铃，不推荐 |
**结构性限制**：P-only + D=0 存在稳态欠转，即使 P=50 也只到 0.58 —— 这是能力上限，**不是 bug**。所有转向候选都没有 NaN/翻滚。
**采用值**：`linear=0.3 0 0`、`left/right steering=2.0 0 0`，**D 必须为 0**。

## 3. 曲率钳制（治"满舵原地绕圈"）
**症状**：对齐真机几何后一次跑出 **91.3% 采样 `v/|w| < 0.271 m`（曲率物理不可达）**、`|w|≥0.95` 占 91.1% → **累计路程 650 m、净转 133 圈，只换来 5.7 m 净位移**，覆盖仅 68.83%、超时。
**根因**：`path_follower` 的 `w = clamp(k*err, ±max_angular_vel=1.0)` 没考虑"这辆车能转多急"。插件按 `steer = atan(L*w/v)` 反解转角，`|w| > |v|·tan(max_steer)/L` 时转角被 `max_steer` 截断 → 实际曲率到不了指令曲率 → 横向误差不收敛 → 满舵绕圈。
**修法**：发布前钳制 `|w| ≤ |v| · tan(max_steer)/L`（`kappa_max=3.695 1/m`），参数 `wheelbase_m=0.144`、`max_steer_rad=0.489`、`curvature_guard_enable=True`；`max_angular_vel` 1.0 → **0.6**；前视改**自适应** `clamp(0.6×到目标距离, 0.4, 1.0)` 且不超过到路径末点距离。
**效果**：不可达曲率 **0/18324（0%）**、`|w|≥0.95` **0%**、净转 133 → **16.35 圈**、路程/净位移 113 → **44.1**。

## 4. 接入轮式里程计（先验证再启用）
- 先开环实测仿真 `/odom`：直线误差 **0.0024%**、转角误差 **0.0157°**、20 s 混合动作累计误差 **0.0000219 m**、位姿不冻结（历史"冻结"是旧 PID=0.01 配置下的现象，当前**不复现**）。
  - ⚠️ 注意：仿真的 `/odom` **取自 Gazebo 世界位姿**（插件里有 `UpdateOdometryWorld()`），所以它代表"完美里程计"；真机轮式里程计仍需轮径/轮距标定 + EKF。
- 接线：URDF 插件 `publish_odom_tf=true`、`robot_base_frame=base_footprint`（由插件动态发 `odom→base_footprint`）；**删掉** `gazebo.launch.py` 里那条"静态恒等 `odom→base_footprint`"（否则同一条 TF 两个发布者）；`cartographer.lua` `use_odometry=true`；`gmapping.launch.py` remap `('odom','/odom')`。
- 验证：`/tf` 发布者只有 ackermann 插件 / rsp / cartographer 三个；`odom→base_footprint` 同帧重复 pair = 0；cartographer 订阅含 `/odom`。

## 5. 回环约束收紧（治"米级平移突跳"——**本轮最关键的一步**）
**症状**：启用里程计后仍在 sim 1204→1208 从 0.063 m **突跳到 1.629 m**（yaw 只差 1.43°）、1306 到 5.901 m、**1458 又"回正"到 0.96 m**、最终固定 7.163 m / 7.68°。同期 `/odom` 对真值 RMSE 仅 0.0076 m。
**根因链**：局部匹配的 `real_time_correlative_scan_matcher.linear_search_window` 只有 **0.1 m**，不可能一次跳 1.6 m → 米级突变只能来自**位姿图回环约束**；而 `constraint_builder.fast_correlative_scan_matcher.linear_search_window` 默认 **±7 m**、`min_score` 仅 0.55 → 在 28×20 长直墙对称房间里，"沿墙平移 1.6~7 m"也能匹配成功 → 错误约束把整段轨迹拉歪，之后又被别的约束拉回（来回跳）。
**修法**（`cartographer.lua`）：
| 键 | 原值 | 新值 |
|---|---|---|
| `constraint_builder.fast_correlative_scan_matcher.linear_search_window` | 7. | **2.** |
| `constraint_builder.fast_correlative_scan_matcher.angular_search_window` | rad(30) | **rad(15)** |
| `constraint_builder.min_score` | 0.55 | **0.75** |
| `constraint_builder.global_localization_min_score` | 0.6 | **0.8** |
| `constraint_builder.max_constraint_distance` | 15. | **8.** |
| `optimization_problem.odometry_translation_weight` | 1e5 | **1e6** |
| `optimization_problem.odometry_rotation_weight` | 1e5 | **1e6** |
| `TRAJECTORY_BUILDER_2D.submaps.num_range_data` | 90 | **180** |
**效果**：最终漂移 **7.163 m → 0.063 m**；`>0.5m` 单秒突跳 **0 次**（最大 0.0517 m）；房间外占据格 **15 → 0**；柱子偏移 1.11/1.06 m → 0.16/0.13 m；覆盖 84% → **97.59%**。

## 6. 目标计数口径（治"覆盖未收敛就提前 done"）
**症状**：上一轮覆盖已 97.59% 却以 `达到最大目标数 200` 结束。200 个"不同目标"里只有 45 次真到达，其余是前沿随地图微调而 0.1~0.5 m 的小幅挪动。
**修法**：新增 `goal_count_min_move_m=1.0` —— 只有新目标相对**上一个已发布目标**位移 ≥1.0 m 才计入 `max_goal_count`（`max_goal_count` 仍 200）。
**效果**：同一场里 164 个不同坐标 → **61 个计入**（106 次被过滤 = 63.5%），预算只用 30.5%，于是**走到自然完成**。计数增长率从 0.1031 次/仿真秒降到 0.0281（慢 3.7 倍）。

## 7. 七轮迭代对比
| 指标 | room6 | demo(RTF3) | align(无钳制) | curve(钳制) | odom(接里程计) | loop(收紧回环) | **final2** |
|---|---|---|---|---|---|---|---|
| 覆盖率 | 96.64% | 93.07% | 68.83% | 93.82% | 84.12% | 97.59% | **98.25%** |
| 结束方式 | 自然 | max_goals | timeout | 自然 | max_goals | max_goals | **自然 `_no_frontier`** |
| 最终漂移 | 0.35 m | 0.6~0.9 m | 2.106 m | 17.13 m | 7.163 m | 0.063 m | **0.0377 m / 0.098°** |
| 房间外占据格 | 0 | 1043 | — | 6300（53.3%） | 15 | 0 | **134（墙缘拖尾）** |
| 墙钟耗时 | 41.6 min | 32.8 min | 50.9 min | 32.5 min | 15.9 min | 16.23 min | **18.12 min** |

## 8. 本轮关键踩坑（务必保留）
1. **PID 的"P 越小越慢"不是稳态问题**：P=0.01 稳态比值 0.95，但起动要 2.18 s；**失效点是制动段**（P≥0.4 会让内部 PID 爆 NaN + 翻滚）。可用上限 **0.3**。
2. **转向 P-only + D=0 有结构性欠转**（最好也只到 0.58 比值）——调参到此为止，别指望调到 1.0。
3. **`max_steer` 改小会连带改变"可达曲率"**：`R_min` 从 0.234 → 0.2706 m，控制器若不钳制曲率就会满舵绕圈。**几何一改，必须同步检查/钳制控制器的角速度上限**。
4. **跳变量级能反推责任层**：局部匹配搜索窗 0.1 m 却出现 1.6 m 跳变 → 一定是**位姿图/回环**，不是局部匹配、不是 odom。
5. **"来回跳 + yaw 几乎不变"= 沿墙平移被弱匹配接受**：对称长直墙环境下要收紧 `linear_search_window` 与 `min_score`，并提高里程计权重。
6. **里程计能救漂移但不能救误回环**：接入 `/odom` 后漂移 17.13 → 7.16 m，再收紧回环才到 0.0377 m。
7. **RTF 3.0 会拖垮 Cartographer**：实测 RTF 2.995 能达到，但实时处理跟不上、漂移变大、反而跑更久；**2.0 是安全上限**。
8. **目标计数会被"前沿小幅挪动"吃光预算**：`<1 m` 的挪动不该算新目标，否则覆盖没收敛就以 `max_goal_count` 结束。

## 9. 复现命令
```
ros2 launch bringup exploration_bringup.launch.py \
  world:=room.world \
  map_save_prefix:=/tmp/final2_maps/room \
  max_exploration_time_sec:=3000 \
  headless:=true
```
预期：约 **18 分钟**墙钟后出现 `探索完成（未找到可达前沿，地图已探索完毕）`，生成 `<prefix>_no_frontier.pgm/.yaml`；房间内覆盖 ≈98%。
原始报告：`/tmp/final2_report.md`、`/tmp/loop_report.md`、`/tmp/odom_report.md`、`/tmp/curve_report.md`、`/tmp/align_report.md`、`/tmp/pid_report.md`；过程图 `/tmp/final2_maps/progression.png`、`progression.gif`。

## 10. 遗留与后续
- **复杂建筑 `3d.world` 仍跑不通**（窄门/窄走廊 + 控制器无局部绕障）。
- **仿真 `/odom` 是"完美里程计"**：真机必须做轮径/轮距标定 + EKF，并复核附十四第 5 节。
- 转向结构性欠转（比值 ~0.5）会让实际转向比指令慢约一半；若真机底盘转向环更好，可在真机上调回更快参数。
- `POSE_GRAPH.constraint_builder.min_score=0.75` 是**针对对称大空房间**的收紧；换到特征丰富的真实建筑里可能过严（回环被拒 → 长距离漂移），需按场景复核。

---

# 附十七、复杂建筑探索 + 阿克曼轨迹规划（2026-10-21）

## 0. 结论速览
- ✅ **复杂建筑（`3d.world`）已能稳定跑通**：无弹飞、无永久卡死、**真的穿过了 0.9 m 门洞**、门后房间 100% 已知。
- 覆盖率：任务外壳口径 **82.82%**；**真实外墙 AABB 内 94.52%**（未知仅 5.48%）。
- 定位：位置 RMSE **0.082 m** / max 0.338 m；yaw RMSE 0.976°；`>0.5m` 单秒突跳 **0 次**。
- 效率：距离场 **0.0358 s**（整图 1.4448 s，**40.4 倍**）、决策吞吐 **0.535 路径/s**、`/map`→目标发布延迟中位 **0.482 s**（99.49% <1 s）。
- ✅ **新增阿克曼轨迹规划模块** `core/ackermann_traj.py`：把 A\* 折线转成"车真能开"的轨迹，构造上保证曲率 ≤ `1/R_min`。
- ⏳ 未到"自然完成"的唯一原因是**时间预算**：70 分钟墙钟上限时车仍在行驶，末段 300 s 覆盖仍 **+1.32 pp** 增长；剩余可达但未知在真实外墙内仅 **60.41 m²**，全部物理可达。

## 1. 为什么 A\* 不够：需要一层"轨迹可行化"
A\* 是**全局规划**，回答"根据已知地图从 A 到 B 走哪条路"；它不回答"这一帧往哪打方向"。差 4 件事：
| 差距 | 具体表现 |
|---|---|
| 曲率不可行 | 折线转角处曲率无穷大；真车 `R_min = 0.35 m` |
| 折角只有 90°/45° | 4/8 邻接只有离散方向，车需要连续转向 |
| 没有速度剖面 | 弯道必须 `v ≤ sqrt(a_lat/κ)`，直线可加速 |
| 起步姿态不匹配 | 车头朝东、路径首段朝西时，阿克曼**不能原地转**（这正是历史"满舵绕圈"的根） |

**关键前提（已核实）**：上游候选前沿的 BFS 可达性只在**已知自由格**上算（`real_work/light_qifa.py::_reachable_distances`，`explored_map != 0` 直接返回全 -1）。因此"规划时把未知格当硬障碍"**不会让任何合法前沿不可达** —— 探索本来就不需要进入未知区，frontier 就在已知边界上。

## 2. 新增模块 `regional_legacy/core/ackermann_traj.py`
纯 numpy、无 ROS 依赖、可离线单测。入口：
```python
plan_ackermann_trajectory(explored_map, grid_to_world_xy, world_to_grid_rc,
                          start_pose_world, goal_world_xy, config,
                          planner=None, distance_field=None) -> Trajectory
```
流水线：**A\*（8 邻接 + 未知硬阻挡）→ 视线捷径 → 圆弧倒角 → 等弧长重采样 → 起步对齐 → 速度剖面 → 全局复核**。

### 车辆模型约定（下游依赖它）
后轴中心参考点；`kappa = tan(delta)/L`、`|delta| ≤ max_steer`；`yaw_rate = v*kappa`（v 带符号）；**倒车要同向累积 yaw 时 `kappa` 必须与 `v` 反号**。

### 圆弧倒角（构造上保证曲率可行）
```
phi = atan2(cross(u_in,u_out), dot(u_in,u_out))
t   = R * tan(|phi|/2)
T1  = Pi - t*u_in ;  T2 = Pi + t*u_out
b   = normalize(u_out - u_in)    # 指向转角内侧
C   = Pi + (R / cos(|phi|/2)) * b
arc = 从 T1 绕 C 到 T2，扫角 phi，kappa = sign(phi)/R
```
半径序列从 `2.0*R_min` 到 `1.0*R_min` 取 **8 档**（从大往小试），失败则把折点沿角平分线向外移（0.5 格/次、最多 20 次），并允许相邻折点一起外移。

### 五条硬性质（离线 fuzz 验证）
1. 处处 `|kappa| ≤ kappa_max + 1e-6`
2. 采样点到已知障碍距离 `≥ safety_cells`（本体占位圈内豁免，见第 4 节）
3. `yaw` 逐区间连续（拼接误差 ~1e-16，`|wrap(tangent-yaw)| ≤ 0.0478 rad`）
4. `|v| ≤ min(v_max, sqrt(a_lat_max/|kappa|))`
5. **永不进入 `-1` 未知格**

### 起步对齐
`align_tol_rad = 0.0175`（1°，仅容忍数值噪声）。超过就必须生成对齐弧：先试纯前进弧（`R=R_min`），碰撞则做**三点掉头**（前进/倒车/前进，倒车段 `kappa` 与 `v` 反号）。夹角接近 ±π 时两个方向都试。

## 3. ROS2 接线
- `frontier_explorer_node`：每轮**只算一次**距离场，复用给 `decide_frontier`（新增 `distance_to_obstacle=` 关键字）、A\*、轨迹模块；发布 `nav_msgs/Path` 时**每个 pose 带切线 yaw**（`z=sin(yaw/2), w=cos(yaw/2)`）；不可行/异常/yaw 缺失 → **回退原始 A\* 折线**（warn 节流）。
- `path_follower`：前视点改**按弧长**取（不再欧氏距离）；新增**曲率限速** `v ≤ sqrt(a_lat_max/κ_local)`；曲率钳制改统一口径 `kappa_max = 1/min_turn_radius_m`。
- 新入口 `bringup/launch/complex_building_explore.launch.py`：`world=3d.world`、`inflation_radius_cells=3.0`、`treat_unknown_as_obstacle=True`、`use_eight_neighbors=True`、`max_goal_count=800`、`goal_count_min_move_m=2.0`、`exploration_bounds="-19.0,19.5,-19.0,15.5"`、`stall_detect_sec=20`、`max_exploration_time_sec=4500`。

## 4. 五轮迭代中修掉的 8 个真 bug
| # | 症状（实测） | 根因 | 修法 |
|---|---|---|---|
| 1 | 阿克曼可用率仅 24%，`A* 无路` 45 | **未知格被一起膨胀**：frontier 天然紧贴未知，3 格膨胀把目标格自己封死 | 未知格**只做硬阻挡、不参与膨胀**；已知障碍照旧膨胀 |
| 2 | 覆盖 31% 就以 `达到最大目标数 200` 结束 | 复杂建筑入口漏了 `max_goal_count`（继承 200） | 800 + `goal_count_min_move_m=2.0` |
| 3 | 上游用 `R_min=0.35`、下游按 0.271 钳制 | follower 没加 `min_turn_radius_m` | 统一 `kappa_max = 1/min_turn_radius_m` |
| 4 | 折点倒角失败 46/169 | 半径候选太粗、外移步长太大 | 8 档半径 + 0.5 格外移 + 相邻折点扩散 → 窄通道成功率 **34.4% → 71.9%** |
| 5 | `A* 无路` 涨到 **201/533** | **机器人自身格常被判障碍/膨胀区**（59 次非自由格），A\* 一开算就无路 | A\* 新增 `start_clear_cells`：清空**本体占位圈**（Chebyshev ≤ `ceil(safety)+1`），轨迹复核对圈内点豁免安全距离（**未知格仍禁止**） |
| 6 | 16% 时间耗在建筑外空地 | 前沿选到外壳外 | `exploration_bounds` 过滤 → 建筑外时间 **15.68% → 0.55%** |
| 7 | 末段 365 s 覆盖只 +0.03pp | 难目标被反复选中（黑名单 180 s 到期又复活） | `hard_target_fail_count=3`：累计 3 次永久排除 |
| 8 | 距离场 **2.43 s/轮**，吞吐掉到 0.3~0.45 轮/s | 纯 Python chamfer 算整图精确距离 | `distance_to_obstacle(max_distance=...)` 上限化 + 向量化 → **2.465 s → 0.0687 s（35.8 倍）**，且 `R=3/5` 膨胀掩膜与精确场**逐元素一致** |

### 第 9 条：**环境里的"隐形杀手"**（不是算法 bug，但最坑）
车卡在 `(-13.2, 4.6)`，该处是 `simple_arm::arm_base` 的 **1×1×0.1 m 贴地碰撞板**（`z[0,0.100]`）。雷达扫描平面在 **z=0.0945**，只比板顶低 5.5 mm → 射线几乎**擦着顶面**过去，基本打不到，**该板在 2D 栅格图上是自由格**，规划/循迹都看不见。卡死 461 s 后车被**物理弹飞**（`z 0→0.955 m`、水平峰值 **15.65 m/s**），Cartographer 位姿偏差 **76 m**、地图崩坏 → 触发**伪完成**（报 `未找到可达前沿`，但外壳覆盖仅 31.78%）。
**修法**：把 `simple_arm` 的 2 个贴地薄板（`arm_base_geom`、`arm_shoulder_pan_geom_arm_shoulder`，均 `z[0,0.1]`）**抬到 0.4 m 高**（`z[0,0.4]`），雷达平面实打实切过去。物理可通行性不变（本来就是碰撞体），只是从"隐形"变"可见"。
**效果**：弹飞消失（`z` 最大 **0.0001 m**、1 s 水平位移峰值 **0.334 m**、真值/地图最大偏差 **0.338 m**）；底座区域在地图上 **146/441 格为障碍、0 未知**；同处最长停留 **41 s** 后脱困。
**教训**：任何 2D 激光探索场景，**凡是顶面高度落在雷达平面 ±1 cm 内的薄几何，都会变成"隐形障碍"**。做场景前先跑一遍"贴地低矮碰撞体扫描"（筛 `zmin ≤ 0.03` 且 `zmax ≤ 0.16` 的 box）。

## 5. 六轮迭代对比
| 指标 | bldg1 | bldg2 | bldg3 | bldg4 | **bldg5** | final2(房间) |
|---|---|---|---|---|---|---|
| 覆盖率（任务外壳） | 31.49% | 42.49% | 74.71% | 45.47% | **82.82%**（真实外墙内 **94.52%**） | 98.25% |
| 结束方式 | max_goals 200 | timeout 3000s | timeout 3000s | **伪完成**（弹飞） | **无 done，70min 人工上限** | 自然 `_no_frontier` |
| 弹飞 | 否 | 否 | 否 | **是**（z→0.955m） | **否** | 否 |
| 阿克曼可用率 | 24.26% | 20.45% | 39.18% | 26.32~35.59% | **37.59%** | N/A |
| 位置漂移 RMSE/max | 0.045/0.150 | 0.091/1.302 | 0.069/0.310 | 0.190/0.311 | **0.082/0.338** | 0.053/0.171 |
| 门洞穿越 | 否 | 否 | **是** | — | **是** | N/A |

## 6. 剩余与后续
- **未到自然完成**：唯一原因是**时间预算**。末段 300 s 覆盖仍 +1.32 pp 增长；真实外墙内剩余可达但未知仅 **60.41 m²**（24,166 格），物理不可达 **0 格**。想跑到 `_no_frontier` 需要把 `max_exploration_time_sec` 提到 ~8000 s 或继续提速。
- **阿克曼可用率 37.59%**：剩余失败以折点倒角、起步对齐为主。要再提升，方向是"失败时直接换一个前方开阔的前沿"（已做连续失败拉黑），以及给倒角加"允许局部绕行"。
- **`3d.world` 的 `<state>` 会把 `Untitled222` 整体平移 (+3.901, +1.153)**：门洞真实位置是 `x∈[7.7557, 8.6557], y=-8.515`，不是源码里的 `x≈3.85, y≈-9.67`。查场景坐标时**必须叠加 `<state>` 偏移**。
- 真机侧仍需：`min_turn_radius_m` 用固件值 0.35、`inflation_radius_cells=3` 是否够安全需按实测车宽复核、`a_lat_max=0.6` 需按真实抓地力标定。

