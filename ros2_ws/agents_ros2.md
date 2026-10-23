# agents_ros2.md — ros2_ws 使用说明

本文件面向三类读者：
1. 想在本机跑仿真的人（第 1–4 节）；
2. 要把这套代码搬到 RDK-X5 真机（ROS2 Humble）的人（第 6 节）；
3. 未来要改「前沿选点」逻辑的人（第 7 节，**最重要**）。

> 一句话总结：`ros2_ws` 是自包含的。算法副本已随包携带
> （`src/regional_explorer/regional_explorer/legacy/`），
> **只拷这一个文件夹**即可运行（目标机需重新 `colcon build`）。

---

## 1. 目录结构

```
ros2_ws/
├── run_sim.sh                     # 一键：清进程 + Gazebo + 建图 + 探索 + 控制 + matplotlib
├── run_sim.log                    # 运行日志（每次覆盖）
├── maps/                          # 探索完成时写出的 pgm/yaml
├── build/  install/  log/         # colcon 产物（绝对路径，换机器必须重编）
└── src/
    ├── bringup/                   # 联调入口 + 可视化 + 桥接
    │   ├── launch/
    │   │   ├── exploration_demo.launch.py        # 房间场景（room.world）+ viz
    │   │   ├── exploration_bringup.launch.py     # 仿真核心：Gazebo+Cartographer+探索+控制
    │   │   ├── complex_building_explore.launch.py# 复杂封闭建筑 + viz
    │   │   └── real_bringup.launch.py            # 真机入口（Humble）
    │   └── bringup/
    │       ├── exploration_viz.py                # matplotlib 建图可视化
    │       ├── odom_tf_broadcaster.py            # 真机：/odom → TF
    │       ├── twist_to_ackermann.py             # 真机：Twist → 阿克曼
    │       └── real_diagnostics.py               # 真机自检
    ├── gazebo_modele/            # URDF（LD14 雷达 0.15~8m、IMU 20Hz）+ world
    │   └── world/{room.world, 3d.world, 3d_enclosed.world}
    ├── nav_slam/                 # Cartographer 配置 + 无回波预处理
    │   ├── config/{cartographer.lua, cartographer_real.lua, cartographer_real_noimu.lua}
    │   ├── launch/gmapping.launch.py
    │   └── nav_slam/scan_no_return_filter.py
    ├── regional_explorer/        # 前沿选点 + A* + 阿克曼轨迹
    │   ├── regional_explorer/frontier_explorer_node.py   # 节点层（选点调度/过滤/重试）
    │   └── regional_explorer/legacy/                     # 算法层（自包含副本）
    │       ├── core/{a_star.py, ackermann_traj.py, obstacle_distance.py}
    │       └── real_work/light_qifa.py                   # 启发式打分（decide_frontier）
    └── path_follower/            # 纯跟踪 + 曲率限速 + 前向刹车/脱困
```

### 数据流

```
gzserver ──/scan──> scan_no_return_filter ──/scan_for_slam──> cartographer_node
                                   │                              │
                          (避障仍用原始 /scan)                  /map
                                                                  │
                        frontier_explorer_node ──/exploration_goal──> path_follower
                              │            └────/exploration_path───> （发 /cmd_vel）
                              └──> exploration_viz（matplotlib 窗口）
```

关键约定：
- 地图分辨率固定 **0.05 m/格**，任何环节都不得缩放/重采样。
- `/scan` 保持原始（含 `inf`）；只有 Cartographer 用滤波后的 `/scan_for_slam`。
- 目标锁定开启（`target_lock_until_reached=True`）：走到目标才换下一个前沿。

---

## 2. 一次性准备

```bash
# ROS2 Foxy（本机）
sudo apt install ros-foxy-cartographer ros-foxy-cartographer-ros \
                 ros-foxy-gazebo-ros-pkgs ros-foxy-tf2-ros \
                 ros-foxy-joint-state-publisher-gui

# 编译（首次或改了 src 之后）
cd <本目录>
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
```

`--symlink-install` 让 Python 文件改完**不用重编**（launch/py 是软链）；
但改了 `setup.py` / `package.xml` 或新增文件，仍要重新 build。

---

## 3. 环境变量（每次开新终端都要）

```bash
source /opt/ros/foxy/setup.bash
source <本目录>/install/setup.bash
source /usr/share/gazebo/setup.sh          # 必须！否则 gzclient 找不到 shader 会闪退
export HOME=/tmp/roshome                   # 建议：隔离 ~/.gazebo，避免旧配置/旧锁干扰
export GAZEBO_MODEL_DATABASE_URI=""        # 断网环境不卡 models.gazebosim.org
```

`HOME=/tmp/roshome` 是可选的，但本仓库所有实测都是在这个 HOME 下跑的；
不设也能跑，只是 `~/.gazebo` 里旧配置可能影响 GUI。

---

## 4. 运行仿真

### 4.1 房间场景（默认，推荐第一次跑）

```bash
bash <本目录>/run_sim.sh
```

这条脚本做四件事：清残留进程 → 起 Gazebo + Cartographer + 探索 + 控制 →
起 matplotlib 可视化 → 把 Gazebo 窗口提到前台。`Ctrl-C` 自动清理。

### 4.2 复杂封闭建筑（3d_enclosed.world）

```bash
# 1) 清残留（端口 11345 被占会导致新 gzserver 直接退出 255）
pkill -9 -f 'ros2[ ]launch'; pkill -9 -f gzserver; pkill -9 -f gzclient
pkill -9 -f cartographer_node; pkill -9 -f frontier_explorer_node
pkill -9 -f path_follower_node; pkill -9 -f scan_no_return_filter
sleep 1

# 2) 环境
source /opt/ros/foxy/setup.bash
source <本目录>/install/setup.bash
source /usr/share/gazebo/setup.sh
export HOME=/tmp/roshome
export GAZEBO_MODEL_DATABASE_URI=""

# 3) 启动（Gazebo 3D 窗口 + matplotlib 建图窗口）
cd <本目录>
ros2 launch bringup complex_building_explore.launch.py world:=3d_enclosed.world
```

### 4.3 只跑逻辑、不弹窗口（服务器/CI 自检）

```bash
ros2 launch bringup complex_building_explore.launch.py \
  world:=3d_enclosed.world headless:=true viz:=false
```

### 4.4 常用参数

| 参数 | 作用 |
|---|---|
| `world:=room.world` / `3d.world` / `3d_enclosed.world` | 换地图 |
| `headless:=true` | 只起 gzserver，不弹 Gazebo 窗口 |
| `viz:=false` | 不弹 matplotlib |
| `software_gl:=true` | Gazebo 走软件 OpenGL（GLX 驱动异常时救急） |
| `inflation_radius_cells` | 硬膨胀半径（房间 5 格 / 复杂建筑 3 格） |
| `clearance_target_cells` | 软性期望净空（默认 10 格 = 0.5 m） |
| `max_exploration_time_sec` | 探索超时 |
| `candidate_retry_count` | 选点失败时最多试几个候选 |

> ⚠️ `headless:=true` 会**全局清空 DISPLAY**（防止 gzserver 建 GLX 崩溃），
> 所以同一 launch 里的 `exploration_viz` 也拿不到显示 → 窗口不出现、也不报错。
> 无显示环境请同时 `viz:=false`。

---

## 5. 观察与排查

```bash
# 看话题
ros2 topic list
ros2 topic echo /scan --no-arr            # 720 条，含 inf，range_max=8.0
ros2 topic echo /scan_for_slam --no-arr   # 720 条全 finite，range_max≈8.501
ros2 topic echo /exploration_goal --field pose.position
ros2 topic hz /map

# TF
ros2 run tf2_tools view_frames

# 日志关键字
grep -E '到达当前目标|清空当前目标|换下一个候选|本轮不出目标|探索完成' run_sim.log
grep -E '开始三点掉头脱困|脱困结束' run_sim.log
grep -E '\[ERROR\]|process has died' run_sim.log

# 地图
ls maps/
```

常见坑（都踩过）：

| 现象 | 原因 / 处理 |
|---|---|
| Gazebo 窗口闪一下就关 | 端口 11345 被旧 gzserver 占用；先 `pkill -9 -f gzserver` |
| gzclient 立即退出、日志有 GLX/X Error | ①在 IDE 沙箱终端跑（拿不到 GLX）→ 换系统终端；②`GAZEBO_RESOURCE_PATH` 被 ~/.bashrc 里失效路径污染 → `source /usr/share/gazebo/setup.sh` |
| matplotlib 窗口不出现 | `headless:=true` 清了 DISPLAY；或终端不能访问 X |
| 地图上近身一圈灰色未知 | 无回波没被清：检查 `scan_no_return_filter` 是否运行、`/scan_for_slam` 是否全 finite |
| 车贴墙反复倒车 | 软净空不足；调 `clearance_target_cells`（见第 7 节） |
| 车不动、done=假 | 上一轮伪完成已修；看日志是否 `本轮不出目标` 连续多轮 |

---

## 6. 迁移到真机（ROS2 Humble，RDK-X5）

### 6.1 必须改的（否则起不来/行为不同）

| 项 | Foxy（本机） | Humble（真机） | 处理方式 |
|---|---|---|---|
| 发行版路径 | `/opt/ros/foxy` | `/opt/ros/humble` | 所有 `source` 改 humble |
| apt 包名 | `ros-foxy-*` | `ros-humble-*` | 重新装 `cartographer_ros`、`gazebo_ros_pkgs`、`tf2_ros`、`joint-state-publisher-gui` |
| occupancy grid 可执行名 | `occupancy_grid_node` | `cartographer_occupancy_grid_node` | `real_bringup.launch.py` 用参数 `occupancy_grid_executable:=cartographer_occupancy_grid_node` |
| `use_sim_time` | 仿真 `True` | 真机 `False` | `real_bringup.launch.py` 已全部写死 `False`，不要套用仿真 launch |
| Python | 3.8 | 3.10 | 代码无版本特性依赖，直接可用 |
| 雷达话题/TF | `velodyne` | 厂商 `laser` / `laser_link` | `scan_topic`、`laser_frame`、`laser_x/y/z` 按实测回填 |
| 里程计 TF | Gazebo 插件发 `odom→base_footprint` | 驱动只发 `/odom` 话题 | `publish_odom_tf:=true` 起内建 `odom_tf_broadcaster` |
| 速度接口 | `/cmd_vel`（Twist） | 可能是阿克曼消息 | `use_ackermann_interface:=true`，走 `twist_to_ackermann` |

建议核对的两处（Humble 上可能只告警、不影响）：

- `static_transform_publisher` 的位置参数形式（`x y z yaw pitch roll parent child`）。
  Humble 更推荐命名参数：`--x --y --z --roll --pitch --yaw --frame-id --child-frame-id`。
  若报错，把 `real_bringup.launch.py` 里的 `_static_tf` 改成命名参数写法。
- `rclpy.qos` 的 preset 名称（`qos_profile_sensor_data`）两版都有，不用改。

### 6.2 拷贝与编译

```bash
# 只拷 ros2_ws 一个文件夹
rsync -av ros2_ws/ wzy@<真机IP>:~/ros2_ws/

# 真机上重编（install/ 里是构建时绝对路径，直接拷过来不能用）
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### 6.3 真机启动顺序（安全第一）

```bash
# 1) 先只连通底盘 + 雷达，确认数据
ros2 launch bringup real_bringup.launch.py \
  start_driver:=true driver_package:=turn_on_wheeltec_robot \
  driver_launch_file:=base_serial.launch.py \
  publish_odom_tf:=true occupancy_grid_executable:=cartographer_occupancy_grid_node

# 2) 手推建图（不给电机使能），确认 map→base_link 跟手推一致
# 3) 架空轮子发零速，确认方向/刹车
# 4) 低速自动跑（默认 max_linear_vel:=0.15），人站急停旁
```

### 6.4 建议一起改的真机参数

| 参数 | 建议值 | 原因 |
|---|---|---|
| `max_linear_vel` | 0.15 | 首次实跑保守 |
| `inflation_radius_cells` | 5 | 车宽/2 ≈ 0.10 m，5 格=0.25 m 留余量 |
| `min_turn_radius_m` | 0.35 | 固件保守值，必须与上游轨迹一致 |
| `target_blacklist_radius` | 1.5 | 同墙相邻前沿间距约 1.06 m |

---

## 7. 如何修改「前沿选点」代码（重点）

选点逻辑分**三层**。先判断你的需求属于哪一层，不要一上来就改算法。

### 7.1 第 0 层：先试参数（不改代码）

大多数需求改 launch 参数就够：

| 需求 | 参数 | 位置 |
|---|---|---|
| 让车更倾向走近的前沿 | `bfs_weight`（默认 8.0） | `frontier_explorer.launch.py` / node 参数 |
| 减少掉头 | `turn_weight`（默认 0.35） | 同上 |
| 更远离障碍 | `obstacle_weight`、`obstacle_safe_distance` | 同上 |
| 别选贴车的前沿 | `min_frontier_distance` | 同上 |
| 别选墙角的点 | `clearance_target_cells`、`obstacle_safe_distance` | `exploration_bringup.launch.py` |
| 别跑到建筑外 | `exploration_bounds="xmin,xmax,ymin,ymax"` | 复杂建筑 launch |
| 选点失败多试几个 | `candidate_retry_count` | 同上 |
| 必到目标才换点 | `target_lock_until_reached` | 同上 |
| 难目标多久放弃 | `stall_detect_sec`、`target_blacklist_sec` | 同上 |

改完直接重启 launch 即可（参数在 launch 里透传，不用重编）。

### 7.2 第 1 层：改打分/硬过滤（算法层，最常用）

**文件**：`src/regional_explorer/regional_explorer/legacy/real_work/light_qifa.py`

**入口函数**：`decide_frontier(explored_map, robot_pos, previous_direction, config, exclude_mask, distance_to_obstacle)`

**它做的事**（顺序固定）：

```
1. build_frontier_mask      → 前沿 = 自由格(0) 且 4 邻域含未知(-1)
2. _reachable_distances     → 从机器人做 BFS，得到可达距离（不可达 = -1）
3. valid_mask 过滤
   a. 可达
   b. exclude_mask（黑名单）取反
   c. 欧氏距离 >= config.min_frontier_distance
   d. 到最近障碍 >= config.obstacle_safe_distance（硬过滤，不是打分）
4. 打分（分数越大越优，公式里都取负号 = 越小越好）
   score = -bfs_weight        * 归一化可达距离
           -turn_weight       * 转向代价
           -obstacle_weight   * 障碍接近代价
5. argmax → 返回 (row, col)
```

**常见改法**

| 想做什么 | 改哪里 |
|---|---|
| 新增一个打分项（如「信息增益」「到未知区域面积」） | 在 `decide_frontier` 里算 `frontier_xxx_cost`（形状 `(N,)`），加进 `frontier_scores`，同时写进 `FrontierDecision` 便于可视化 |
| 把某硬过滤改成软打分 | 把 `valid_mask &= (...)` 改成 `frontier_xxx_cost` 并加权重 |
| 改「前沿」定义（如要求 8 邻域、或至少 2 个未知邻居） | `build_frontier_mask` |
| 改距离度量（BFS 4 邻接 → 8 邻接） | `_reachable_distances` |
| 改转向代价 | `_turn_cost`（`previous_direction` 是上一个目标方向） |
| 加「候选必须离未知区域 N 格以内」 | 在 `valid_mask` 后加一条基于 `explored_map == -1` 的距离过滤 |

**新增参数的规范**：

1. 在 `LightHeuristicConfig` 加字段（带默认值，保持离线脚本兼容）；
2. 在 `frontier_explorer_node.py` 里 `declare_parameter` 同名参数并传进 `config_cls(...)`；
3. 在 `exploration_bringup.launch.py` 加 `DeclareLaunchArgument` 并透传；
4. 在 `rules`/文档（本文件）登记；
5. 跑回归：同一 world 跑 1–2 轮，对比「到达次数 / 脱困次数 / 覆盖率 / 耗时」。

### 7.3 第 2 层：改节点调度与目标生命周期（节点层）

**文件**：`src/regional_explorer/regional_explorer/frontier_explorer_node.py`

关键函数与职责：

| 函数 | 作用 | 什么时候改 |
|---|---|---|
| `_on_timer` | 主循环：取位姿 → 到达判定 → 保持目标 → 选点 → A* → 发布 | 改流程顺序 |
| `_keep_current_target` | 目标锁定：是否继续朝当前目标走 | 改「何时允许换目标」 |
| `_replan_if_path_blocked` | 路径被新障碍挡住时重规划（不换目标） | 改阻塞判定 |
| `_apply_exploration_bounds` | 过滤建筑外前沿 | 改边界策略 |
| `_build_exclude_mask` | 临时黑名单 / 永久排除 | 改黑名单语义 |
| `_update_stall` | 卡死检测 → 放弃目标 | 改「多久算卡死」 |
| `_record_traj_result` | 轨迹生成失败统计 | 改失败降级策略 |
| `_build_and_publish_path` | 阿克曼轨迹生成 + 回退折线 | 改轨迹/回退策略 |
| `candidate_retry_count` 循环（`_on_timer` 内） | 候选逐个重试 + 降级膨胀 | 改「选不到怎么办」 |

**已定型、不要轻易回退的设计**（回退会重新引入老问题）：

1. **未知格不参与膨胀**：只做硬阻挡。否则前沿目标自己会被未知格膨胀圈封死（历史 45 次 `A* 无路`）。
2. **不做零膨胀回退**：0 格净空路径最贴墙，是反复顶墙的根源。失败就换候选 / 不出目标。
3. **起点本体占位圈清空**（`start_clear_cells = ceil(inflation)+1`）：只清起点附近，圈外安全口径不变。
4. **目标锁定**：行进中不换目标，只有「A* 不可达」和「卡死」两个逃生口。
5. **软净空与硬膨胀分工**：`inflation_radius_cells` 是硬下限，`clearance_target_cells` 是软偏好；
   两者都传给轨迹模块，避免「上游撑开、下游捷径拉直」。

### 7.4 修改后的必做步骤

```bash
# 1) 若改了 legacy 副本，同步回 regional_legacy（两份代码要一致）
cp ros2_ws/src/regional_explorer/regional_explorer/legacy/core/{a_star.py,ackermann_traj.py,obstacle_distance.py} \
   regional_legacy/core/
cp ros2_ws/src/regional_explorer/regional_explorer/legacy/real_work/light_qifa.py \
   regional_legacy/real_work/

# 2) 语法检查
python3 -m py_compile ros2_ws/src/regional_explorer/regional_explorer/frontier_explorer_node.py

# 3) 重编（--symlink-install 下纯 .py 改动其实可跳过，但新增文件/入口要编）
cd ros2_ws && colcon build --symlink-install --packages-select regional_explorer

# 4) 回归（headless，避免 GUI 干扰）
ros2 launch bringup complex_building_explore.launch.py \
  world:=3d_enclosed.world headless:=true viz:=false
```

**验收看什么**（每次改动都记录）：

| 指标 | 怎么取 | 健康范围 |
|---|---|---|
| 到达次数 | `grep -c '到达当前目标' run_sim.log` | 越高越好 |
| 脱困次数 / 到达 | `grep -c '开始三点掉头脱困'` ÷ 到达 | 越低越好（当前约 2.0） |
| 伪完成 | `grep '探索完成'` 时的地图 unknown 比例 | 不应 >20% |
| 路径最小净空 | 对 `/exploration_path` 逐点查距离场 | 中位 ≥ 8 格 |
| 收敛 | 完成原因应为「未找到可达前沿」 | 不是「本轮不出目标」连续 |

### 7.5 绝对不要碰的硬约束

- `/map` 分辨率 **0.05 m/格**：不许缩放、重采样、改 `occupancy_grid_node` 的 resolution。
- 雷达量程 **0.15–8.0 m**（对齐真机 LD14P）：改 URDF 量程会同时改变建图与避障行为。
- `min_turn_radius_m` 必须在 explorer / follower / traj 三处一致（默认 0.35）。
- 不要同时跑 `exploration_bringup` 和 `nav_slam/2dpoints.launch.py`（两者都发 `/map` 与 TF）。

---

## 8. 快速命令索引

```bash
# 房间 + 可视化
bash ros2_ws/run_sim.sh

# 复杂封闭建筑 + 可视化
ros2 launch bringup complex_building_explore.launch.py world:=3d_enclosed.world

# 无窗口自检
ros2 launch bringup complex_building_explore.launch.py world:=3d_enclosed.world headless:=true viz:=false

# 关掉软净空做 A/B
ros2 launch bringup complex_building_explore.launch.py world:=3d_enclosed.world \
  clearance_target_cells:=0 clearance_weight:=0.0

# 关掉目标锁定做 A/B
ros2 launch bringup complex_building_explore.launch.py world:=3d_enclosed.world \
  target_lock_until_reached:=false
```
