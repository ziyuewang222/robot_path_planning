from __future__ import annotations

"""区域级几何启发式二维栅格自主探索 ROS2 节点。

输入：
  - nav_msgs/msg/OccupancyGrid（默认话题 map）
  - TF: <global_frame> -> <robot_frame>（默认 map -> base_link）

输出：
  - geometry_msgs/msg/PoseStamped（默认话题 exploration_goal）：下一个前沿目标点
  - nav_msgs/msg/Path（默认话题 exploration_path）：到该目标的初步 A* 路径（不做平滑）
  - std_msgs/msg/Bool（默认话题 exploration_done）：探索完成标志（latched/TRANSIENT_LOCAL，
    让后加入的订阅者也能收到 True）

算法本体复用 regional_legacy/real_work/light_qifa.py 与 regional_legacy/core/a_star.py，
通过参数 algorithm_root 指定 regional_legacy 目录（也可用环境变量 REGIONAL_LEGACY_ROOT）。

参数（名称 | 默认值 | 含义）：
  - algorithm_root            | ""                          | regional_legacy 目录（或环境变量 REGIONAL_LEGACY_ROOT）
  - map_topic                 | "map"                       | 订阅的栅格地图话题
  - goal_topic                | "exploration_goal"          | 前沿目标点发布话题
  - path_topic                | "exploration_path"          | 初步路径发布话题
  - global_frame              | "map"                       | TF 全局坐标系
  - robot_frame               | "base_link"                 | TF 机器人坐标系
  - replan_rate_hz            | 1.0                         | 规划循环频率（Hz）
  - occupied_threshold        | 50                          | 判为障碍占据的灰度阈值
  - done_topic                | "exploration_done"          | 探索完成标志发布话题（latched）
  - no_frontier_confirm_count | 10                          | 连续多少轮"没有产出可达目标"才判定完成（防地图刚到时误判）
  - min_known_cells           | 100                         | 已知栅格数（explored_map != -1 的个数）少于此值时不计入"无前沿"轮数（环境未就绪）
  - max_exploration_time_sec  | 1800.0                      | 从节点启动算起（单调钟），超过该秒数即判定完成
  - max_goal_count            | 200                         | 累计发布的不同目标数（世界坐标变化、且相对上一目标位移 ≥ goal_count_min_move_m 才计数）达到该值即判定完成
  - goal_count_min_move_m     | 1.0                         | 目标计数的最小有效位移（m），过滤前沿的小幅挪动
  - target_hysteresis_sec     | 10.0                        | 目标迟滞时长（s）：未超过该时长且当前目标仍可达/仍是前沿时沿用，不重新选点
  - target_keep_verify        | True                        | 沿用当前目标前是否校验"仍是前沿候选"
  - goal_tolerance            | 0.35                        | 到达判定半径（m）：机器人到当前目标小于该值即清空当前目标，允许换目标
  - stall_detect_sec          | 45.0                        | 机器人在该时长内位移小于阈值即视为"目标推进不了"
  - stall_move_eps            | 0.30                        | 目标推进检测的位移阈值（m）
  - target_blacklist_sec      | 180.0                       | 被放弃的目标进入临时黑名单的时长（s）
  - target_blacklist_radius   | 1.5                         | 黑名单半径（m）：该点周围这个半径内的前沿一并排除（须大于同墙相邻前沿间距，实测约 1.06 m）
  - traj_fail_blacklist_count | 2                           | 同一目标连续多少次无法生成阿克曼轨迹后拉黑换目标（复用 target_blacklist_sec 时长）
  - hard_target_fail_count    | 3                           | 同一目标累计多少次被"轨迹失败拉黑"后永久排除（跨黑名单周期累计，本次运行内不再选中）
  - exploration_bounds        | ""                          | 建筑外壳边界 "xmin,xmax,ymin,ymax"（米，世界坐标）；非空时过滤边界外前沿，为空不启用
  - map_save_prefix           | "maps/exploration_map"      | 完成时自研写出 map_server 标准 <prefix>.pgm/.yaml（空字符串表示跳过存图）
  - obstacle_safe_distance    | 5.0                         | 障碍安全包络半径（栅格，真机基准 0.25 m = 5 格）
  - min_frontier_distance    | 10.0                        | 前沿与机器人的直线（欧氏）栅格距离下限，避免选中贴在车身上的退化前沿
  - inflation_radius_cells    | 5.0                         | 前向安全的障碍硬膨胀半径（格），真机安全包络 0.25m = 5 格；同时作为阿克曼轨迹的 safety_cells（不新增 traj_safety_cells）
  - enable_ackermann_traj     | True                        | 是否把 A* 折线转成阿克曼可行轨迹再发布；False 时退回原始 A* 折线（A/B 对比用）
  - wheelbase_m               | 0.144                       | 车辆轴距（m），传给 AckermannTrajConfig
  - max_steer_rad             | 0.489                       | 前轮最大转角（rad），传给 AckermannTrajConfig
  - min_turn_radius_m         | 0.35                        | 最小转弯半径（m），传给 AckermannTrajConfig（0.0 表示用 L/tan(max_steer)）
  - traj_ds_m                 | 0.05                        | 轨迹等弧长重采样步长（m）
  - traj_v_max                | 0.30                        | 轨迹速度上限（m/s）
  - traj_v_min                | 0.08                        | 轨迹速度下限（m/s）
  - traj_a_lat_max            | 0.60                        | 横向加速度上限（m/s²），决定弯道限速
  - traj_a_long_max           | 0.25                        | 纵向加速度上限（m/s²），速度剖面扫描
  - use_eight_neighbors       | True                        | A* 是否用 8 邻接（对角步代价 sqrt(2)，禁止穿角）
  - treat_unknown_as_obstacle | True                        | A* 是否把未知格(-1)当硬障碍；候选前沿可达性本来只在已知自由格上算，故不会误杀合法前沿
  - path_block_check_m        | 1.0                         | 路径受阻检查长度（m）：缓存路径前该距离内出现新障碍即立即重规划（跳过目标迟滞）

完成判定（Step 3）：
  - 以下任一条件满足即调用 _declare_done()，发布 True 到 done_topic，并停止发布 goal/path：
    1) 连续 no_frontier_confirm_count 轮"没有产出可达目标"（无前沿，或 A* 规划失败）；
    2) 运行时长超过 max_exploration_time_sec；
    3) 累计发布目标数达到 max_goal_count。
  - 区分"因膨胀无路"与"真不可达"（避免过早判完成、也避免目标发布永久停止）：A* 在
    inflation_radius_cells 下规划失败时，会再用 inflation_radius_cells=0 重试一次。
    若重试成功 -> 只是"膨胀暂时进不去"（如窄门被膨胀封住、目标格贴着安全边界），
    则**直接采用这条零膨胀路径正常发布目标与路径**（仍重置 streak、按世界坐标判新目标、
    累加 _goal_count），并打一条 10s 节流 info；目标位置本身的安全性由 light_qifa 的
    候选过滤（dist >= obstacle_safe_distance）保证。只有重试也失败，才算"真不可达"，
    计入 streak。这样任何前沿都不会让目标发布长期停摆。
  - "没有产出可达目标"计数有两道门槛（统一在 _bump_no_target 内）：尚未产出过任何目标
    （_goal_count == 0），或已知栅格数 < min_known_cells，均视为环境未就绪、不计入 streak。
    这两道门槛只作用于条件 1)；超时与最大目标数不受影响。
  - 机器人位姿查不到（TF 未就绪）属于环境未就绪，不计入"无前沿"轮数。
  - 以下两种"看起来没有目标"的情况也**不计入** streak（避免 10 轮后永久提前 done）：
    1) 机器人当前所在栅格不是自由格（未知/障碍）：BFS 会全 -1，属于定位/建图未就绪；
    2) 候选只是被临时黑名单全部排除：用不带 exclude_mask 的参数再探测一次，若能选出
       目标则说明黑名单暂时封住了全部候选，此时**清空黑名单**（否则会永久空转），
       本轮跳过（探测结果不发布，避免误用被拉黑目标）。
  - 目标迟滞：当前目标发布后 target_hysteresis_sec 内，若该目标仍是前沿候选且 A* 可达，
    则直接重新发布同一目标/路径，不重新选点、不累加 _goal_count，避免每隔几秒换目标。
  - 当前目标的"世界坐标"与"最近发布目标世界坐标"合并为同一个字段
    _current_target_world（附发布时间 _current_target_time），不再保存栅格行列，避免
    /map origin 漂移导致黑名单错位、_goal_count 虚增。
  - _declare_done() 只会执行一次（_done 守卫），完成时会尝试存图，存图失败不打断节点。
    存图文件名带完成原因后缀：<prefix>_no_frontier / _timeout / _max_goals。
"""

import inspect
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener


def _as_bool(value) -> bool:
    """把布尔参数读成 bool：兼容 launch 以字符串（"True"/"False"）覆盖的情形。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def save_occupancy_grid_map(
    msg: OccupancyGrid, prefix: str, occupied_threshold: int = 50
) -> Tuple[bool, str]:
    """把一帧 OccupancyGrid 写成 map_server 标准的 <prefix>.pgm + <prefix>.yaml。

    返回 (是否成功, 说明文字)。内部自行 try/except，绝不向外抛异常。

    格式约定（yaml 中 negate: 0，即白=自由、黑=占据）：
      - data == -1（未知）            -> 205（灰）
      - data >= occupied_threshold     -> 0  （黑，占据）
      - 其它（0 .. threshold-1）       -> 255（白，自由）

    PGM 为二进制 P5。注意 PGM 第 0 行是图像顶部，而 OccupancyGrid 的 data 索引 0
    是左下角，因此输出到图像第 r 行时取栅格第 (h-1-r) 行。
    """
    try:
        # 规整前缀，防止 prefix 以 "/" 结尾时写出 "image: .pgm"。
        prefix = prefix.rstrip("/")
        info = msg.info
        w = int(info.width)
        h = int(info.height)
        if w <= 0 or h <= 0:
            return (False, f"地图尺寸非法: {w}x{h}")
        data = list(msg.data)

        # 目录准备：前缀可能带目录（默认 maps/exploration_map）。
        d = os.path.dirname(prefix)
        if d:
            os.makedirs(d, exist_ok=True)

        # ------- 二进制 PGM（P5）-------
        pgm_path = f"{prefix}.pgm"
        header = b"P5\n%d %d\n255\n" % (w, h)
        pixels = bytearray(w * h)
        for r in range(h):
            src_row = h - 1 - r  # OccupancyGrid 左下角原点 -> 图像从上到下
            base_src = src_row * w
            base_dst = r * w
            for c in range(w):
                v = data[base_src + c]
                if v < 0:
                    pixels[base_dst + c] = 205
                elif v >= occupied_threshold:
                    pixels[base_dst + c] = 0
                else:
                    pixels[base_dst + c] = 255
        with open(pgm_path, "wb") as f:
            f.write(header)
            f.write(bytes(pixels))

        # ------- 同名 yaml（字段顺序固定）-------
        q = info.origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        yaml_path = f"{prefix}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"image: {os.path.basename(prefix)}.pgm\n")
            f.write(f"resolution: {info.resolution}\n")
            f.write(
                "origin: [{:.6f}, {:.6f}, {:.6f}]\n".format(
                    info.origin.position.x, info.origin.position.y, yaw
                )
            )
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.196\n")

        return (True, f"已写出 {prefix}.pgm 与 {prefix}.yaml ({w}x{h}@{info.resolution})")
    except Exception as exc:  # 存图失败绝不打断节点
        return (False, f"写图失败: {exc}")


def _yaw_from_quaternion_optional(x: float, y: float, z: float, w: float) -> Optional[float]:
    """从四元数解出绕 Z 轴的 yaw；四元数全为 0（无效）时返回 None。

    这里不引入 tf_transformations，避免额外依赖。无效四元数时让调用方回退原始 A* 折线，
    而不是把 yaw 猜成 0（猜错会让阿克曼轨迹从错误朝向起步）。
    """
    if abs(x) < 1e-12 and abs(y) < 1e-12 and abs(z) < 1e-12 and abs(w) < 1e-12:
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _load_algorithm_modules(algorithm_root: str):
    """从 regional_legacy 目录动态导入算法模块，保证与离线调参使用同一份代码。

    返回值新增 ``AckermannTrajConfig`` / ``plan_ackermann_trajectory`` /
    ``distance_to_obstacle``，供探索器把 A* 折线转成阿克曼可行轨迹。

    `decide_frontier` 的返回是一个**薄包装**，支持 ``distance_to_obstacle=`` 关键字：
      * 若上游 `decide_frontier` 已原生支持该参数（未来版本），直接透传；
      * 若尚未支持（当前 light_qifa），则在该次调用期间临时替换模块内的
        ``_distance_to_obstacle``，让它的硬过滤与软代价复用探索器预先算好的距离场。
    这样整轮（前沿决策 + A* 膨胀 + 轨迹生成）只做一次距离变换，满足性能硬要求；
    不修改 light_qifa.py 源码。
    """
    if not algorithm_root:
        raise RuntimeError(
            "未指定 algorithm_root，请设置 ROS 参数 algorithm_root 或环境变量 "
            "REGIONAL_LEGACY_ROOT 指向 regional_legacy 目录"
        )
    if not os.path.isdir(algorithm_root):
        raise RuntimeError(f"algorithm_root 目录不存在: {algorithm_root}")
    if algorithm_root not in sys.path:
        sys.path.insert(0, algorithm_root)

    from real_work.light_qifa import LightHeuristicConfig, decide_frontier
    from core.a_star import AStarPlanner
    from core.ackermann_traj import AckermannTrajConfig, plan_ackermann_trajectory
    from core.obstacle_distance import distance_to_obstacle

    # 上游 decide_frontier 现已原生支持 distance_to_obstacle 关键字（默认 None，向后兼容），
    # 因此这里直接薄封装透传即可，**不再**用"临时替换模块级距离函数"的 hack（那种做法在
    # 多线程/异常路径下很脆）。
    def decide_frontier_reusing_distance(
        explored_map,
        robot_pos,
        previous_direction=None,
        config=None,
        exclude_mask=None,
        distance_to_obstacle=None,
    ):
        """调用 decide_frontier，并复用一个预先算好的"到最近障碍距离场"。"""
        return decide_frontier(
            explored_map,
            robot_pos,
            previous_direction=previous_direction,
            config=config,
            exclude_mask=exclude_mask,
            distance_to_obstacle=distance_to_obstacle,
        )

    return (
        LightHeuristicConfig,
        decide_frontier_reusing_distance,
        AStarPlanner,
        AckermannTrajConfig,
        plan_ackermann_trajectory,
        distance_to_obstacle,
    )


class FrontierExplorerNode(Node):
    """订阅栅格地图与 TF，周期性输出前沿目标点与初步路径。"""

    def __init__(self) -> None:
        super().__init__("frontier_explorer")

        self.declare_parameter("algorithm_root", os.environ.get("REGIONAL_LEGACY_ROOT", ""))
        self.declare_parameter("map_topic", "map")
        self.declare_parameter("goal_topic", "exploration_goal")
        self.declare_parameter("path_topic", "exploration_path")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("replan_rate_hz", 1.0)
        self.declare_parameter("occupied_threshold", 50)

        # 完成判定（Step 3）相关参数。
        self.declare_parameter("done_topic", "exploration_done")
        self.declare_parameter("no_frontier_confirm_count", 10)
        self.declare_parameter("min_known_cells", 100)
        self.declare_parameter("max_exploration_time_sec", 1800.0)
        self.declare_parameter("max_goal_count", 200)
        # 目标计数的最小有效位移（m）：新目标相对上一个已发布目标位移小于该值时不计数，
        # 避免前沿"小幅挪动"把目标预算耗光（实测会导致覆盖未收敛就提前结束）。
        self.declare_parameter("goal_count_min_move_m", 1.0)
        self.declare_parameter("map_save_prefix", "maps/exploration_map")

        # 目标迟滞：避免每次决策都换目标（实测中位 5.3s 就换一次，160 个目标只到达 2 次）。
        self.declare_parameter("target_hysteresis_sec", 10.0)
        self.declare_parameter("target_keep_verify", True)
        # 到达判定半径（m）：机器人到当前目标小于该值即承认"已到达"，清空当前目标。
        self.declare_parameter("goal_tolerance", 0.35)

        # 目标推进检测 + 临时黑名单（目标推进不了就放弃并换一个前沿）。
        self.declare_parameter("stall_detect_sec", 45.0)
        self.declare_parameter("stall_move_eps", 0.30)
        self.declare_parameter("target_blacklist_sec", 180.0)
        self.declare_parameter("target_blacklist_radius", 1.5)
        # 修复 B：同一目标连续多少次无法生成阿克曼轨迹就拉黑换目标（复用上面的黑名单）。
        self.declare_parameter("traj_fail_blacklist_count", 2)
        # 修复 2：同一目标累计多少次被"轨迹失败拉黑"后永久排除（跨黑名单周期累计）。
        # 背景：180s 黑名单到期后会再次选中同一批"难生成/难起步"目标，形成目标空转。
        self.declare_parameter("hard_target_fail_count", 3)
        # 修复 C：建筑外壳边界 "xmin,xmax,ymin,ymax"（米，世界坐标）；空字符串表示不启用。
        # 复杂建筑入口可传 "-19,19.5,-19,15.5"，避免把 16% 的时间耗在建筑外西侧空地上。
        self.declare_parameter("exploration_bounds", "")

        # 纯几何启发式权重，默认取 real_work 调参得到的较优配置。
        self.declare_parameter("bfs_weight", 8.0)
        self.declare_parameter("turn_weight", 0.35)
        self.declare_parameter("obstacle_weight", 1.0)
        self.declare_parameter("obstacle_safe_distance", 5.0)
        self.declare_parameter("min_frontier_distance", 10.0)
        # 前向安全的障碍硬膨胀半径（格）：真机安全包络 0.25m = 5 格；部署侧需显式传值，
        # 因为 A* 库的默认值是 0（不膨胀，供离线/训练/评测链路保持原行为）。
        self.declare_parameter("inflation_radius_cells", 5.0)

        # ---- 阿克曼轨迹生成参数（复用 inflation_radius_cells 作为 safety_cells）----
        # 关掉 enable_ackermann_traj 时退回原始 A* 折线，便于 A/B 对比。
        self.declare_parameter("enable_ackermann_traj", True)
        self.declare_parameter("wheelbase_m", 0.144)
        self.declare_parameter("max_steer_rad", 0.489)
        self.declare_parameter("min_turn_radius_m", 0.35)
        self.declare_parameter("traj_ds_m", 0.05)
        self.declare_parameter("traj_v_max", 0.30)
        self.declare_parameter("traj_v_min", 0.08)
        self.declare_parameter("traj_a_lat_max", 0.60)
        self.declare_parameter("traj_a_long_max", 0.25)
        # ---- A* 搜索策略（部署侧默认打开）----
        # 候选前沿的 BFS 可达性只在已知自由格上计算，因此把未知格当障碍不会误杀合法前沿。
        self.declare_parameter("use_eight_neighbors", True)
        self.declare_parameter("treat_unknown_as_obstacle", True)
        # ---- 路径受阻检查 ----
        # 缓存路径前该距离内出现新观测障碍即清空当前目标、立即重规划（跳过目标迟滞）。
        self.declare_parameter("path_block_check_m", 1.0)

        algorithm_root = self.get_parameter("algorithm_root").value
        (
            config_cls,
            self._decide_frontier,
            planner_cls,
            self._AckermannTrajConfig,
            self._plan_ackermann_trajectory,
            self._distance_to_obstacle,
        ) = _load_algorithm_modules(algorithm_root)
        self._planner = planner_cls()

        self._heuristic_config = config_cls(
            bfs_weight=float(self.get_parameter("bfs_weight").value),
            turn_weight=float(self.get_parameter("turn_weight").value),
            obstacle_weight=float(self.get_parameter("obstacle_weight").value),
            obstacle_safe_distance=float(self.get_parameter("obstacle_safe_distance").value),
            min_frontier_distance=float(self.get_parameter("min_frontier_distance").value),
        )
        # 兜底守卫用：选出的前沿到机器人的栅格距离小于该值即跳过本轮（不发布退化目标）。
        self._min_frontier_distance = float(
            self.get_parameter("min_frontier_distance").value
        )
        # A* 障碍硬膨胀半径（格）：调用 plan 时显式传入，避免依赖库默认值（库默认 0）。
        self._inflation_radius_cells = int(
            self.get_parameter("inflation_radius_cells").value
        )
        # ---- 修复 1：距离场上限（性能）----
        # 在线调用方只关心 `dist < inflation_radius_cells`、`dist >= obstacle_safe_distance`
        # 以及软代价 clip(1 - d/safe, 0, 1)（d >= safe 时恒 0），因此超出安全距离的值完全等价。
        # 上限取 max(safe, inflation) + 2 格：保证所有阈值比较都落在未截断区间内。
        # 复杂建筑 946×1063 纯 Python chamfer 单轮约 2.43s，上限化向量化后为毫秒级。
        self._distance_field_max_distance = (
            max(
                float(self._heuristic_config.obstacle_safe_distance),
                float(self._inflation_radius_cells),
            )
            + 2.0
        )
        # 外部传入的距离场函数是否接受 max_distance 关键字；不支持时只回退一次（见
        # _compute_distance_field），保证兼容旧实现/自定义可调用对象。
        self._distance_accepts_max_distance = self._supports_max_distance_argument(
            self._distance_to_obstacle
        )
        # 阿克曼轨迹参数：traj_safety_cells 直接复用 _inflation_radius_cells（房间 5 / 复杂建筑 3）。
        self._enable_ackermann_traj = _as_bool(
            self.get_parameter("enable_ackermann_traj").value
        )
        self._wheelbase_m = float(self.get_parameter("wheelbase_m").value)
        self._max_steer_rad = float(self.get_parameter("max_steer_rad").value)
        self._min_turn_radius_m = float(self.get_parameter("min_turn_radius_m").value)
        self._traj_ds_m = float(self.get_parameter("traj_ds_m").value)
        self._traj_v_max = float(self.get_parameter("traj_v_max").value)
        self._traj_v_min = float(self.get_parameter("traj_v_min").value)
        self._traj_a_lat_max = float(self.get_parameter("traj_a_lat_max").value)
        self._traj_a_long_max = float(self.get_parameter("traj_a_long_max").value)
        # A* 搜索策略。
        self._use_eight_neighbors = _as_bool(
            self.get_parameter("use_eight_neighbors").value
        )
        self._treat_unknown_as_obstacle = _as_bool(
            self.get_parameter("treat_unknown_as_obstacle").value
        )
        # 路径受阻检查长度（m）。
        self._path_block_check_m = max(
            float(self.get_parameter("path_block_check_m").value), 0.0
        )

        self._global_frame = self.get_parameter("global_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)

        # 完成判定参数/状态（Step 3）。
        self._no_frontier_confirm_count = int(
            self.get_parameter("no_frontier_confirm_count").value
        )
        self._min_known_cells = int(self.get_parameter("min_known_cells").value)
        self._max_exploration_time_sec = float(
            self.get_parameter("max_exploration_time_sec").value
        )
        self._max_goal_count = int(self.get_parameter("max_goal_count").value)
        self._goal_count_min_move = float(
            self.get_parameter("goal_count_min_move_m").value
        )
        self._map_save_prefix = str(self.get_parameter("map_save_prefix").value)
        # 目标迟滞参数。
        self._target_hysteresis_sec = float(
            self.get_parameter("target_hysteresis_sec").value
        )
        self._target_keep_verify = _as_bool(
            self.get_parameter("target_keep_verify").value
        )
        self._goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        # 目标推进检测参数。
        self._stall_detect_sec = float(self.get_parameter("stall_detect_sec").value)
        self._stall_move_eps = float(self.get_parameter("stall_move_eps").value)
        self._target_blacklist_sec = float(
            self.get_parameter("target_blacklist_sec").value
        )
        self._target_blacklist_radius = float(
            self.get_parameter("target_blacklist_radius").value
        )
        # 修复 B：连续轨迹失败计数阈值，至少为 1（每次失败立即拉黑）。
        self._traj_fail_blacklist_count = max(
            int(self.get_parameter("traj_fail_blacklist_count").value), 1
        )
        # 修复 B：同一目标的连续轨迹失败次数。key = (世界 x, 世界 y) 各 round 到 0.01m，
        # 与目标判重/黑名单使用同一坐标口径；成功生成轨迹或目标切换时清理。
        self._traj_fail_counts: dict = {}

        # ---- 修复 2：顽固目标永久排除（跨黑名单周期）----
        # 修复 B 的黑名单是 180s 临时的，到期后同一批"难生成/难起步"目标会被反复选中，
        # 造成目标空转（实测最后 365s 覆盖率只 +0.03pp）。这里维护**不随黑名单到期**
        # 的累计失败次数与永久排除列表：
        #   _hard_target_fail_totals[(x,y)]：每次因轨迹失败被拉黑时 +1（成功则清零）；
        #   _hard_targets：累计次数达到 hard_target_fail_count 的目标，本次运行内永久排除。
        self._hard_target_fail_count = max(
            int(self.get_parameter("hard_target_fail_count").value), 1
        )
        self._hard_target_fail_totals: dict = {}
        # 永久排除列表上限：超过时删最早的一个，避免字典/掩码无限增长拖慢每轮决策。
        self._hard_targets_limit = 200
        self._hard_targets: List[Tuple[float, float]] = []
        # 永久排除掩码半径（格）：用 2 格覆盖坐标离散化误差，避免同一目标"漏网"。
        self._hard_target_radius_cells = 2.0
        # 修复 C：解析建筑外壳边界；非法或为空时按"不启用"处理（行为与改造前一致）。
        bounds_text = str(self.get_parameter("exploration_bounds").value).strip()
        self._exploration_bounds = self._parse_exploration_bounds(bounds_text)
        if bounds_text and self._exploration_bounds is None:
            self.get_logger().warn(
                "exploration_bounds 格式非法（应为 \"xmin,xmax,ymin,ymax\"，单位米）："
                f"{bounds_text}，按不启用处理"
            )

        # 目标推进检测状态：上次"有效移动"的机器人世界坐标与时刻（秒）。
        self._last_move_pose: Optional[Tuple[float, float]] = None
        self._last_move_time: Optional[float] = None
        # 临时黑名单：[(世界 x, 世界 y, 过期时刻 s), ...]。
        self._blacklist: List[Tuple[float, float, float]] = []

        self._grid_msg: Optional[OccupancyGrid] = None
        # 最近一帧原始地图消息，供完成时存图使用。
        self._last_map_msg: Optional[OccupancyGrid] = None
        self._previous_direction: Optional[Tuple[float, float]] = None

        # 完成标志：True 之后不再规划、不再发布 goal/path。
        self._done = False
        # 连续"没有产出可达目标"（无前沿或 A* 失败）的轮数。
        self._no_frontier_streak = 0
        # 累计发布的目标总数。
        self._goal_count = 0
        # 已发布目标的**世界坐标**（合并了"当前目标"与"上次发布目标"两份状态，避免打架）：
        #   - 目标迟滞：now - _current_target_time < target_hysteresis_sec 时尝试沿用；
        #   - _goal_count：按世界坐标（round 到 0.01m）区分"新目标"与"同一目标重发"；
        #   - stall 黑名单：直接用该世界坐标，不再用当前 origin 解释旧栅格行列。
        # 到达目标（距离 < goal_tolerance）或被加入黑名单时清空。
        self._current_target_world: Optional[Tuple[float, float]] = None
        # 当前目标最近一次发布时刻（ROS 时钟，秒），用于迟滞判定。
        self._current_target_time: Optional[float] = None
        # 最近一次发布的世界坐标路径（阿克曼轨迹或原始 A* 折线，二选一）。
        # 供"路径受阻检查"使用：每轮截取前 path_block_check_m 米逐点查当前地图。
        self._current_path_world: Optional[List[Tuple[float, float]]] = None
        # 节点启动时刻（单调钟，避免 use_sim_time 下 /clock 未就绪导致误判超时）。
        self._start_monotonic = time.monotonic()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_pub = self.create_publisher(
            PoseStamped, self.get_parameter("goal_topic").value, 10
        )
        self._path_pub = self.create_publisher(
            Path, self.get_parameter("path_topic").value, 10
        )
        # 完成标志发布者：TRANSIENT_LOCAL（latched），后加入的订阅者也能收到 True。
        self._done_pub = self.create_publisher(
            Bool,
            self.get_parameter("done_topic").value,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(
            OccupancyGrid, self.get_parameter("map_topic").value, self._on_map, 1
        )

        period = 1.0 / max(float(self.get_parameter("replan_rate_hz").value), 1e-3)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"frontier_explorer 启动 | algorithm_root={algorithm_root} | "
            f"map={self.get_parameter('map_topic').value} | "
            f"goal={self.get_parameter('goal_topic').value} | "
            f"path={self.get_parameter('path_topic').value} | "
            f"done={self.get_parameter('done_topic').value} | "
            f"confirm={self._no_frontier_confirm_count} | "
            f"min_known={self._min_known_cells} | "
            f"timeout={self._max_exploration_time_sec}s | "
            f"max_goal={self._max_goal_count} | "
            f"hysteresis={self._target_hysteresis_sec}s/verify={self._target_keep_verify}/"
            f"tol={self._goal_tolerance}m | "
            f"stall={self._stall_detect_sec}s/{self._stall_move_eps}m | "
            f"blacklist={self._target_blacklist_sec}s/"
            f"r={self._target_blacklist_radius}m/"
            f"traj_fail={self._traj_fail_blacklist_count}次/"
            f"hard_fail={self._hard_target_fail_count}次 | "
            f"dist_cap={self._distance_field_max_distance:g}格 | "
            f"bounds={self._exploration_bounds} | "
            f"tf={self._global_frame}->{self._robot_frame} | "
            f"ackermann={self._enable_ackermann_traj}"
            f"(L={self._wheelbase_m}m/steer {self._max_steer_rad}rad/"
            f"Rmin={self._min_turn_radius_m}m/safety {self._inflation_radius_cells}格) | "
            f"astar8={self._use_eight_neighbors}/unknown_block="
            f"{self._treat_unknown_as_obstacle} | "
            f"path_block={self._path_block_check_m}m"
        )

    # ------------------------------------------------------------- 距离场（上限化）
    @staticmethod
    def _supports_max_distance_argument(distance_callable) -> bool:
        """判断外部传入的距离场函数是否接受 ``max_distance`` 关键字。

        用 ``inspect.signature`` 探测一次即可（避免每轮 try/except 影响热路径）：
          - 显式参数名 ``max_distance`` → 支持；
          - 有 ``**kwargs``（VAR_KEYWORD） → 支持；
          - 内建函数等拿不到签名（ValueError/TypeError） → 保守按"不支持"处理。
        """
        try:
            parameters = inspect.signature(distance_callable).parameters.values()
        except (TypeError, ValueError):
            return False
        for parameter in parameters:
            if parameter.name == "max_distance":
                return True
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    def _compute_distance_field(
        self, obstacle_mask: np.ndarray, max_distance: float
    ) -> np.ndarray:
        """计算本轮唯一一份距离场（上限化，毫秒级）。

        上限化依据：调用方只做阈值比较（`< inflation_radius` / `>= safe_distance`）与
        软代价 `clip(1 - d/safe, 0, 1)`（d >= safe 时恒 0），因此 `d >= safe` 的精确值
        对判定没有意义，截断到 ``max_distance = max(safe, inflation) + 2`` 完全等价。

        若外部注入的 ``_distance_to_obstacle`` 是旧实现（不接受 ``max_distance``），
        首次调用捕获 ``TypeError`` 后退回不支持上限的整图计算，并永久关闭该标志，
        保证兼容性与鲁棒性（不重复试探、不影响后续每轮性能）。
        """
        if self._distance_accepts_max_distance:
            try:
                return self._distance_to_obstacle(
                    obstacle_mask, max_distance=float(max_distance)
                )
            except TypeError:
                # 只回退一次：标记后后续轮次直接走不支持上限的分支。
                self._distance_accepts_max_distance = False
                self.get_logger().warn(
                    "外部距离场函数不支持 max_distance，退回整图精确距离场"
                    "（性能会下降，但判定结果不变）",
                    throttle_duration_sec=10.0,
                )
        return self._distance_to_obstacle(obstacle_mask)

    # ------------------------------------------------------------------ 回调
    def _on_map(self, msg: OccupancyGrid) -> None:
        self._grid_msg = msg
        self._last_map_msg = msg

    def _on_timer(self) -> None:
        # 完成后不再规划、不再发布 goal/path。
        if self._done:
            return

        # 超时判定：用单调钟从节点启动算起（不受 use_sim_time / clock 影响）。
        elapsed = time.monotonic() - self._start_monotonic
        if elapsed > self._max_exploration_time_sec:
            # 存图文件名带 _timeout 后缀，便于和自然完成（_no_frontier）区分。
            self._declare_done(
                f"超过最大探索时长 {self._max_exploration_time_sec}s",
                save_suffix="timeout",
            )
            return

        if self._grid_msg is None:
            return

        # 机器人位姿查不到（TF 未就绪/越界）属于环境未就绪，不计入"无前沿"轮数。
        # 只查一次 TF，同时得到世界坐标、yaw（阿克曼轨迹起点用）与栅格坐标（规划用）。
        robot_pose = self._lookup_robot_pose_world()
        if robot_pose is None:
            return
        robot_x, robot_y, robot_yaw = robot_pose
        robot_world = (robot_x, robot_y)
        robot_grid = self._world_to_grid(robot_world)
        if robot_grid is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        # 目标推进检测：机器人长时间没动 -> 放弃当前目标并加入临时黑名单。
        self._update_stall(now_sec, robot_world)

        explored_map = self._to_explored_map(self._grid_msg)

        # P0-4 判定 1：机器人当前所在格不是自由格（未知=-1 / 障碍=1）时，BFS 会全 -1、
        # decide_frontier 必然返回 None。这属于定位/建图未就绪，**不计 streak**，否则连续
        # no_frontier_confirm_count 轮后会永久提前 done。
        robot_cell_value = int(explored_map[robot_grid[0], robot_grid[1]])
        if robot_cell_value != 0:
            self.get_logger().info(
                f"机器人当前所在格不是自由格（值={robot_cell_value}），"
                f"本轮不计入探索完成判定",
                throttle_duration_sec=5.0,
            )
            return

        # ---- 路径受阻检查（必须在目标迟滞判断之前）----
        # 缓存路径前 path_block_check_m 米内若出现**新观测障碍**，说明原路线已不可走：
        # 立刻清空当前目标（跳过迟滞）并在本轮重新选点。未知格只提示、不重规划——本轮
        # treat_unknown_as_obstacle=True 下正常不会出现这种路径。
        self._replan_if_path_blocked(explored_map)

        # ---- 每轮只算一次距离场，后续全程复用 ----
        # 距离场以**已知障碍**为源（未知格不参与安全距离，否则探索前沿永远不可达）。
        # 复用点：decide_frontier 的候选硬过滤/软代价、A* 的膨胀、阿克曼轨迹生成。
        # 这样整轮 distance_to_obstacle 只被调用 1 次。
        # 修复 1：传 max_distance 上限（max(safe, inflation)+2 格），让无 scipy 环境走
        # "向量化 8 邻域上限化"快速路径（946×1063 从 ~2.43s 降到毫秒级）。
        # 有 scipy 时仍是精确欧氏 + np.minimum 截断，行为与整图完全一致。
        distance_field = self._compute_distance_field(
            explored_map == 1, self._distance_field_max_distance
        )
        # A* / 轨迹用的派生场：a_star 在收到 distance_field 时直接用 `field < R` 作为
        # 阻挡掩膜，不会再单独并入 unknown；若只传"已知障碍"距离场，未知格的
        # distance 可能 >= R 而从膨胀里漏过去。因此这里把未知格的距离置 0（**只复制
        # 一次数组，不重新做距离变换**），让膨胀掩膜真正挡住未知格。
        # decide_frontier 仍然用纯"已知障碍"场（未知格不能参与候选安全距离，否则前沿
        # 永远不可达）；轨迹模块自身的 _point_safe 也会硬判 explored == -1，
        # 因此置 0 不改变它的安全语义。
        traj_distance_field = distance_field
        if np.any(explored_map == -1):
            traj_distance_field = distance_field.copy()
            traj_distance_field[explored_map == -1] = 0.0
        planner_distance_field = (
            traj_distance_field
            if self._treat_unknown_as_obstacle
            else distance_field
        )

        # P0-3 到达判定（简化实现）：机器人到当前目标 < goal_tolerance 即视为到达，
        # 清空当前目标，允许下一轮换目标。
        if self._current_target_world is not None:
            dist_to_current = math.hypot(
                self._current_target_world[0] - robot_world[0],
                self._current_target_world[1] - robot_world[1],
            )
            if dist_to_current < self._goal_tolerance:
                self.get_logger().info(
                    f"到达当前目标 ({self._current_target_world[0]:.2f}, "
                    f"{self._current_target_world[1]:.2f})，距离 {dist_to_current:.3f}m，"
                    f"允许重新选点"
                )
                self._current_target_world = None
                self._current_target_time = None
                # 到达后旧路径已走完，清掉缓存避免下一轮误判"路径受阻"。
                self._current_path_world = None

        # P0-3 目标迟滞：当前目标仍有效、仍是前沿候选且 A* 可达 -> 直接沿用并重发，
        # 跳过本轮重新选点，也不累加 _goal_count。
        if self._keep_current_target(
            now_sec,
            explored_map,
            robot_grid,
            robot_pose,
            planner_distance_field,
            traj_distance_field,
        ):
            return

        # 黑名单过滤掩码：以被放弃目标的世界坐标为圆心、target_blacklist_radius 为半径，
        # 圆内前沿本轮不作为候选；过期黑名单在 _build_exclude_mask 里顺便清理。
        exclude_mask = self._build_exclude_mask(now_sec)
        decision = self._decide_frontier(
            explored_map,
            robot_grid,
            previous_direction=self._previous_direction,
            config=self._heuristic_config,
            exclude_mask=exclude_mask,
            # 复用本轮唯一一次距离场，保证 decide_frontier 不再重复计算。
            distance_to_obstacle=distance_field,
        )

        if decision.target is None:
            # 只是被临时黑名单排除，不等于"真无前沿"。
            # 用不带 exclude_mask 的参数再探测一次（只判定原因，不发布，避免误用被拉黑目标）。
            if exclude_mask is not None:
                probe = self._decide_frontier(
                    explored_map,
                    robot_grid,
                    previous_direction=self._previous_direction,
                    config=self._heuristic_config,
                    exclude_mask=None,
                    distance_to_obstacle=distance_field,
                )
                if probe.target is not None and self._blacklist:
                    # 临时黑名单把所有候选都排除了：继续保留只会永久空转（每轮都跳过），
                    # 因此清空黑名单，让下一轮重新选点。本轮仍不计 streak。
                    # 注意：只有**临时黑名单**非空时才清空；若候选只是被"顽固目标永久
                    # 排除列表"挡住，清空黑名单不起作用（会每轮重复探测→空转），
                    # 这种情况按"真无前沿"处理（下方 _bump_no_target），让探索正常收敛。
                    self.get_logger().info(
                        "当前候选全部被临时黑名单排除，清空黑名单以便继续探索"
                    )
                    self._blacklist.clear()
                    return
            # 真无前沿：计入"本轮没产出可达目标"（环境未就绪时 _bump_no_target 内部不计数）。
            streak_before = self._no_frontier_streak
            self._bump_no_target("未找到可达前沿，地图已探索完毕", explored_map)
            # 仅当计数真正生效（已产出过目标且已知栅格足够）后才打印进度，避免刚启动就刷 0/10。
            if not self._done and self._no_frontier_streak > streak_before:
                self.get_logger().info(
                    f"未找到可达前沿（第 {self._no_frontier_streak}/"
                    f"{self._no_frontier_confirm_count} 轮）",
                    throttle_duration_sec=5.0,
                )
            return

        # 修复 C：过滤建筑外壳之外的前沿（未配置 exploration_bounds 时原样返回）。
        # 必须在"距离过近"守卫与 A* 之前完成，避免为边界外目标白算一条路径。
        decision = self._apply_exploration_bounds(
            decision, explored_map, robot_grid, exclude_mask, distance_field
        )

        # 兜底守卫：绝不发布贴在机器人身上的退化前沿（否则控制端第一帧即判"已到达"、
        # 车全程不动、地图不增长）。此处不计入"无前沿"轮数，也不累加目标数。
        target_row, target_col = decision.target
        grid_distance = math.hypot(target_row - robot_grid[0], target_col - robot_grid[1])
        if grid_distance < self._min_frontier_distance:
            self.get_logger().warn(
                f"选出的前沿与机器人直线距离过近（{grid_distance:.1f} 格 < "
                f"{self._min_frontier_distance:g} 格），跳过本轮",
                throttle_duration_sec=5.0,
            )
            return

        # A* 栅格路径：保留作为回退路径与"路径受阻检查"的依据（每轮距离场复用）。
        path_indices = self._planner.plan(
            explored_map,
            robot_grid,
            decision.target,
            inflation_radius_cells=self._inflation_radius_cells,
            use_eight_neighbors=self._use_eight_neighbors,
            treat_unknown_as_obstacle=self._treat_unknown_as_obstacle,
            distance_field=planner_distance_field,
        )
        if path_indices is None:
            # 区分"因膨胀无路"与"真不可达"：用 inflation_radius_cells=0 重试一次。
            # 若重试成功 -> 只是膨胀把路/目标格暂时封住（如窄门、目标格贴着安全边界）。
            # **接受这条零膨胀路径继续发布目标与路径**，不再"只打日志然后 return"：
            # 否则下一轮仍选中同一前沿、同样失败，目标发布永久停止，8s 后 path_follower
            # 判 stale 零速，形成死锁。目标位置本身的安全性由 light_qifa 的候选过滤
            # （dist >= obstacle_safe_distance）保证，且下游有前向刹车与脱困兜底。
            retry_path = self._planner.plan(
                explored_map,
                robot_grid,
                decision.target,
                inflation_radius_cells=0,
                use_eight_neighbors=self._use_eight_neighbors,
                treat_unknown_as_obstacle=self._treat_unknown_as_obstacle,
                distance_field=planner_distance_field,
            )
            if retry_path is not None:
                self.get_logger().info(
                    f"该前沿在膨胀半径 {self._inflation_radius_cells} 格下不可达，"
                    f"改用零膨胀路径发布（已启用前向刹车与脱困兜底）",
                    throttle_duration_sec=10.0,
                )
                # 用零膨胀路径正常走完后面的成功发布流程（重置 streak、判新目标、
                # 累加 _goal_count 等），保证目标发布持续推进。
                path_indices = retry_path
            else:
                # 重试也失败：维持原行为，计入"A* 未能规划"那条 streak（真不可达）。
                self.get_logger().warn("A* 未能规划到该前沿，本轮跳过")
                self._bump_no_target(
                    "连续多轮未产出可达目标，地图已探索完毕", explored_map
                )
                return

        goal_world = self._grid_to_world(decision.target)
        # 用机器人->目标方向作为朝向（robot_world 为本帧 TF 世界坐标，避免再按旧 origin 换算）。
        yaw = math.atan2(goal_world[1] - robot_world[1], goal_world[0] - robot_world[0])

        self._publish_goal(goal_world, yaw)
        # 生成并发布路径：优先生成阿克曼可行轨迹（带每点切线 yaw）；任何失败都回退
        # 原始 A* 折线，并把最终发布的世界坐标路径缓存到 _current_path_world。
        # 返回值用于修复 B 的"同一目标连续失败"计数（None=未尝试，不计失败）。
        traj_success = self._build_and_publish_path(
            explored_map, robot_pose, goal_world, path_indices, traj_distance_field
        )

        # 成功发布目标：重置"无前沿"计数。
        self._no_frontier_streak = 0
        # P1-5：按**世界坐标**（round 到 0.01m）判断是否新目标；同一目标周期性重发不计数。
        is_new_target = (
            self._current_target_world is None
            or round(goal_world[0], 2) != round(self._current_target_world[0], 2)
            or round(goal_world[1], 2) != round(self._current_target_world[1], 2)
        )
        # 目标计数的"有效位移"过滤：只有新目标相对**上一个已发布目标**位移超过
        # goal_count_min_move_m 时才计入 max_goal_count。
        # 原因：前沿会随地图细微变化而小幅"挪动"（实测一场里 200 个"不同目标"只有 45 次
        # 真正到达，其余是 0.1~0.5m 的小步跳变），若无过滤会把目标预算耗尽、在覆盖尚未
        # 收敛时以"达到最大目标数"提前结束（实测覆盖 97.6% 时被截断，没能走到 no_frontier）。
        # 过滤后 max_goal_count 只统计"真正的里程碑式推进"。
        if is_new_target and self._current_target_world is not None:
            hop = math.hypot(
                goal_world[0] - self._current_target_world[0],
                goal_world[1] - self._current_target_world[1],
            )
            if hop < self._goal_count_min_move:
                is_new_target = False
        # 合并字段：_current_target_world 同时作为"当前目标/最近发布目标"的世界坐标。
        self._current_target_world = goal_world
        self._current_target_time = now_sec
        # 修复 B：统计本次轨迹生成结果；连续失败达到阈值时拉黑并清空当前目标，
        # 让下一轮换一个前沿（回退原始 A* 折线的行为保持不变）。
        self._record_traj_result(goal_world, traj_success, now_sec)
        if is_new_target:
            self._goal_count += 1
        if self._goal_count >= self._max_goal_count:
            self._declare_done(
                f"达到最大目标数 {self._max_goal_count}",
                save_suffix="max_goals",
            )

        if len(path_indices) >= 2:
            first = path_indices[1]
            self._previous_direction = (
                float(first[0] - robot_grid[0]),
                float(first[1] - robot_grid[1]),
            )

    # ------------------------------------------------------------- 目标迟滞
    def _clear_current_target(self, reason: str) -> None:
        """清空当前目标（世界坐标 + 发布时间），下一轮进入正常重新选点。"""
        if self._current_target_world is not None:
            self.get_logger().info(
                f"清空当前目标 ({self._current_target_world[0]:.2f}, "
                f"{self._current_target_world[1]:.2f})：{reason}"
            )
        self._current_target_world = None
        self._current_target_time = None
        self._current_path_world = None

    def _keep_current_target(
        self,
        now_sec: float,
        explored_map: np.ndarray,
        robot_grid: Tuple[int, int],
        robot_pose: Tuple[float, float, Optional[float]],
        planner_distance_field: np.ndarray,
        traj_distance_field: np.ndarray,
    ) -> bool:
        """目标迟滞：在有效期内沿用当前目标，返回 True 表示本轮已重发并应直接 return。

        判定顺序：
          1) 已有当前目标，且 now - _current_target_time < target_hysteresis_sec；
          2) 把当前目标世界坐标转成栅格（用**当帧** origin，避免漂移错位）；
          3) target_keep_verify 为 True 时校验：该格仍是自由格且 4 邻域含未知格
             （即仍是前沿候选）；否则视为已不再是前沿，清空后重新选点；
          4) 复用本轮距离场做 plan(..., inflation_radius_cells) 验证可达；失败则清空；
          5) 全部通过 -> 重新发布同一目标/路径（仍走阿克曼轨迹生成，避免与首轮路径
             在折线/平滑轨迹之间来回切换），刷新 _current_target_time，不累加 _goal_count。
        """
        if self._current_target_world is None or self._current_target_time is None:
            return False
        if (now_sec - self._current_target_time) >= self._target_hysteresis_sec:
            return False

        target_grid = self._world_to_grid(self._current_target_world, warn=False)
        if target_grid is None:
            self._clear_current_target("已不在地图范围内")
            return False

        if self._target_keep_verify:
            row, col = target_grid
            # explored_map 语义：-1 未知 / 0 自由 / 1 障碍。只有自由格（0）可能是前沿。
            value = int(explored_map[row, col])
            if value != 0:
                self._clear_current_target(f"所在格不再是自由格（值={value}）")
                return False
            height, width = explored_map.shape
            adjacent_unknown = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width and explored_map[rr, cc] < 0:
                    adjacent_unknown = True
                    break
            if not adjacent_unknown:
                self._clear_current_target("4 邻域已无未知格（不再是前沿）")
                return False

        path_indices = self._planner.plan(
            explored_map,
            robot_grid,
            target_grid,
            inflation_radius_cells=self._inflation_radius_cells,
            use_eight_neighbors=self._use_eight_neighbors,
            treat_unknown_as_obstacle=self._treat_unknown_as_obstacle,
            distance_field=planner_distance_field,
        )
        if path_indices is None:
            self._clear_current_target("当前膨胀半径下 A* 不可达")
            return False

        # 沿用成功：重新发布同一目标/路径（下游 stale 兜底依赖持续刷新）。
        goal_world = self._current_target_world
        robot_world = (robot_pose[0], robot_pose[1])
        yaw = math.atan2(
            goal_world[1] - robot_world[1], goal_world[0] - robot_world[0]
        )
        self._publish_goal(goal_world, yaw)
        # 与首轮一致：仍优先发布阿克曼轨迹，避免路径在折线与平滑轨迹间来回跳变。
        traj_success = self._build_and_publish_path(
            explored_map, robot_pose, goal_world, path_indices, traj_distance_field
        )
        # 刷新发布时间：只要目标仍可达就继续沿用，避免每 target_hysteresis_sec 无条件换目标。
        self._current_target_time = now_sec
        # 修复 B：沿用路径同样统计"连续轨迹失败"，达到阈值就拉黑并清空当前目标
        # （下一轮正常重新选点；迟滞不会再把已拉黑的目标捞回来）。
        self._record_traj_result(goal_world, traj_success, now_sec)
        self._no_frontier_streak = 0

        if len(path_indices) >= 2:
            first = path_indices[1]
            self._previous_direction = (
                float(first[0] - robot_grid[0]),
                float(first[1] - robot_grid[1]),
            )
        return True

    # ------------------------------------------------------------- 完成判定
    def _bump_no_target(self, done_reason: str, explored_map: np.ndarray) -> None:
        """本轮没有产出可达目标：累计 streak，达到阈值即判定完成。
        环境未就绪时不计数：还没产出过任何目标，或已知栅格数太少。"""
        known_cells = int(np.count_nonzero(explored_map != -1))
        if self._goal_count == 0 or known_cells < self._min_known_cells:
            return
        self._no_frontier_streak += 1
        if self._no_frontier_streak >= self._no_frontier_confirm_count:
            self._declare_done(done_reason, save_suffix="no_frontier")

    def _declare_done(self, reason: str, save_suffix: str = "no_frontier") -> None:
        """判定探索完成：发布完成标志，并（只做一次）尝试存图。

        用 _done 守卫，保证只会执行一次；且 _declare_done 之后不再发布 goal/path。
        save_suffix 会拼到存图文件名后（如 _timeout / _max_goals / _no_frontier），
        便于区分完成原因；空字符串表示不加后缀。
        """
        if self._done:
            return
        self._done = True
        self._done_pub.publish(Bool(data=True))
        self.get_logger().info(f"探索完成（{reason}）：停止发布目标与路径")
        if self._map_save_prefix:
            self._save_map(save_suffix)

    def _save_map(self, save_suffix: str = "") -> None:
        """完成时把当前地图保存为 map_server 标准格式。

        自研写出 <prefix>[_<save_suffix>].pgm/.yaml（二进制 P5、negate: 0），
        可被 nav2 map_server 直接加载。存图失败只告警，绝不打断节点。
        """
        if self._last_map_msg is None:
            self.get_logger().warn("探索完成但尚未收到 /map，跳过存图")
            return
        prefix = self._map_save_prefix
        if save_suffix:
            prefix = f"{prefix}_{save_suffix}"
        ok, detail = save_occupancy_grid_map(
            self._last_map_msg,
            prefix,
            int(self.get_parameter("occupied_threshold").value),
        )
        if ok:
            self.get_logger().info(detail)
        else:
            self.get_logger().warn(f"存图失败：{detail}")

    # ------------------------------------------------------------- 坐标转换
    def _lookup_robot_pose_world(
        self,
    ) -> Optional[Tuple[float, float, Optional[float]]]:
        """查询机器人当前位姿 ``(x, y, yaw)``；TF 未就绪时返回 None。

        yaw 从 map->base_link 的四元数解出（阿克曼轨迹必须知道起步朝向）；四元数无效
        （全 0）时 yaw 返回 None，调用方据此退回原始 A* 折线并告警。
        每轮只查一次 TF，避免重复查询带来的抖动。
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame, self._robot_frame, Time()
            )
        except Exception:  # TF 尚未就绪，等下一轮
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = _yaw_from_quaternion_optional(
            float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
        )
        return (float(translation.x), float(translation.y), yaw)

    def _world_to_grid_rc_callable(self, x: float, y: float) -> Tuple[int, int]:
        """把 `_world_to_grid` 包成阿克曼轨迹模块要求的 (x, y) -> (row, col) callable。

        越界返回 (-1, -1)：轨迹模块会按"起点/终点越界"判失败并回退，不会去索引地图。
        """
        grid = self._world_to_grid((float(x), float(y)), warn=False)
        return grid if grid is not None else (-1, -1)

    def _grid_to_world_xy_callable(self, row: int, col: int) -> Tuple[float, float]:
        """把 `_grid_to_world` 包成轨迹模块要求的 (row, col) -> (x, y) callable。"""
        return self._grid_to_world((int(row), int(col)))

    def _world_to_grid(
        self, world_xy: Tuple[float, float], warn: bool = True
    ) -> Optional[Tuple[int, int]]:
        """世界坐标 -> 栅格索引 (row, col)；越界时返回 None。

        warn=False 用于目标迟滞里的旧目标换算（地图 origin 漂移导致临时越界时不必刷告警）。
        注意每次换算都必须使用**当帧** /map 的 origin，绝不能用旧 origin 解释旧行列。
        """
        info = self._grid_msg.info
        col = int(math.floor((world_xy[0] - info.origin.position.x) / info.resolution))
        row = int(math.floor((world_xy[1] - info.origin.position.y) / info.resolution))
        if 0 <= row < info.height and 0 <= col < info.width:
            return (row, col)
        if warn:
            self.get_logger().warn(
                f"坐标 ({world_xy[0]:.2f}, {world_xy[1]:.2f}) 对应的栅格 "
                f"({row}, {col}) 不在地图范围内 ({info.height}x{info.width})"
            )
        return None

    # ------------------------------------------------------------- 目标推进检测
    def _update_stall(self, now_sec: float, robot_world: Tuple[float, float]) -> None:
        """目标推进检测：stall_detect_sec 内位移 < stall_move_eps 即放弃当前目标。

        - 位移 > stall_move_eps：刷新 _last_move_pose / _last_move_time；
        - 超过 stall_detect_sec 仍未有效移动且当前有已发布目标：把该目标世界坐标加入
          临时黑名单（过期时刻 now + target_blacklist_sec），打一条 warn，然后重置计时，
          避免每轮重复告警。
        """
        if self._last_move_pose is None or self._last_move_time is None:
            self._last_move_pose = robot_world
            self._last_move_time = now_sec
            return

        moved = math.hypot(
            robot_world[0] - self._last_move_pose[0],
            robot_world[1] - self._last_move_pose[1],
        )
        if moved > self._stall_move_eps:
            self._last_move_pose = robot_world
            self._last_move_time = now_sec
            return

        stalled_sec = now_sec - self._last_move_time
        if stalled_sec <= self._stall_detect_sec:
            return

        # 当前有目标才需要放弃；没有目标时只重置计时（避免长时间停机后立即误判）。
        # P1-5：直接用已保存的**世界坐标**，不再拿当前 origin 去解释旧栅格行列。
        if self._current_target_world is not None:
            target_world = self._current_target_world
            self._blacklist.append(
                (target_world[0], target_world[1], now_sec + self._target_blacklist_sec)
            )
            self.get_logger().warn(
                f"目标 ({target_world[0]:.2f}, {target_world[1]:.2f}) 在 "
                f"{stalled_sec:.1f}s 内未推进（机器人未移动），"
                f"放弃并加入黑名单 {self._target_blacklist_sec:.0f}s"
            )
            # 必须清空当前目标：否则下一轮目标迟滞会继续沿用这个已被拉黑的点。
            self._current_target_world = None
            self._current_target_time = None
            self._current_path_world = None
        self._last_move_pose = robot_world
        self._last_move_time = now_sec

    @staticmethod
    def _stamp_exclude_circle(
        exclude: np.ndarray,
        info,
        world_x: float,
        world_y: float,
        radius_cells: float,
        snap_to_cell: bool = False,
    ) -> None:
        """把世界坐标 ``(world_x, world_y)`` 周围 ``radius_cells`` 格的圆标为排除。

        直接就地修改 ``exclude``（True=不作为候选）；圆心按浮点栅格坐标计算，边界裁剪。

        :param snap_to_cell: True 时把圆心吸附到最近的整数栅格（像素中心），用于**小半径**
            （如顽固目标 2 格）场景：目标世界坐标通常在半个像素处，不吸附会让 "±2 格"
            在两侧分别落在 1.5 / 2.5 格，严格 2 格半径就覆盖不到一侧。吸附后半径 2 格
            恰好覆盖索引 ±2，避免"同一目标因离散化误差漏网"。
        """
        height, width = int(info.height), int(info.width)
        resolution = max(float(info.resolution), 1e-9)
        # 圆心在地图坐标系下的浮点栅格坐标（不做 floor，保证圆位置精确）。
        center_col = (world_x - info.origin.position.x) / resolution - 0.5
        center_row = (world_y - info.origin.position.y) / resolution - 0.5
        if snap_to_cell:
            # 最近整数栅格（.5 向上取整），保证圆心落在真实像素中心上。
            center_col = float(math.floor(center_col + 0.5))
            center_row = float(math.floor(center_row + 0.5))
        radius_int = int(math.ceil(radius_cells))
        # 边界裁剪：包围盒与地图求交后只处理相交部分。
        r0 = max(0, int(math.floor(center_row - radius_int)))
        r1 = min(height, int(math.ceil(center_row + radius_int)) + 1)
        c0 = max(0, int(math.floor(center_col - radius_int)))
        c1 = min(width, int(math.ceil(center_col + radius_int)) + 1)
        if r0 >= r1 or c0 >= c1:
            return
        rows = np.arange(r0, r1, dtype=np.float32) - float(center_row)
        cols = np.arange(c0, c1, dtype=np.float32) - float(center_col)
        dist2 = rows[:, None] ** 2 + cols[None, :] ** 2
        exclude[r0:r1, c0:c1] |= dist2 <= float(radius_cells) ** 2

    def _build_exclude_mask(self, now_sec: float) -> Optional[np.ndarray]:
        """按未过期的目标黑名单 + 顽固目标永久排除列表构造前沿排除掩码（True=排除）。

        - 临时黑名单：世界坐标为圆心、``target_blacklist_radius`` 为半径（米→格）；
        - 顽固目标 ``_hard_targets``：半径固定 ``_hard_target_radius_cells`` 格（约 2 格），
          用于覆盖坐标离散化误差，避免同一目标换个整数栅格又"复活"。
          该列表**不随黑名单到期消失**，从根上治"180s 到期后重复选中→目标空转"。

        每轮先清理过期黑名单；两类列表都为空时返回 None，保证 decide_frontier 的
        ``exclude_mask=None`` 分支行为完全不变。
        """
        # 清理过期黑名单（_hard_targets 是永久列表，不在这里清理）。
        self._blacklist = [item for item in self._blacklist if item[2] > now_sec]
        if not self._blacklist and not self._hard_targets:
            return None

        info = self._grid_msg.info
        height, width = int(info.height), int(info.width)
        exclude = np.zeros((height, width), dtype=bool)

        # 临时黑名单：半径按米换算成格。
        resolution = max(float(info.resolution), 1e-9)
        blacklist_radius_cells = self._target_blacklist_radius / resolution
        for world_x, world_y, _expire in self._blacklist:
            self._stamp_exclude_circle(
                exclude, info, world_x, world_y, blacklist_radius_cells
            )

        # 顽固目标：直接用格为单位的固定半径（2 格），并吸附到最近整数栅格，
        # 避免"目标坐标在半个像素处"导致 ±2 格在一侧覆盖不到。
        for world_x, world_y in self._hard_targets:
            self._stamp_exclude_circle(
                exclude,
                info,
                world_x,
                world_y,
                self._hard_target_radius_cells,
                snap_to_cell=True,
            )

        if not np.any(exclude):
            return None
        return exclude

    # ------------------------------------------------------------- 探索边界过滤
    @staticmethod
    def _parse_exploration_bounds(
        value,
    ) -> Optional[Tuple[float, float, float, float]]:
        """解析 ``exploration_bounds``：``"xmin,xmax,ymin,ymax"``（米，世界坐标）。

        空字符串返回 None（不启用）；格式非法或上下界颠倒也返回 None，由调用方告警。
        """
        text = "" if value is None else str(value).strip()
        if not text:
            return None
        parts = [item.strip() for item in text.split(",")]
        if len(parts) != 4:
            return None
        try:
            xmin, xmax, ymin, ymax = (float(item) for item in parts)
        except ValueError:
            return None
        if not (xmin < xmax and ymin < ymax):
            return None
        return (xmin, xmax, ymin, ymax)

    def _target_within_bounds(self, grid_index: Tuple[int, int]) -> bool:
        """前沿目标（栅格）的世界坐标是否落在 exploration_bounds 内。"""
        if self._exploration_bounds is None:
            return True
        world_x, world_y = self._grid_to_world(grid_index)
        xmin, xmax, ymin, ymax = self._exploration_bounds
        return (xmin <= world_x <= xmax) and (ymin <= world_y <= ymax)

    def _build_bounds_exclude_mask(self) -> Optional[np.ndarray]:
        """把 exploration_bounds 之外的所有栅格标记为排除（True=不作为候选）。

        用栅格中心的世界坐标判断，与 ``_grid_to_world`` 完全同口径。整张地图都在
        边界内时返回 None（与不启用时行为一致）。
        """
        if self._exploration_bounds is None:
            return None
        info = self._grid_msg.info
        height, width = int(info.height), int(info.width)
        resolution = max(float(info.resolution), 1e-9)
        xmin, xmax, ymin, ymax = self._exploration_bounds
        # 栅格中心的世界坐标（与 _grid_to_world 一致：col + 0.5 / row + 0.5）。
        xs = info.origin.position.x + (np.arange(width, dtype=np.float64) + 0.5) * resolution
        ys = info.origin.position.y + (np.arange(height, dtype=np.float64) + 0.5) * resolution
        outside_x = (xs < xmin) | (xs > xmax)
        outside_y = (ys < ymin) | (ys > ymax)
        exclude = outside_y[:, None] | outside_x[None, :]
        if not np.any(exclude):
            return None
        return exclude

    def _apply_exploration_bounds(
        self,
        decision,
        explored_map: np.ndarray,
        robot_grid: Tuple[int, int],
        exclude_mask: Optional[np.ndarray],
        distance_field: np.ndarray,
    ):
        """修复 C：过滤掉建筑外壳之外的前沿，避免把时间耗在建筑外围空地上。

        策略（不改变未配置时的行为）：
          1) ``exploration_bounds`` 为空 -> 原样返回；
          2) 选出的最佳候选就在边界内 -> 原样返回（不做额外开销）；
          3) 最佳候选在边界外 -> 把"边界外"并进 exclude_mask 重新决策一次：
             - 新结果非空：返回边界内的候选；
             - 新结果为空：**退回原候选**（宁可在边界外探索，也绝不把探索彻底卡死），
               并打一条 10s 节流 warn。
        """
        if self._exploration_bounds is None:
            return decision
        if decision.target is None:
            return decision
        if self._target_within_bounds(decision.target):
            return decision

        bounds_mask = self._build_bounds_exclude_mask()
        if bounds_mask is None:
            return decision
        combined = (
            bounds_mask
            if exclude_mask is None
            else (np.asarray(exclude_mask, dtype=bool) | bounds_mask)
        )
        filtered = self._decide_frontier(
            explored_map,
            robot_grid,
            previous_direction=self._previous_direction,
            config=self._heuristic_config,
            exclude_mask=combined,
            distance_to_obstacle=distance_field,
        )
        if filtered.target is not None:
            return filtered

        # 过滤后为空：不过滤，返回原候选，避免探索彻底卡死。
        self.get_logger().warn(
            "探索边界过滤后已无候选前沿，本轮退回边界外候选（避免探索卡死）",
            throttle_duration_sec=10.0,
        )
        return decision

    def _grid_to_world(self, grid_index: Tuple[int, int]) -> Tuple[float, float]:
        info = self._grid_msg.info
        row, col = grid_index
        return (
            info.origin.position.x + (col + 0.5) * info.resolution,
            info.origin.position.y + (row + 0.5) * info.resolution,
        )

    # ------------------------------------------------------------- 路径受阻检查
    def _truncate_path_preview(
        self, path_world: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """按累计距离截取路径前 ``path_block_check_m`` 米（含跨越阈值的那一点）。"""
        if not path_world:
            return []
        limit = self._path_block_check_m
        if limit <= 0.0 or len(path_world) <= 1:
            return [path_world[0]] if path_world else []
        preview = [path_world[0]]
        accumulated = 0.0
        for i in range(1, len(path_world)):
            accumulated += math.hypot(
                path_world[i][0] - path_world[i - 1][0],
                path_world[i][1] - path_world[i - 1][1],
            )
            preview.append(path_world[i])
            if accumulated >= limit:
                break
        return preview

    def _replan_if_path_blocked(self, explored_map: np.ndarray) -> bool:
        """路径前 path_block_check_m 米内出现新障碍时立即清空目标并返回 True。

        必须放在目标迟滞判断**之前**：否则新观测障碍出现后，迟滞仍会沿用旧目标并
        重发同一条已不可走的路径，直到迟滞过期才重新选点。
        - 任一点是障碍（1）-> warn（2s 节流）+ 清空当前目标/路径，本轮立即重规划；
        - 任一点是未知（-1）-> info（5s 节流），不重规划（本轮未知格按障碍处理，
          正常不会出现在已发布路径上）。
        """
        if not self._current_path_world:
            return False
        preview = self._truncate_path_preview(self._current_path_world)
        has_obstacle = False
        has_unknown = False
        for world_x, world_y in preview:
            grid = self._world_to_grid((world_x, world_y), warn=False)
            if grid is None:
                # 地图还没长到这一点：无法判定，按未观测区域处理（不阻断）。
                continue
            value = int(explored_map[grid[0], grid[1]])
            if value == 1:
                has_obstacle = True
                break
            if value == -1:
                has_unknown = True

        if has_obstacle:
            self.get_logger().warn(
                f"路径前 {self._path_block_check_m:g}m 被新观测障碍占据，"
                f"立即重规划（跳过迟滞）",
                throttle_duration_sec=2.0,
            )
            self._current_target_world = None
            self._current_target_time = None
            self._current_path_world = None
            return True
        if has_unknown:
            self.get_logger().info(
                f"路径前 {self._path_block_check_m:g}m 进入未观测区域",
                throttle_duration_sec=5.0,
            )
        return False

    # ------------------------------------------------------------- 轨迹生成与发布
    def _build_and_publish_path(
        self,
        explored_map: np.ndarray,
        robot_pose: Tuple[float, float, Optional[float]],
        goal_world: Tuple[float, float],
        path_indices: List,
        distance_field: np.ndarray,
    ) -> Optional[bool]:
        """发布路径：优先阿克曼可行轨迹，失败/关闭时回退原始 A* 折线。

        - 成功：Path 每个 pose 带切线 yaw（orientation.z=sin(yaw/2), w=cos(yaw/2)），
          下游 path_follower 可直接用真实朝向做曲率限速；
        - 失败：退回 A* 折线（不含朝向），warn 节流 5s，保证探索链路不中断；
        - 两条路径都会缓存到 _current_path_world，供下一轮"路径受阻检查"。

        :return: 阿克曼轨迹是否生成成功。True=成功；False=尝试了但失败（回退折线）；
            None=未尝试（enable_ackermann_traj 关闭 / TF 解不出 yaw）。修复 B 用这个
            返回值统计"同一目标连续失败次数"，不会把"未尝试"误算成失败。
        """
        raw_points = [self._grid_to_world(idx) for idx in path_indices]
        published_points = raw_points
        published_yaws = None
        traj_success: Optional[bool] = None

        if self._enable_ackermann_traj:
            robot_yaw = robot_pose[2]
            if robot_yaw is None:
                # TF 只能给出 (x,y) 而解不出朝向：不能猜 yaw，直接退回原始折线。
                self.get_logger().warn(
                    "未获得机器人朝向 yaw，无法生成阿克曼轨迹，回退原始 A* 折线",
                    throttle_duration_sec=5.0,
                )
            else:
                trajectory = None
                try:
                    trajectory_config = self._AckermannTrajConfig(
                        wheelbase_m=self._wheelbase_m,
                        max_steer_rad=self._max_steer_rad,
                        min_turn_radius_m=self._min_turn_radius_m,
                        # res_m 取当帧地图分辨率（轨迹模块的平移/碰撞采样依赖它）。
                        res_m=float(self._grid_msg.info.resolution),
                        # traj_safety_cells 不新增参数：直接复用 inflation_radius_cells。
                        safety_cells=float(self._inflation_radius_cells),
                        ds_m=self._traj_ds_m,
                        v_max=self._traj_v_max,
                        v_min=self._traj_v_min,
                        a_lat_max=self._traj_a_lat_max,
                        a_long_max=self._traj_a_long_max,
                    )
                    trajectory = self._plan_ackermann_trajectory(
                        explored_map,
                        grid_to_world_xy=self._grid_to_world_xy_callable,
                        world_to_grid_rc=self._world_to_grid_rc_callable,
                        start_pose_world=(
                            float(robot_pose[0]),
                            float(robot_pose[1]),
                            float(robot_yaw),
                        ),
                        goal_world_xy=(float(goal_world[0]), float(goal_world[1])),
                        config=trajectory_config,
                        planner=self._planner,
                        # 复用本轮唯一一份距离场（模块内部 A* 也不再计算）。
                        distance_field=distance_field,
                    )
                except Exception as exc:  # 轨迹模块任何异常都不能打断探索链路
                    self.get_logger().warn(
                        f"阿克曼轨迹生成异常：{exc}，回退原始 A* 折线",
                        throttle_duration_sec=5.0,
                    )
                    trajectory = None

                # 修复 B：记录本次轨迹生成结果（异常/不可行都算失败）。
                traj_success = bool(
                    trajectory is not None
                    and trajectory.feasible
                    and len(trajectory.x) >= 2
                )

                if traj_success:
                    published_points = [
                        (float(x), float(y))
                        for x, y in zip(trajectory.x, trajectory.y)
                    ]
                    published_yaws = [float(value) for value in trajectory.yaw]
                    length_m = (
                        float(trajectory.s[-1]) if len(trajectory.s) > 0 else 0.0
                    )
                    has_reverse = bool(np.any(trajectory.v < -1e-6))
                    self.get_logger().info(
                        f"已发布阿克曼可行轨迹（N={len(published_points)}, "
                        f"长={length_m:.2f}m, 含倒车={has_reverse}）",
                        throttle_duration_sec=10.0,
                    )
                else:
                    reason = (
                        trajectory.reason
                        if trajectory is not None and trajectory.reason
                        else "轨迹不可行或点数不足"
                    )
                    self.get_logger().warn(
                        f"阿克曼轨迹不可用（{reason}），回退原始 A* 折线",
                        throttle_duration_sec=5.0,
                    )

        self._publish_path(published_points, published_yaws)
        # 两条路径都同步缓存，供下一轮路径受阻检查使用。
        self._current_path_world = list(published_points)
        return traj_success

    # ------------------------------------------------------------- 连续轨迹失败拉黑
    def _record_traj_result(
        self,
        goal_world: Tuple[float, float],
        traj_success: Optional[bool],
        now_sec: float,
    ) -> None:
        """修复 B：统计同一目标连续生成阿克曼轨迹的失败次数，达到阈值就拉黑换目标。

        - ``traj_success is None``：未尝试（关闭轨迹生成 / 解不出 yaw），不计失败；
        - 成功：该目标连续计数与累计失败计数都清零并删除（说明它其实可达）；
        - 失败：连续计数 +1；达到 ``traj_fail_blacklist_count`` 时加入现有 ``_blacklist``
          （复用 ``target_blacklist_sec`` 时长），清空当前目标，让下一轮换一个前沿。
          同时**跨黑名单周期**给 ``_hard_target_fail_totals`` 累计 +1；累计达到
          ``hard_target_fail_count`` 时把该坐标加入 ``_hard_targets``（本次运行内永久排除，
          不再随 180s 黑名单到期而复活，治"黑名单到期又重复选中"的目标空转）。
        - 清理：每次成功发布/拉黑后，删除不在"当前目标"上的**连续**计数 key，避免字典
          无限增长；``_hard_target_fail_totals`` 按设计跨周期保留，只在成功时清零。

        注意：**回退原始 A* 折线仍然保留**（本轮不取消回退），只是"连续失败多次
        就不再死磕这个目标"。被拉黑的目标在本轮已经发布过，下一轮由黑名单掩码过滤。
        """
        if traj_success is None:
            return

        key = (round(float(goal_world[0]), 2), round(float(goal_world[1]), 2))

        if traj_success:
            # 成功生成：连续计数清零，累计失败计数也清零（目标其实可达）。
            self._traj_fail_counts.pop(key, None)
            self._hard_target_fail_totals.pop(key, None)
        else:
            count = int(self._traj_fail_counts.get(key, 0)) + 1
            self._traj_fail_counts[key] = count
            if count >= self._traj_fail_blacklist_count:
                # 达到阈值：拉黑（复用现有黑名单机制与时长），清空当前目标。
                self._blacklist.append(
                    (
                        float(goal_world[0]),
                        float(goal_world[1]),
                        float(now_sec) + self._target_blacklist_sec,
                    )
                )
                self.get_logger().warn(
                    f"目标 ({goal_world[0]:.2f}, {goal_world[1]:.2f}) 连续 {count} 次"
                    f"无法生成阿克曼轨迹，拉黑 {self._target_blacklist_sec:.0f}s 后换目标",
                    throttle_duration_sec=2.0,
                )
                # 修复 2：跨黑名单周期的累计失败计数（每次"被拉黑"算一次）。
                # 黑名单只有 target_blacklist_sec 时长，到期后同一目标会被反复选中；
                # 累计到 hard_target_fail_count 次即永久排除。
                total = int(self._hard_target_fail_totals.get(key, 0)) + 1
                self._hard_target_fail_totals[key] = total
                if (
                    total >= self._hard_target_fail_count
                    and key not in self._hard_targets
                ):
                    self._hard_targets.append(key)
                    # 上限保护：超过 200 个时删最早的一个（pop(0)），避免掩码构造变慢。
                    while len(self._hard_targets) > self._hard_targets_limit:
                        self._hard_targets.pop(0)
                    self.get_logger().warn(
                        f"目标 ({goal_world[0]:.2f}, {goal_world[1]:.2f}) 累计 {total} 次"
                        f"生成轨迹失败，本次运行内永久排除",
                        throttle_duration_sec=2.0,
                    )
                self._current_target_world = None
                self._current_target_time = None
                self._current_path_world = None
                # 删除该 key（连续计数已处理完毕；累计计数按设计保留）。
                self._traj_fail_counts.pop(key, None)

        # 清理过期 key：只保留"当前目标"的计数。
        if self._current_target_world is not None:
            current_key = (
                round(float(self._current_target_world[0]), 2),
                round(float(self._current_target_world[1]), 2),
            )
        else:
            current_key = None
        for stale_key in list(self._traj_fail_counts.keys()):
            if stale_key != current_key:
                del self._traj_fail_counts[stale_key]

    def _to_explored_map(self, msg: OccupancyGrid) -> np.ndarray:
        """OccupancyGrid -> int8 栅格：-1 未知 / 0 空地 / 1 障碍，形状 (row, col)。"""
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        explored = np.zeros(data.shape, dtype=np.int8)
        explored[data < 0] = -1
        explored[data >= self._occupied_threshold] = 1
        return explored

    # ------------------------------------------------------------- 话题发布
    def _publish_goal(self, world_xy: Tuple[float, float], yaw: float) -> None:
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self._global_frame
        goal.pose.position.x = float(world_xy[0])
        goal.pose.position.y = float(world_xy[1])
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self._goal_pub.publish(goal)

    def _publish_path(
        self,
        world_points: List[Tuple[float, float]],
        yaws: Optional[List[float]] = None,
    ) -> None:
        """发布世界坐标路径。

        ``yaws`` 非空时（阿克曼轨迹）每个 pose 写入切线朝向：
        ``orientation.z = sin(yaw/2)``、``orientation.w = cos(yaw/2)``；
        为空时（原始 A* 折线回退）沿用旧发布方式（w=1），下游 follower 会把
        "全为单位四元数"识别为无朝向，并用相邻点几何切线补算。
        """
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self._global_frame
        for index, (x, y) in enumerate(world_points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            if yaws is not None and index < len(yaws):
                yaw = float(yaws[index])
                pose.pose.orientation.z = math.sin(yaw / 2.0)
                pose.pose.orientation.w = math.cos(yaw / 2.0)
            else:
                pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._path_pub.publish(path)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = FrontierExplorerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
