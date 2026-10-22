from __future__ import annotations

"""路径跟随控制节点：把探索器的目标点/路径变成 /cmd_vel 速度指令。

输入：
  - nav_msgs/msg/Path（默认话题 exploration_path）：机器人当前位置到前沿目标的 A* 路径，
    map 坐标系，poses[0] 靠近机器人、末尾接近目标。
  - geometry_msgs/msg/PoseStamped（默认话题 exploration_goal）：当前前沿目标点。
  - std_msgs/msg/Bool（默认话题 exploration_done）：探索完成标志，收到 True 后立即停车。
  - TF: <global_frame> -> <robot_frame>（默认 map -> base_link），用于取机器人当前位姿。

输出：
  - geometry_msgs/msg/Twist（默认话题 cmd_vel）：供 gazebo_ros_ackermann_drive 插件使用。

参数（名称 | 默认值 | 含义）：
  - path_topic           | "exploration_path" | 上游路径话题
  - goal_topic           | "exploration_goal" | 上游目标点话题
  - done_topic           | "exploration_done" | 上游探索完成标志话题（latched）
  - cmd_vel_topic        | "cmd_vel"          | 速度指令发布话题
  - global_frame         | "map"              | TF 全局坐标系
  - robot_frame          | "base_link"        | TF 机器人坐标系
  - control_rate_hz      | 10.0               | 控制循环频率（Hz）
  - lookahead_dist       | 0.4                | 固定前视点（carrot）距离（米）；仅在 lookahead_adaptive=False 时生效
  - lookahead_adaptive   | True               | 是否启用自适应前视距离
  - lookahead_min        | 0.4                | 自适应前视距离下限（米）
  - lookahead_max        | 1.0                | 自适应前视距离上限（米）
  - goal_tolerance       | 0.25               | 到达判定半径（米）
  - slowdown_dist        | 0.8                | 接近最终目标的减速段距离（米）
  - max_linear_vel       | 0.3                | 最大线速度（m/s）
  - traj_v_max           | 0.30               | 曲率限速上限（m/s），与上游阿克曼轨迹参数一致
  - traj_v_min           | 0.08               | 曲率限速下限（m/s），保证弯道仍有转向能力
  - a_lat_max            | 0.60               | 横向加速度上限（m/s²），v_curv=sqrt(a_lat_max/κ)
  - min_linear_vel       | 0.08               | 最小线速度（m/s）；防止阿克曼底盘在目标位于正后方（|err|→pi）时线速度归零而无法转向
  - max_angular_vel      | 0.6                | 最大角速度（rad/s）；默认 0.6 已收进真机可达范围，再配合曲率钳制形成双重保护
  - min_turn_radius_m    | 0.35               | **可达曲率钳制使用的最小转弯半径（m）**；必
                          |                     须与上游阿克曼轨迹模块的 min_turn_radius_m 完全一致，
                          |                     否则上游生成的可行轨迹会被下游钳得更紧/放得更松。
                          |                     <=0 时回退到 tan(max_steer_rad)/wheelbase_m（旧行为）
  - wheelbase_m          | 0.144              | 真机轴距（m）；仅在 min_turn_radius_m<=0 的回退
                          |                     路径（或其它兼容逻辑）中使用
  - max_steer_rad        | 0.489              | 真机前轮最大转角（rad，约 28°）；仅用于回退
                          |                     路径，不再直接参与默认曲率钳制
  - curvature_guard_enable | True             | 是否启用"可达曲率钳制"（把角速度限制在当前线速度物理可实现的曲率内）
  - k_angular            | 1.5                | 航向比例增益
  - stale_timeout_sec    | 8.0                | 上游 path/goal 超过该秒数未更新即视为失效、发零 Twist
  - reached_watchdog_sec | 8.0                | reached 后无新目标超过该秒数时打印一次告警（仅告警，不恢复行进）
  - scan_topic           | "scan"             | 前向脱困状态机所用的激光雷达话题（LaserScan）
  - enable_safety_brake  | True               | 是否启用前向脱困状态机
  - brake_distance       | 0.30               | 脱困触发距离（m）：正前方小于该值即进入脱困
  - brake_clear_distance | 0.55               | 脱困退出距离（m）：正前方大于该值才算脱离
                          （与 brake_distance 构成 0.30/0.55 回差，避免一进入就退出）
  - brake_half_angle_deg | 35.0               | 前向扇区半角（度）
  - escape_speed         | 0.15               | 脱困倒车速度（m/s）
  - escape_forward_speed | 0.10               | 三点掉头"前进相"速度（m/s）
  - escape_angular       | 0.35               | 脱困转向角速度（rad/s），朝空间更大的一侧转。
                          |                     取 0.35 是因为插件按 steer=atan(L*w/v) 反解转角：
                          |                     w=0.35、v=0.15 时 steer≈0.44 < max_steer 0.6 可达；
                          |                     若取 0.6 则 steer≈0.67 被截断，转向反而打不满。
  - shuttle_sec          | 2.5                | 三点掉头单个相位（倒车/前进）的持续时间（s）。
                          |                     取 2.5 是因为实测转向执行器滞后严重（P=0.1，2s 仅到
                          |                     目标的约 37%），0.8s 相位根本来不及收敛，yaw 无法累积。
  - escape_timeout_sec   | 10.0               | 单次脱困最长时间（s）
  - scan_timeout_sec     | 1.0                | /scan 失效阈值（s）：超过该秒数未收到新帧即视为没有 scan 数据、不脱困
  - stuck_detect_sec     | 3.0                | 卡死检测时间窗口（s）
  - stuck_move_eps       | 0.05               | 卡死位移阈值（m）
  - reverse_speed        | 0.12               | 倒车自救速度（m/s）
  - reverse_sec          | 1.0                | 倒车自救持续时间（s）
  - reverse_half_angle_deg | 30.0             | "正后方"锥形半角（度）：只统计 |angle| > 180°-该值 的扇区（默认正后方 ±30°）
  - reverse_brake_distance | 0.25             | 倒车后向安全距离（m）：正后方锥形最小测距小于该值则取消本次倒车
  - reverse_angular      | 0.4                | 倒车时的角速度（rad/s），让倒车顺便改变朝向，避免倒完仍朝同一面墙
  - enable_stuck_recovery| True               | 是否启用卡死检测 + 倒车自救
  - stuck_recover_cooldown_sec | 5.0          | 两次倒车自救之间的最小间隔（s）

控制策略：
  - 前视点（carrot）：先找离机器人最近的路径索引 i0，再沿**累计弧长**取
    s[i0] + lookahead 对应的点；剩余弧长不足 lookahead 时取最后一个点。无路径但收到
    目标点时直接用目标当前视点。（用弧长而不是欧氏距离，弯道上不会提前切角。）
  - 曲率限速：从当前前视索引往前取约 0.5m 路径段，用每点朝向差估计局部曲率
    kappa = |wrap(yaw[j]-yaw[i])| / ds，v_curv = sqrt(a_lat_max/kappa)，
    再令 linear_x = min(linear_x, clamp(v_curv, traj_v_min, traj_v_max))。
    倒车段不经过 Path 传递，仍由下方脱困/倒车自救状态机负责。
  - 前视距离 lookahead：lookahead_adaptive=True 时取
    lookahead = clamp(0.6 * 到最终目标的距离, lookahead_min, lookahead_max)，
    并且**不超过到路径末点（最终目标）的距离**——远处目标用大前视（曲率需求小、更稳），
    接近目标时自动缩小；lookahead_adaptive=False 时用固定 lookahead_dist。
  - 航向误差 err = atan2(dy, dx) - yaw（归一化到 [-pi, pi]）。
  - angular.z = clamp(k_angular * err, ±max_angular_vel)。
  - linear.x = clamp(max_linear_vel * max(0, 1 - |err|/pi), 下限 min_linear_vel)：航向偏得越多
    线速度越小，但用 min_linear_vel 兜底，保证阿克曼底盘在目标位于正后方（|err|→pi）时仍有
    线速度以产生转向，避免卡死。
  - 再按到最终目标的距离限速（减速段）：scale = min(1.0, target_dist / slowdown_dist)。
  - 最后做**可达曲率钳制**（curvature_guard_enable=True）：
    |angular.z| <= |linear.x| * kappa_max，其中 **kappa_max = 1/min_turn_radius_m**
    （min_turn_radius_m<=0 时才回退为 tan(max_steer_rad)/wheelbase_m）。
    上下游必须使用同一个 min_turn_radius_m：上游阿克曼轨迹按 0.35m 生成的弧线，
    如果下游按 L/tan(max_steer)=0.271m 反算更紧的曲率上限，刚够用的轨迹会被控制器
    额外钳掉角速度 -> 跟不上、横向误差不收敛。反之若下游更松，则可能给出物理不可达
    的曲率，转角被 max_steer 截断 -> 满舵绕圈。因此必须在发布前用与上游一致的口径钳制，
    把角速度限制在"当前线速度下物理可实现的曲率"内（低速时 w_limit 也小，这是阿克曼的
    真实约束；由 min_linear_vel 保证低速时仍有可用的转向能力）。

安全兜底：
  - 从未收到过 path/goal，或最近一帧超过 stale_timeout_sec 秒未更新 -> 发零 Twist。
    （上游无 scipy 时单轮 decide_frontier 可达 2.4~4.6s，默认 8.0s 避免周期性误判失效。）
  - 到达目标（距离 <= goal_tolerance）-> 发零 Twist 并进入 reached 状态，直到上游给出
    一个与上次到达点不同的新目标、且当前离该目标已超过 goal_tolerance 时才恢复行进。
  - reached 后若上游长时间（reached_watchdog_sec）未换新目标，仅打印一次告警，行为
    仍是保持停车（该前沿可能不可达，或探索已结束）。
  - 收到探索完成标志（exploration_done=True）-> 立即发零 Twist 并永久停车（不再恢复行进）。
  - 前向脱困状态机（带回差）：订阅 LaserScan，取车体正前方 ±brake_half_angle_deg 扇区内
    有效测距的最小值 front_range。若未处于脱困、启用刹车、scan 新鲜且
    front_range < brake_distance（0.30m），立即进入脱困：
      * 比较激光左侧扇区（+brake_half_angle .. +90°）与右侧扇区（-90° .. -brake_half_angle）
        的最小有效测距，朝**空间更大**的一侧转向（左=+1，右=-1；两侧都无数据取 +1）；
      * 两相位三点掉头（shuttle turn）：_escape_phase 在 "reverse" 与 "forward" 之间
        交替，每个相位持续 shuttle_sec（或按可用空间提前切换）：
          - reverse 相：发 Twist(-escape_speed, +dir*escape_angular)；若正后方窄锥形被
            判为过近（_rear_is_blocked，默认仅正后方 ±30°），立即切到 forward 相；
          - forward 相：发 Twist(+escape_forward_speed, -dir*escape_angular)，转向符号
            取反，使倒车+左转与前进+右转都让 yaw 朝**同一方向**累积（三点掉头）；
            若正前方 < brake_distance，立即切回 reverse 相；
          - 相位到达 _escape_phase_end_ts 时也切换一次，保证两相位交替，不会长期吊在
            一个相位上；
      * 脱困期间**不走正常跟线**，也不再"朝墙爬行"（旧逻辑的刹车限幅已删除）；
        脱困每周期都必须输出非零指令，绝不再因"后向被判定 blocked"而发零速死锁。
    退出条件：front_range > brake_clear_distance（0.55m，与 0.30m 构成回差）或
    (now - 起始时刻) > escape_timeout_sec；退出后本周期继续正常跟线。进入/退出各只打
    一条日志，不做 throttle 反复刷屏。
    后向安全：只统计**正后方 ±reverse_half_angle_deg 锥形**（默认 |angle| > 150°），
    不再把侧方墙面/柱子误判成车尾障碍。优先级：done > 倒车自救 _reversing > 脱困
    _escaping > stale > reached > 跟线；进入 _reversing 时会清掉 _escaping，二者不互相
    打架。
    /scan 失效保护：最近一帧 scan 距现在超过 scan_timeout_sec 时视为"没有 scan 数据"，
    不进入脱困（按原逻辑走），避免 scan 停在一帧贴近障碍的数据后永久停。
    从未收到过 scan（如 /scan 没起）时不脱困，按原逻辑走，避免因缺数据永久停。
  - 卡死检测 + 倒车自救：若 stuck_detect_sec 内机器人位移 < stuck_move_eps 且期间指令速度
    非零（"想走走不动"），判定卡死并倒车 reverse_sec（linear.x=-reverse_speed，
    angular.z=reverse_angular，一边倒一边转向）。两次触发之间至少间隔
    stuck_recover_cooldown_sec。倒车前/倒车中做后向安全校验：/scan 正后方窄锥形
    （|angle| > 180°-reverse_half_angle_deg）最小有效测距 < reverse_brake_distance 时
    取消本次倒车（避免倒进墙里）。done 状态下永远
    优先零速停车；reached 状态下（本来零速）不参与卡死判定。
  - 节点销毁时补发一次零 Twist。
"""

import math
from collections import deque
from typing import Any, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener


def _as_bool(value: Any) -> bool:
    """把布尔参数读成 bool：兼容 launch 以字符串（"True"/"False"）覆盖的情形。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """从四元数提取绕 Z 轴的偏航角（rad），避免额外依赖 tf_transformations。"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle: float) -> float:
    """把角度归一化到 [-pi, pi]。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ---------------------------------------------------------------------------
# 纯几何工具（不依赖 ROS，便于离线自检）
# ---------------------------------------------------------------------------


def path_geometry_from_poses(poses):
    """从 Path.poses 提取世界坐标点与可用朝向。

    返回 ``(points, yaws)``：
      * ``points``：``[(x, y), ...]``；
      * ``yaws``：每点朝向列表；若整条路径的四元数**全为 0**（回退路径没带朝向）或
        **全为单位四元数**（旧发布方式把 w 写成 1，同样没有有效朝向），则返回 None，
        由 ``compute_path_arc_lengths`` 用相邻点几何切线补算。

    这里只做 duck-typing（读取 pose.position / pose.orientation），不依赖具体消息类型，
    离线脚本可用 SimpleNamespace 构造假 Path 直接测试。
    """
    points = []
    quat_yaws = []
    all_zero = True
    all_identity = True
    for pose_stamped in poses:
        position = pose_stamped.pose.position
        orientation = pose_stamped.pose.orientation
        points.append((float(position.x), float(position.y)))
        quat_yaws.append(
            _yaw_from_quaternion(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
        )
        if (
            abs(orientation.x) > 1e-12
            or abs(orientation.y) > 1e-12
            or abs(orientation.z) > 1e-12
            or abs(orientation.w) > 1e-12
        ):
            all_zero = False
        if (
            abs(orientation.x) > 1e-12
            or abs(orientation.y) > 1e-12
            or abs(orientation.z) > 1e-12
            or abs(orientation.w - 1.0) > 1e-12
        ):
            all_identity = False
    use_orientation = bool(points) and not (all_zero or all_identity)
    return points, (quat_yaws if use_orientation else None)


def compute_path_arc_lengths(points, yaws=None):
    """计算路径的累计弧长数组 ``s`` 与每点朝向 ``yaw``（均为与 points 等长的列表）。

    ``yaws`` 为 None 时（回退 A* 折线没有带朝向）用相邻点几何切线补算：
      * 第 i 点取 i -> i+1 的方向（出射切线），最后一点沿用倒数第二点；
      * 这样折角处的航向变化全部落在拐点对应的弧长区间上，曲率估计不会被抹平。

    有朝向时直接使用上游给的 yaw（阿克曼轨迹发布的就是每点真实切线朝向）。
    """
    n = len(points)
    if n == 0:
        return [], []
    s = [0.0] * n
    for i in range(1, n):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        s[i] = s[i - 1] + math.hypot(dx, dy)
    if yaws is not None and len(yaws) == n:
        return s, [float(value) for value in yaws]

    yaw = [0.0] * n
    for i in range(n - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        if math.hypot(dx, dy) > 1e-9:
            yaw[i] = math.atan2(dy, dx)
    if n >= 2:
        yaw[-1] = yaw[-2]
    return s, yaw


def nearest_path_index(points, robot_x: float, robot_y: float) -> int:
    """返回离机器人最近的路径点索引（线性扫描，路径通常只有几百个点）。"""
    best_index = 0
    best_dist2 = math.inf
    for index, (px, py) in enumerate(points):
        dist2 = (px - robot_x) ** 2 + (py - robot_y) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_index = index
    return best_index


def carrot_index_by_arc_length(s, start_index: int, lookahead: float) -> int:
    """按弧长在路径上取前视点索引。

    1) ``start_index`` 是离机器人最近的路径索引（``s`` 的累计弧长零点在该路径起点）；
    2) 目标弧长 = ``s[start_index] + lookahead``；
    3) 从 start_index 向后线性扫描，返回第一个 ``s[j] >= 目标弧长`` 的 j；若剩余弧长
       不足 lookahead，则返回最后一点。

    用弧长而不是欧氏距离，弯道上不会因为"直线距离先到"而提前切角。
    """
    n = len(s)
    if n == 0:
        return 0
    start_index = max(0, min(int(start_index), n - 1))
    target_s = s[start_index] + max(float(lookahead), 0.0)
    j = start_index
    while j < n - 1 and s[j] < target_s:
        j += 1
    return j


def local_curvature_ahead(s, yaw, index: int, window_m: float = 0.5) -> float:
    """估计前视点前方约 ``window_m`` 米路径段的平均曲率（单位 1/m）。

    ``kappa = |wrap(yaw[j] - yaw[i])| / ds``，ds 为该段弧长。取"前方"而不是"后方"
    是因为机器人接下来要走的是前视点之后的路；弯道提前减速。
    若前方剩余弧长不足一个采样步长（路径末端），退回用最后一段的航向变化估计，
    避免末端曲率恒为 0 而在终点前突然超速。
    """
    n = len(s)
    if n < 2 or index < 0 or len(yaw) != n:
        return 0.0
    index = max(0, min(int(index), n - 1))
    j = index
    while j < n - 1 and (s[j] - s[index]) < float(window_m):
        j += 1
    ds = s[j] - s[index]
    if ds < 1e-6:
        # 前视点已在路径末端：用最后一段的航向变化估计局部曲率。
        if index > 0:
            ds = s[index] - s[index - 1]
            if ds < 1e-6:
                return 0.0
            return abs(_normalize_angle(yaw[index] - yaw[index - 1])) / ds
        return 0.0
    return abs(_normalize_angle(yaw[j] - yaw[index])) / ds


def apply_curvature_speed_limit(
    linear_x: float,
    s,
    yaw,
    index: int,
    a_lat_max: float,
    v_min: float,
    v_max: float,
    window_m: float = 0.5,
) -> float:
    """曲率限速：``v_curv = sqrt(a_lat_max / kappa)``，并把线速度钳到 [v_min, v_max]。

    只处理前进速度（linear_x > 0）；倒车段由路径跟随节点现有的脱困/自救状态机负责，
    不在这里处理（Path 消息不传递轨迹里的 v 符号）。
    """
    if linear_x <= 0.0:
        return linear_x
    kappa = local_curvature_ahead(s, yaw, index, window_m=window_m)
    v_curv = math.sqrt(max(a_lat_max, 0.0) / max(kappa, 1e-6))
    v_cap = max(float(v_min), min(float(v_max), v_curv))
    return min(linear_x, v_cap)


class PathFollowerNode(Node):
    """订阅探索器目标/路径与 TF，按前视点策略生成 /cmd_vel 速度指令。"""

    def __init__(self) -> None:
        super().__init__("path_follower")

        # ---------------------------------------------------------- 参数声明
        self.declare_parameter("path_topic", "exploration_path")
        self.declare_parameter("goal_topic", "exploration_goal")
        self.declare_parameter("done_topic", "exploration_done")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("lookahead_dist", 0.4)
        # 自适应前视：远处目标用大前视（曲率需求小、更稳），接近目标时自动缩小。
        self.declare_parameter("lookahead_adaptive", True)
        self.declare_parameter("lookahead_min", 0.4)
        self.declare_parameter("lookahead_max", 1.0)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("max_linear_vel", 0.3)
        # ---- 曲率限速（与上游阿克曼轨迹的 v_max / v_min / a_lat_max 保持一致）----
        # 弯道 v_curv = sqrt(a_lat_max / kappa)，再钳到 [traj_v_min, traj_v_max]。
        self.declare_parameter("traj_v_max", 0.30)
        self.declare_parameter("traj_v_min", 0.08)
        self.declare_parameter("a_lat_max", 0.60)
        # 默认 0.6：本来就不该超过真机可达范围，配合可达曲率钳制形成双重保护。
        self.declare_parameter("max_angular_vel", 0.6)
        self.declare_parameter("k_angular", 1.5)
        self.declare_parameter("min_linear_vel", 0.08)
        self.declare_parameter("slowdown_dist", 0.8)
        # ---- 可达曲率钳制 ----
        # 曲率钳制口径**只认 min_turn_radius_m**（默认 0.35，与上游阿克曼轨迹一致）：
        #   kappa_max = 1/min_turn_radius_m。
        # 上下游必须同一个 R_min，否则会出现"上游轨迹可行、下游控制器更紧"（实测
        # v/|w| < 0.35 占 49.8%，而 <0.271 为 0%，即 follower 按 0.271 钳得比上游更紧）。
        # wheelbase_m / max_steer_rad 保留，仅用于 min_turn_radius_m<=0 的旧行为回退
        # （以及日志/兼容）。
        self.declare_parameter("min_turn_radius_m", 0.35)
        self.declare_parameter("wheelbase_m", 0.144)
        self.declare_parameter("max_steer_rad", 0.489)
        self.declare_parameter("curvature_guard_enable", True)
        # 上游数据失效阈值（s）。上游无 scipy 时单轮 decide_frontier 可达 2.4~4.6s，
        # 默认 8.0 避免规划期间被误判为 stale 而周期性发零速。
        self.declare_parameter("stale_timeout_sec", 15.0)
        self.declare_parameter("reached_watchdog_sec", 8.0)
        # ---- 前向脱困状态机 ----
        self.declare_parameter("scan_topic", "scan")
        self.declare_parameter("enable_safety_brake", True)
        # 触发距离：正前方小于该值进入脱困。
        self.declare_parameter("brake_distance", 0.30)
        # 退出距离（回差）：正前方大于该值才算脱离（与 brake_distance 形成 0.30/0.55 回差）。
        self.declare_parameter("brake_clear_distance", 0.55)
        self.declare_parameter("brake_half_angle_deg", 35.0)
        # 脱困倒车速度 / 转向角速度 / 单次脱困最长时间。
        self.declare_parameter("escape_speed", 0.15)
        # 三点掉头"前进相"速度（m/s）与单个相位持续时间（s）。
        self.declare_parameter("escape_forward_speed", 0.10)
        self.declare_parameter("shuttle_sec", 0.25)
        self.declare_parameter("escape_angular", 0.35)
        self.declare_parameter("escape_timeout_sec", 10.0)
        # /scan 失效阈值：超过该秒数未收到新帧即视为没有 scan 数据、不脱困。
        self.declare_parameter("scan_timeout_sec", 1.0)
        # ---- 卡死检测 + 倒车自救 ----
        self.declare_parameter("stuck_detect_sec", 3.0)
        self.declare_parameter("stuck_move_eps", 0.05)
        self.declare_parameter("reverse_speed", 0.12)
        self.declare_parameter("reverse_sec", 1.0)
        # "正后方"锥形半角（度）：只把 |angle| > 180°-该值 的扇区算作车尾，默认正后方 ±30°。
        # 绝不能再用整个后半区（|angle|>90°），否则贴柱/贴墙平行停放时侧方柱面会被误判成
        # 车尾障碍，导致脱困只发零速而永久死锁。
        self.declare_parameter("reverse_half_angle_deg", 30.0)
        # 倒车后向安全距离（正后方锥形最小测距小于该值则取消本次倒车）。
        self.declare_parameter("reverse_brake_distance", 0.25)
        # 倒车时的角速度：一边倒一边转向，避免倒完仍朝同一面墙。
        self.declare_parameter("reverse_angular", 0.4)
        self.declare_parameter("enable_stuck_recovery", True)
        self.declare_parameter("stuck_recover_cooldown_sec", 5.0)

        self._global_frame = self.get_parameter("global_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._lookahead_dist = float(self.get_parameter("lookahead_dist").value)
        self._lookahead_adaptive = _as_bool(self.get_parameter("lookahead_adaptive").value)
        self._lookahead_min = float(self.get_parameter("lookahead_min").value)
        self._lookahead_max = float(self.get_parameter("lookahead_max").value)
        self._goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self._max_linear_vel = float(self.get_parameter("max_linear_vel").value)
        # 曲率限速参数。
        self._traj_v_max = float(self.get_parameter("traj_v_max").value)
        self._traj_v_min = float(self.get_parameter("traj_v_min").value)
        self._a_lat_max = float(self.get_parameter("a_lat_max").value)
        self._max_angular_vel = float(self.get_parameter("max_angular_vel").value)
        self._k_angular = float(self.get_parameter("k_angular").value)
        self._min_linear_vel = float(self.get_parameter("min_linear_vel").value)
        self._slowdown_dist = float(self.get_parameter("slowdown_dist").value)
        # 可达曲率钳制参数：min_turn_radius_m 是主口径，轴距/转角仅作回退。
        self._min_turn_radius_m = float(self.get_parameter("min_turn_radius_m").value)
        self._wheelbase_m = float(self.get_parameter("wheelbase_m").value)
        self._max_steer_rad = float(self.get_parameter("max_steer_rad").value)
        self._curvature_guard_enable = _as_bool(
            self.get_parameter("curvature_guard_enable").value
        )
        self._stale_timeout_sec = float(self.get_parameter("stale_timeout_sec").value)
        self._reached_watchdog_sec = float(self.get_parameter("reached_watchdog_sec").value)
        # 脱困状态机参数。
        self._scan_topic = self.get_parameter("scan_topic").value
        self._enable_safety_brake = _as_bool(self.get_parameter("enable_safety_brake").value)
        self._brake_distance = float(self.get_parameter("brake_distance").value)
        self._brake_clear_distance = float(self.get_parameter("brake_clear_distance").value)
        self._brake_half_angle_deg = float(self.get_parameter("brake_half_angle_deg").value)
        self._escape_speed = float(self.get_parameter("escape_speed").value)
        self._escape_forward_speed = float(self.get_parameter("escape_forward_speed").value)
        self._shuttle_sec = max(float(self.get_parameter("shuttle_sec").value), 1e-3)
        self._escape_angular = float(self.get_parameter("escape_angular").value)
        self._escape_timeout_sec = float(self.get_parameter("escape_timeout_sec").value)
        self._scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        # 回差必须大于触发距离，否则一进入就可能立刻退出（配置写反时自动纠正）。
        if self._brake_clear_distance <= self._brake_distance:
            self.get_logger().warn(
                f"brake_clear_distance({self._brake_clear_distance}) <= "
                f"brake_distance({self._brake_distance})，自动取 brake_distance + 0.25"
            )
            self._brake_clear_distance = self._brake_distance + 0.25
        # 卡死检测 / 倒车自救参数。
        self._stuck_detect_sec = float(self.get_parameter("stuck_detect_sec").value)
        self._stuck_move_eps = float(self.get_parameter("stuck_move_eps").value)
        self._reverse_speed = float(self.get_parameter("reverse_speed").value)
        self._reverse_sec = float(self.get_parameter("reverse_sec").value)
        self._reverse_half_angle_deg = float(
            self.get_parameter("reverse_half_angle_deg").value
        )
        self._reverse_brake_distance = float(
            self.get_parameter("reverse_brake_distance").value
        )
        self._reverse_angular = float(self.get_parameter("reverse_angular").value)
        self._enable_stuck_recovery = _as_bool(self.get_parameter("enable_stuck_recovery").value)
        self._stuck_recover_cooldown_sec = float(
            self.get_parameter("stuck_recover_cooldown_sec").value
        )

        # H1：非法配置兜底，min_linear_vel 不得大于等于 max_linear_vel。
        if self._min_linear_vel >= self._max_linear_vel:
            self.get_logger().warn(
                f"min_linear_vel({self._min_linear_vel}) >= max_linear_vel({self._max_linear_vel})，"
                f"自动压到 max_linear_vel 的一半"
            )
            self._min_linear_vel = self._max_linear_vel * 0.5

        # 曲率限速参数兜底：下限不得高于上限，横向加速度必须为正。
        if self._traj_v_min > self._traj_v_max:
            self.get_logger().warn(
                f"traj_v_min({self._traj_v_min}) > traj_v_max({self._traj_v_max})，交换两者"
            )
            self._traj_v_min, self._traj_v_max = self._traj_v_max, self._traj_v_min
        self._traj_v_min = max(0.0, self._traj_v_min)
        self._traj_v_max = max(self._traj_v_min, self._traj_v_max)
        if self._a_lat_max <= 0.0:
            self.get_logger().warn(
                f"a_lat_max({self._a_lat_max}) 非法，曲率限速改用默认 0.60m/s²"
            )
            self._a_lat_max = 0.60

        # H2：自适应前视的上下限写反时自动纠正（否则 clamp 会退化成固定下限）。
        if self._lookahead_min > self._lookahead_max:
            self.get_logger().warn(
                f"lookahead_min({self._lookahead_min}) > lookahead_max({self._lookahead_max})，"
                f"交换两者"
            )
            self._lookahead_min, self._lookahead_max = (
                self._lookahead_max,
                self._lookahead_min,
            )
        # 上限不得为负；轴距/转角非法时钳制会失去意义，这里只做兜底（计算时也会取 max）。
        self._lookahead_min = max(0.0, self._lookahead_min)
        self._lookahead_max = max(self._lookahead_min, self._lookahead_max)
        if self._curvature_guard_enable:
            # min_turn_radius_m <= 0 表示显式要求回退到几何值 tan(max_steer)/L（旧行为）。
            if self._min_turn_radius_m <= 0.0:
                self.get_logger().info(
                    f"min_turn_radius_m({self._min_turn_radius_m}) <= 0，"
                    f"曲率钳制回退为 tan(max_steer)/wheelbase_m"
                )
            # 轴距/转角只在回退路径与日志中使用，仍做合法性兜底。
            if self._wheelbase_m <= 1e-6:
                self.get_logger().warn(
                    f"wheelbase_m({self._wheelbase_m}) 非法，曲率钳制回退值改用默认 0.144m"
                )
                self._wheelbase_m = 0.144
            if not (0.0 < self._max_steer_rad < math.pi / 2.0):
                self.get_logger().warn(
                    f"max_steer_rad({self._max_steer_rad}) 不在 (0, pi/2) 内，"
                    f"曲率钳制回退值改用默认 0.489rad"
                )
                self._max_steer_rad = 0.489

        # ---------------------------------------------------------- 运行状态
        self._path_msg: Optional[Path] = None
        self._goal_msg: Optional[PoseStamped] = None
        # 路径的累计弧长数组与每点朝向（与 _path_msg.poses 等长；无路径时为 None）。
        # 弧长前视与曲率限速都直接查这两个数组，避免每个控制周期重复计算。
        self._path_s: Optional[list] = None
        self._path_yaw: Optional[list] = None
        # 探索完成标志：收到 True 后永久停车。
        self._exploration_done = False
        # 两路输入各自最近一次的接收时间，用于失效判定。
        self._path_recv_time: Optional[Time] = None
        self._goal_recv_time: Optional[Time] = None
        # 到达状态及其对应的目标位置，用于识别上游是否规划出了新前沿。
        self._reached = False
        self._reached_target: Optional[Tuple[float, float]] = None
        # reached 起始时刻与"已告警"标记，用于驻留看门狗。
        self._reached_time: Optional[Time] = None
        self._reached_warned = False

        # ---------------------------------------------------- 安全刹车运行状态
        # 是否收到过任何 scan（没收到就绝不刹车，避免缺数据永久停）。
        self._scan_received = False
        # 最近一帧 scan 的接收时间（用于 scan_timeout_sec 失效保护）。
        self._last_scan_time: Optional[Time] = None
        # 车体正前方扇区内有效测距的最小值（米）；无有效点时保持 None。
        self._front_range: Optional[float] = None
        # 车体正后方窄锥形（|angle| > π - radians(reverse_half_angle_deg)）有效测距的
        # 最小值（米）；倒车后向安全校验用。侧方扇区不计入，避免侧墙/柱子被误判成车尾障碍。
        self._rear_range: Optional[float] = None
        # 左右侧扇区（brake_half_angle .. 90°）最小有效测距：脱困时朝空间更大的一侧转。
        self._left_range: Optional[float] = None
        self._right_range: Optional[float] = None
        # 脱困状态机：是否在脱困、起始时刻（秒）、转向方向（左=+1 / 右=-1）。
        self._escaping = False
        self._escape_start_ts: Optional[float] = None
        self._escape_dir = 1
        # 两相位三点掉头：当前相位（"reverse"/"forward"）与该相位的结束时刻（秒）。
        self._escape_phase = "reverse"
        self._escape_phase_end_ts: Optional[float] = None

        # -------------------------------------------------- 卡死检测/倒车运行状态
        # 最近 stuck_detect_sec 秒的位姿窗口：(时刻s, x, y) 三元组队列。
        self._pose_window: deque = deque()
        # 最近一次"指令非零"的时刻（秒），用于判断"想走走不动"。
        self._cmd_nonzero_ts: Optional[float] = None
        # 倒车自救状态机：是否在倒车、倒车结束时刻（秒）、上次触发时刻（秒）。
        self._reversing = False
        self._reverse_end_ts: Optional[float] = None
        self._last_recover_ts: Optional[float] = None

        # ---------------------------------------------------------- TF
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---------------------------------------------------------- 收发
        self._cmd_vel_pub = self.create_publisher(
            Twist, self.get_parameter("cmd_vel_topic").value, 10
        )
        self.create_subscription(
            Path, self.get_parameter("path_topic").value, self._on_path, 10
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("goal_topic").value, self._on_goal, 10
        )
        # 完成标志订阅：QoS 必须与发布端匹配（TRANSIENT_LOCAL/latched）。
        self.create_subscription(
            Bool,
            self.get_parameter("done_topic").value,
            self._on_done,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # 激光雷达订阅：用于前向安全刹车（默认 10 深度即可）。
        self.create_subscription(
            LaserScan, self._scan_topic, self._on_scan, 10
        )

        period = 1.0 / max(float(self.get_parameter("control_rate_hz").value), 1e-3)
        self.create_timer(period, self._on_control)

        # 日志用：实际生效的最小转弯半径（min_turn_radius_m<=0 时按几何值回退）。
        effective_r_min = (
            self._min_turn_radius_m
            if self._min_turn_radius_m > 0.0
            else self._wheelbase_m / max(math.tan(self._max_steer_rad), 1e-6)
        )
        self.get_logger().info(
            f"path_follower 启动 | path={self.get_parameter('path_topic').value} | "
            f"goal={self.get_parameter('goal_topic').value} | "
            f"done={self.get_parameter('done_topic').value} | "
            f"cmd_vel={self.get_parameter('cmd_vel_topic').value} | "
            f"tf={self._global_frame}->{self._robot_frame} | "
            f"vmax={self._max_linear_vel} | wmax={self._max_angular_vel} | "
            f"lookahead={self._lookahead_dist}"
            f"(adaptive={self._lookahead_adaptive},"
            f"{self._lookahead_min}~{self._lookahead_max}m) | "
            f"tolerance={self._goal_tolerance} | "
            f"vmin={self._min_linear_vel} | slow={self._slowdown_dist} | "
            f"curv_speed={self._traj_v_min}~{self._traj_v_max}m/s"
            f"(a_lat={self._a_lat_max}) | "
            f"curv_guard={self._curvature_guard_enable}"
            f"(Rmin={effective_r_min:.3f}m[min_turn_radius={self._min_turn_radius_m}m]/"
            f"L={self._wheelbase_m}m/steer {self._max_steer_rad}rad) | "
            f"stale={self._stale_timeout_sec}s | "
            f"scan={self._scan_topic} | brake={self._enable_safety_brake}"
            f"({self._brake_distance}m/clear {self._brake_clear_distance}m/"
            f"±{self._brake_half_angle_deg}°) | "
            f"escape={self._escape_speed}m/s+fwd {self._escape_forward_speed}m/s/"
            f"{self._escape_angular}rad/s/shuttle {self._shuttle_sec}s/"
            f"{self._escape_timeout_sec}s | "
            f"rear_cone=±{self._reverse_half_angle_deg}°({self._reverse_brake_distance}m) | "
            f"stuck_recover={self._enable_stuck_recovery}"
            f"({self._stuck_detect_sec}s/{self._stuck_move_eps}m)"
        )

    # ------------------------------------------------------------------ 回调
    def _on_path(self, msg: Path) -> None:
        """缓存最新一帧路径，并预先算好累计弧长与每点朝向。

        朝优先取 pose.orientation 解出的 yaw（阿克曼轨迹会带真实切线朝向）；若整条路径
        的四元数全为 0 或全为单位四元数（回退 A* 折线没带朝向），则用相邻点几何切线补算。
        这样控制循环里的弧长前视与曲率限速都只是数组查询，不再每周期重算。
        """
        self._path_msg = msg
        self._path_recv_time = self.get_clock().now()
        points, yaws = path_geometry_from_poses(msg.poses)
        self._path_s, self._path_yaw = compute_path_arc_lengths(points, yaws)

    def _on_goal(self, msg: PoseStamped) -> None:
        """缓存最新一帧目标点。"""
        self._goal_msg = msg
        self._goal_recv_time = self.get_clock().now()

    def _on_done(self, msg: Bool) -> None:
        """收到探索完成标志：置位一次并打印，之后 _on_control 会持续发零 Twist。"""
        if msg.data and not self._exploration_done:
            self._exploration_done = True
            # done 拥有最高优先级（永远零速停车），顺带清掉倒车自救/脱困状态。
            self._reversing = False
            self._reverse_end_ts = None
            self._escaping = False
            self._escape_start_ts = None
            self._escape_phase_end_ts = None
            self.get_logger().info("收到探索完成标志，停止行进")

    def _on_scan(self, msg: LaserScan) -> None:
        """缓存一帧激光雷达的前向/正后方/左右扇区最小测距。

        前向：只取车体正前方 ±brake_half_angle_deg 扇区内、且 range 有效的点，取最小值。
        左右侧：+brake_half_angle .. +90° 与 -90° .. -brake_half_angle 两个扇区的最小值，
        供脱困时选择"朝空间更大的一侧转"。
        **正后方窄锥形**：只统计 |angle| > π - radians(reverse_half_angle_deg) 的点
        （默认 |angle| > 150°，即正后方 ±30°），供倒车后向安全校验使用。绝不能再用整个
        后半区（|angle| > π/2）：贴柱/贴墙平行停放时侧方柱面会落进后半区，被误判成
        "车尾障碍"，让脱困退化成只发零速而死锁。
        忽略 None / nan / inf / 非正 / 超出 range_max 的测距。
        """
        self._scan_received = True
        # 记录最近一帧 scan 的接收时间（用 ros 时钟，兼容 use_sim_time）。
        self._last_scan_time = self.get_clock().now()
        half_angle = math.radians(self._brake_half_angle_deg)
        # 正后方锥形边界：|angle| 必须大于该值才算"车尾"（默认 180°-30°=150°）。
        rear_cone_limit = math.pi - math.radians(self._reverse_half_angle_deg)
        front = math.inf
        rear = math.inf
        left = math.inf
        right = math.inf
        for index, distance in enumerate(msg.ranges):
            # LaserScan 角度约定：angle = angle_min + i * angle_increment，0 rad 指向车体正前方。
            angle = msg.angle_min + index * msg.angle_increment
            if distance is None:
                continue
            if math.isnan(distance) or math.isinf(distance):
                continue
            if distance <= 0.0 or distance > msg.range_max:
                continue
            # 正后方窄锥形（默认 |angle| > 150°）用于倒车后向安全校验。
            if abs(angle) > rear_cone_limit:
                if distance < rear:
                    rear = distance
                continue
            # 侧方扇区（90° < |angle| <= 150°）既不算车尾也不算左右侧，直接忽略。
            if abs(angle) > math.pi / 2.0:
                continue
            # 左侧扇区（+half_angle .. +90°）与右侧扇区（-90° .. -half_angle）。
            if angle >= half_angle:
                if distance < left:
                    left = distance
            elif angle <= -half_angle:
                if distance < right:
                    right = distance
            if abs(angle) > half_angle:
                continue
            if distance < front:
                front = distance
        # 扇区内没有任何有效测距时，对应 range 记为 None（此时不触发/不阻断）。
        self._front_range = None if math.isinf(front) else float(front)
        self._rear_range = None if math.isinf(rear) else float(rear)
        self._left_range = None if math.isinf(left) else float(left)
        self._right_range = None if math.isinf(right) else float(right)

    # ---------------------------------------------------------------- 工具函数
    def _is_stale(self) -> bool:
        """判断上游数据是否缺失或过期。"""
        times = [t for t in (self._path_recv_time, self._goal_recv_time) if t is not None]
        if not times:
            # 从未收到过任何 path/goal。
            return True
        newest = max(times)
        age = (self.get_clock().now() - newest).nanoseconds * 1e-9
        return age > self._stale_timeout_sec

    def _final_target(self) -> Optional[Tuple[float, float]]:
        """返回最终目标位置：优先用路径最后一个点，否则用收到的 goal。"""
        if self._path_msg is not None and len(self._path_msg.poses) > 0:
            last = self._path_msg.poses[-1].pose.position
            return (last.x, last.y)
        if self._goal_msg is not None:
            pos = self._goal_msg.pose.position
            return (pos.x, pos.y)
        return None

    def _lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """查询机器人当前位姿 (x, y, yaw)，失败返回 None。"""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame, self._robot_frame, Time()
            )
        except TransformException as exc:  # TF 未就绪/超时/断链等
            self.get_logger().warn(
                f"TF 查询失败 {self._global_frame}->{self._robot_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = _yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
        return (translation.x, translation.y, yaw)

    def _resolve_lookahead(
        self, robot_x: float, robot_y: float, target_x: float, target_y: float
    ) -> float:
        """计算本周期使用的前视距离（米）。

        lookahead_adaptive=True 时按到最终目标的距离自适应：
            lookahead = clamp(0.6 * target_dist, lookahead_min, lookahead_max)
        并且**不超过到路径末点（最终目标）的距离**：目标很近时前视点直接落在目标上，
        不会越过目标继续"看"到目标之外的点。
        远处目标用大前视 -> 同样横向偏差对应的期望曲率更小、跟踪更稳；接近目标时自动
        缩小前视，避免抄近路切角。lookahead_adaptive=False 时退回固定 lookahead_dist。
        """
        if not self._lookahead_adaptive:
            return self._lookahead_dist
        target_dist = math.hypot(target_x - robot_x, target_y - robot_y)
        lookahead = 0.6 * target_dist
        lookahead = max(self._lookahead_min, min(self._lookahead_max, lookahead))
        # 上限不超过"到路径末点（最终目标）的距离"。
        return min(lookahead, target_dist)

    def _pick_carrot(
        self, robot_x: float, robot_y: float, lookahead: float
    ) -> Optional[Tuple[float, float, int]]:
        """按**弧长**在路径上选取前视点；无路径时退回 goal。

        返回 ``(x, y, index)``；``index`` 为路径点索引（无路径时 -1），供曲率限速使用。

        步骤（替换原来的欧氏"第一个距离 >= lookahead"）：
          1) 找离机器人最近的路径索引 i0；
          2) 从 ``s[i0] + lookahead`` 找到对应索引 j（线性扫描）；
          3) 返回 poses[j]；若剩余弧长不足 lookahead，返回最后一点。

        为什么用弧长：欧氏前视在弯道上会选到"直线距离先够、但绕路更远"的点，导致切角；
        弧长前视保证前视点沿路径前进固定距离。
        """
        if self._path_msg is not None and len(self._path_msg.poses) > 0:
            points = [
                (pose_stamped.pose.position.x, pose_stamped.pose.position.y)
                for pose_stamped in self._path_msg.poses
            ]
            # 防御：path_s 与 poses 长度不一致时（理论上 _on_path 已同步）重建一次。
            if self._path_s is None or self._path_yaw is None or len(self._path_s) != len(points):
                self._path_s, self._path_yaw = compute_path_arc_lengths(points, None)
            i0 = nearest_path_index(points, robot_x, robot_y)
            j = carrot_index_by_arc_length(self._path_s, i0, lookahead)
            return (points[j][0], points[j][1], j)
        # 没有路径但有目标点时，用目标当当前视点（没有路径弧长，曲率限速不生效）。
        if self._goal_msg is not None:
            pos = self._goal_msg.pose.position
            return (pos.x, pos.y, -1)
        return None

    def _apply_curvature_guard(self, linear_x: float, angular_z: float) -> float:
        """可达曲率钳制：把角速度限制在当前线速度下物理可实现的曲率内。

        **钳制口径（问题 3 修复）**：``kappa_max = 1.0 / min_turn_radius_m``。
        必须与上游阿克曼轨迹模块的 ``min_turn_radius_m`` 使用同一个值：
          - 上游按 0.35m 生成的圆弧是"曲率恰好 1/0.35"的可行轨迹；
          - 若下游按 ``tan(max_steer)/L``（=0.271m 的几何值）反算，钳制更紧，
            上游刚够用的角速度会被额外削掉 -> 跟线滞后、横向误差不收敛
            （实测 v/|w| < 0.35 占 49.8%，而 < 0.271 为 0%，正是两层口径不一致）；
          - 若下游更松，则可能下发物理不可达的曲率，转角被 max_steer 截断 ->
            满舵绕圈。
        二者保持同一个 R_min，才能让"上游可行"等价于"下游可执行"。

        ``min_turn_radius_m <= 0`` 时回退到旧行为 ``tan(max_steer)/wheelbase_m``，
        仅用于兼容旧配置/离线对照。

        为什么仍需钳制：gazebo_ros_ackermann_drive 插件按 steer = atan(L*w/v) 反解
        前轮转角，|w| 超过 |v|*kappa_max 时转角会被 max_steer 截断，实际曲率到不了
        指令曲率 -> 横向误差不收敛 -> 一直满舵绕圈（实测 91% 采样 |w|>=0.95、
        净转 133 圈只换来 5.7m 净位移）。低速时 w_limit 也小，这是阿克曼底盘的
        真实约束（不能原地转）；min_linear_vel 保证了低速时仍保留可用转向能力。
        """
        if not self._curvature_guard_enable:
            return angular_z
        if self._min_turn_radius_m > 0.0:
            # 主口径：与上游阿克曼轨迹完全一致。
            kappa_max = 1.0 / self._min_turn_radius_m
        else:
            # 兼容旧行为：几何可达曲率上限。
            kappa_max = math.tan(self._max_steer_rad) / max(self._wheelbase_m, 1e-6)
        w_limit = abs(linear_x) * kappa_max
        if w_limit <= 1e-6:
            # 线速度接近 0：阿克曼底盘几乎不能转，保持原值（调用方的 min_linear_vel
            # 会兜住正常跟线场景，这里不做额外放开）。
            return angular_z
        clamped = max(-w_limit, min(w_limit, angular_z))
        if abs(clamped - angular_z) > 1e-9:
            # 节流 5s，避免每周期刷屏。
            self.get_logger().info(
                f"角速度被可达曲率钳制：w {angular_z:.2f} -> {clamped:.2f}"
                f"（v={linear_x:.2f}, R_min={1.0 / kappa_max:.3f}m）",
                throttle_duration_sec=5.0,
            )
        return clamped

    def _publish_twist(self, linear_x: float, angular_z: float) -> None:
        """发布一帧速度指令。"""
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(twist)
        # 记录最近一次"指令非零"的时刻：卡死检测用它判断这几秒是否一直在"想走"。
        if abs(linear_x) > 1e-6 or abs(angular_z) > 1e-6:
            self._cmd_nonzero_ts = self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------- 安全刹车 / 卡死自救
    def _scan_is_fresh(self) -> bool:
        """scan 是否新鲜：最近一帧 scan 距现在不超过 scan_timeout_sec。

        用于 /scan 失效保护：超过阈值视为"没有 scan 数据"，此时既不刹车也不阻断倒车，
        避免 scan 停在一帧贴近障碍的数据后永久刹车。
        """
        if self._last_scan_time is None:
            return False
        age = (self.get_clock().now() - self._last_scan_time).nanoseconds * 1e-9
        return age <= self._scan_timeout_sec

    def _front_is_blocked(self) -> bool:
        """是否需要进入脱困状态。

        条件：启用刹车、收到过 scan、scan 新鲜、且正前方扇区最小测距
        front_range < brake_distance。从未收到过 scan（如 /scan 没起）、scan 失效或扇形内
        无有效点时返回 False（不脱困，按原逻辑走）。
        """
        if not self._enable_safety_brake or not self._scan_received:
            return False
        if not self._scan_is_fresh():
            return False
        if self._front_range is None:
            return False
        return self._front_range < self._brake_distance

    def _choose_escape_dir(self) -> int:
        """选择脱困转向方向：朝左右侧空间中更大的一侧转（左=+1，右=-1）。

        比较激光左侧扇区（+brake_half_angle .. +90°）与右侧扇区
        （-90° .. -brake_half_angle）的最小有效测距；某侧无有效点视为该侧开阔。
        两侧都无有效数据时取 +1（左）。
        """
        left = self._left_range
        right = self._right_range
        if left is None and right is None:
            return 1
        if left is None:
            return 1
        if right is None:
            return -1
        return 1 if left >= right else -1

    def _switch_escape_phase(self, now_sec: float) -> None:
        """切换三点掉头相位（reverse <-> forward），并把该相位结束时刻推到 now+shuttle_sec。"""
        self._escape_phase = "forward" if self._escape_phase == "reverse" else "reverse"
        self._escape_phase_end_ts = now_sec + self._shuttle_sec

    def _rear_is_blocked(self) -> bool:
        """倒车后向安全校验：车体**正后方窄锥形**最小测距是否过近。

        只统计 /scan 中 |angle| > π - radians(reverse_half_angle_deg) 的点（默认正后方
        ±30°），最小有效测距 < reverse_brake_distance 时返回 True。侧方（90° < |angle| <=
        150°）不再参与判定——贴柱平行停放时侧方柱面不会被误当成车尾障碍。
        未收到 scan / 无有效点 / scan 失效时返回 False（不阻断倒车，避免缺数据导致永不倒车）。
        """
        if self._rear_range is None:
            return False
        if not self._scan_is_fresh():
            return False
        return self._rear_range < self._reverse_brake_distance

    def _recover_cooldown_ok(self, now_sec: float) -> bool:
        """距上次倒车自救是否已过 stuck_recover_cooldown_sec 冷却期。"""
        return (
            self._last_recover_ts is None
            or (now_sec - self._last_recover_ts) >= self._stuck_recover_cooldown_sec
        )

    def _start_reverse(self, now_sec: float, warn_msg: str) -> bool:
        """发起一段倒车自救：进入倒车状态并发一帧倒车指令，成功返回 True。

        统一入口（卡死检测走这里；脱困由 _on_control 里的独立状态机处理）：
          - 冷却期保护：距上次不足 stuck_recover_cooldown_sec 则拒绝触发；
          - 后向安全校验：车尾过近（_rear_is_blocked）则取消本次倒车（避免倒进墙里）；
          - 倒车期间发 Twist(-reverse_speed, reverse_angular)：一边倒一边转向改变朝向；
          - 进入倒车时清掉脱困状态（_reversing 优先级高于 _escaping）。
        """
        if not self._recover_cooldown_ok(now_sec):
            return False
        if self._rear_is_blocked():
            self.get_logger().warn(
                f"车尾后方 {self._rear_range:.2f}m 内有障碍"
                f"（< {self._reverse_brake_distance}m），取消本次倒车自救",
                throttle_duration_sec=2.0,
            )
            return False

        self._reversing = True
        self._reverse_end_ts = now_sec + self._reverse_sec
        self._last_recover_ts = now_sec
        self._pose_window.clear()
        # 倒车自救优先级高于脱困：进入自救时清掉脱困状态（含三点掉头相位），二者不互相打架。
        self._escaping = False
        self._escape_start_ts = None
        self._escape_phase_end_ts = None
        self.get_logger().warn(warn_msg)
        # 倒车带一点转向，让它顺便改变朝向（否则倒完还朝同一面墙）。
        self._publish_twist(-self._reverse_speed, self._reverse_angular)
        return True

    def _update_stuck_and_maybe_reverse(self, robot_x: float, robot_y: float) -> bool:
        """卡死检测 + 倒车自救状态机。

        维护最近 stuck_detect_sec 秒的位姿窗口；若窗口内机器人净位移 < stuck_move_eps
        且期间指令速度非零（"想走走不动"），判定卡死并发起一段 reverse_sec 的倒车自救。
        连续两次触发之间至少间隔 stuck_recover_cooldown_sec，避免抖动式反复倒车。
        返回 True 表示本周期已发出倒车指令（调用方应直接 return）。
        """
        if not self._enable_stuck_recovery:
            return False

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        # 追加当前位姿，并丢弃超出时间窗口的旧样本。
        self._pose_window.append((now_sec, robot_x, robot_y))
        while self._pose_window and now_sec - self._pose_window[0][0] > self._stuck_detect_sec:
            self._pose_window.popleft()

        # 已在倒车中 / 从未发过非零指令 -> 不判定卡死。
        if self._reversing or self._cmd_nonzero_ts is None:
            return False

        oldest_sec, oldest_x, oldest_y = self._pose_window[0]
        window_span = now_sec - oldest_sec
        # 窗口内是否存在非零指令（用最近一次非零指令时刻衡量"一直在想走"）。
        cmd_recent = (now_sec - self._cmd_nonzero_ts) <= self._stuck_detect_sec
        # 需要足够长的观测窗口 + 期间指令非零，才做位移判定（冷却/后向校验在 _start_reverse 内）。
        if window_span < self._stuck_detect_sec or not cmd_recent:
            return False

        displacement = math.hypot(robot_x - oldest_x, robot_y - oldest_y)
        if displacement >= self._stuck_move_eps:
            return False

        # 触发倒车自救：进入一段 reverse_sec 的倒车状态（期间不再重入触发）。
        return self._start_reverse(
            now_sec,
            f"检测到卡死（{self._stuck_detect_sec}s 内位移 < {self._stuck_move_eps}m "
            f"且指令非零），倒车自救 {self._reverse_sec}s",
        )

    # ------------------------------------------------------------------ 主循环
    def _on_control(self) -> None:
        # 0) 探索已完成：永久停车，不再跟随任何目标。
        if self._exploration_done:
            self._publish_twist(0.0, 0.0)
            return

        # 0.5) 倒车自救进行中：优先执行倒车动作（不重入触发），结束后回到正常跟踪。
        if self._reversing:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if self._reverse_end_ts is not None and now_sec >= self._reverse_end_ts:
                # 倒车结束：清状态，本周期继续按正常跟踪走（不 return）。
                self._reversing = False
                self._reverse_end_ts = None
            elif self._rear_is_blocked():
                # 倒车中车尾遇到障碍：立即取消倒车（避免倒进墙里），本周期继续正常跟踪。
                self._reversing = False
                self._reverse_end_ts = None
                self.get_logger().warn(
                    f"倒车中车尾 {self._rear_range:.2f}m 内有障碍"
                    f"（< {self._reverse_brake_distance}m），取消倒车",
                    throttle_duration_sec=2.0,
                )
            else:
                # 倒车期间带一点转向（reverse_angular），顺便改变朝向，避免倒完还朝同一面墙。
                self._publish_twist(-self._reverse_speed, self._reverse_angular)
                return

        # 0.8) 阿克曼脱困状态机：两相位三点掉头（优先级：done > _reversing > _escaping > stale）。
        #      触发：未在脱困、启用刹车、scan 新鲜且正前方 < brake_distance。
        #      动作：在倒车相与前进相之间交替，两相转向符号相反，使 yaw 朝同一方向累积；
        #      退出：正前方 > brake_clear_distance（0.30/0.55 回差）或单次脱困超时。
        #      脱困期间不走正常跟线；**每个周期都必须输出非零指令**，绝不发零速死锁。
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if not self._escaping and self._front_is_blocked():
            # 进入脱困：选方向、进入倒车相、记录起始时刻与相位结束时刻，只打一条日志。
            self._escaping = True
            self._escape_start_ts = now_sec
            self._escape_dir = self._choose_escape_dir()
            self._escape_phase = "reverse"
            self._escape_phase_end_ts = now_sec + self._shuttle_sec
            direction_text = "左" if self._escape_dir > 0 else "右"
            self.get_logger().warn(
                f"正前方 {self._front_range:.2f}m 有障碍，开始三点掉头脱困"
                f"（倒车/前进交替，转向{direction_text}）"
            )

        if self._escaping:
            # scan 新鲜但前向扇区无有效测距 -> 视为前方已清空（开阔）。
            front_clear = self._scan_is_fresh() and (
                self._front_range is None
                or self._front_range > self._brake_clear_distance
            )
            escape_timeout = (
                self._escape_start_ts is not None
                and (now_sec - self._escape_start_ts) > self._escape_timeout_sec
            )
            if front_clear or escape_timeout:
                # 退出脱困：只打一条日志，本周期继续正常跟线（不 return）。
                self._escaping = False
                self._escape_start_ts = None
                self._escape_phase_end_ts = None
                front_text = (
                    f"{self._front_range:.2f}m"
                    if self._front_range is not None
                    else "无有效测距"
                )
                self.get_logger().info(f"脱困结束（前方 {front_text}）")
            else:
                # 相位超时：切换一次相位，保证两相位交替，不会长期吊在一个相位上。
                if (
                    self._escape_phase_end_ts is not None
                    and now_sec >= self._escape_phase_end_ts
                ):
                    self._switch_escape_phase(now_sec)

                just_switched_to_forward = False
                if self._escape_phase == "reverse":
                    if self._rear_is_blocked():
                        # 正后方窄锥形贴得太近：不能倒，切到前进相，本周期改发前进指令。
                        self._switch_escape_phase(now_sec)
                        just_switched_to_forward = True
                    else:
                        # 倒车相：倒车 + 朝空间更大的一侧转向（dir 左=+1 / 右=-1）。
                        self._publish_twist(
                            -self._escape_speed,
                            self._escape_dir * self._escape_angular,
                        )
                        return

                if self._escape_phase == "forward":
                    if (
                        not just_switched_to_forward
                        and self._front_range is not None
                        and self._front_range < self._brake_distance
                    ):
                        # 前方贴得太近，不能再进：切回倒车相，本周期发倒车指令（非零）。
                        self._switch_escape_phase(now_sec)
                        self._publish_twist(
                            -self._escape_speed,
                            self._escape_dir * self._escape_angular,
                        )
                        return
                    # 前进相：**转向符号与倒车相相同**（都用 +dir）。
                    # 原因：gazebo_ros_ackermann_drive 先按 steer = atan(L*w/v) 反解前轮转角，
                    # 再由 steer 产生横摆，等效 yaw_rate = v*tan(steer)/L == w。
                    # 也就是说倒车（v<0）时转角自动反向，yaw_rate 仍等于 w、符号不变。
                    # 因此两相位用**同号** angular.z 才能让 yaw 单向累积（真正的三点掉头）；
                    # 若前进相取反号，两相 yaw 正负相消，等于原地空耗（实测 77 次脱困
                    # 航向净变化中位仅 0.07 rad）。
                    self._publish_twist(
                        self._escape_forward_speed,
                        self._escape_dir * self._escape_angular,
                    )
                    return

        # 1) 上游数据缺失/过期：发零 Twist 兜底。
        if self._is_stale():
            self._publish_twist(0.0, 0.0)
            return

        # 2) 取机器人当前位姿，失败则发零 Twist。
        pose = self._lookup_robot_pose()
        if pose is None:
            self._publish_twist(0.0, 0.0)
            return
        robot_x, robot_y, yaw = pose

        target = self._final_target()
        if target is None:
            # 理论上被 _is_stale() 挡住，这里再兜一层。
            self._publish_twist(0.0, 0.0)
            return
        target_x, target_y = target

        # 3) 到达判定。
        target_dist = math.hypot(target_x - robot_x, target_y - robot_y)
        if target_dist <= self._goal_tolerance:
            if not self._reached:
                self._reached = True
                self._reached_target = (target_x, target_y)
                self._reached_time = self.get_clock().now()
                self._reached_warned = False
                # 已到达：本状态零速，不参与卡死判定，清空位姿窗口避免残留样本误判。
                self._pose_window.clear()
                self.get_logger().info(
                    f"已到达目标 ({target_x:.2f}, {target_y:.2f})，"
                    f"距离 {target_dist:.3f} m，停止行进等待新前沿"
                )
            self._publish_twist(0.0, 0.0)
            return

        # 4) 处于 reached 状态：判断是否恢复行进。
        #    改动理由：上游探索器有时会连续重发"步长很小"的新目标（相邻前沿只挪动不到
        #    0.25m），此时机器人明明已经站到该点到达半径内，却因为旧判据"新目标相对上次
        #    到达点位移 > goal_tolerance"一直不满足，导致零速干等数十秒。
        #    新判据兼顾两点：① 目标坐标与上次到达点不同（确实换了目标，保留"同一目标不重复
        #    追"的原意，避免对同一个已到达点反复启停）；② 当前机器人到该目标的距离已经大于
        #    goal_tolerance（说明现在确实还没到它）。只要满足这两点就恢复行进；只有"确实还是
        #    同一个已到达点"时才继续停车等待。
        if self._reached:
            if self._reached_target is None:
                # 理论上不会发生，兜底清状态。
                self.get_logger().warn("reached 状态缺少记录目标，重置为行进")
                self._reached = False
                self._reached_time = None
                self._reached_warned = False
            else:
                same_target = (
                    target_x == self._reached_target[0] and target_y == self._reached_target[1]
                )
                if (not same_target) and target_dist > self._goal_tolerance:
                    moved = math.hypot(
                        target_x - self._reached_target[0], target_y - self._reached_target[1]
                    )
                    self.get_logger().info(
                        f"检测到新前沿目标 ({target_x:.2f}, {target_y:.2f})，"
                        f"距上一到达点 {moved:.3f} m，恢复行进"
                    )
                    self._reached = False
                    self._reached_target = None
                    self._reached_time = None
                    self._reached_warned = False
                    # 恢复行进：清空位姿窗口，卡死检测从新目标开始重新累计。
                    self._pose_window.clear()
                else:
                    # 驻留看门狗：长时间没有换新目标时，仅告警一次，行为仍是停车。
                    if (
                        not self._reached_warned
                        and self._reached_time is not None
                        and (self.get_clock().now() - self._reached_time).nanoseconds * 1e-9
                        > self._reached_watchdog_sec
                    ):
                        self._reached_warned = True
                        self.get_logger().warn(
                            f"已到达 ({self._reached_target[0]:.2f}, {self._reached_target[1]:.2f}) "
                            f"且 {self._reached_watchdog_sec:.1f}s 内上游未给出新的更远目标："
                            f"该前沿可能不可达，或探索已结束（完成判定见 Step 3）"
                        )
                    self._publish_twist(0.0, 0.0)
                    return

        # 4.5) 前向脱困已在上面的 0.8) 处理：脱困期间直接倒车+转向并 return，
        #      不再"把线速度压到爬行速度继续朝墙前进"（旧刹车限幅逻辑已删除）。

        # 5) 选取前视点并计算航向误差。
        #    前视距离可自适应（按到最终目标的距离），上限不超过到路径末点的距离。
        lookahead = self._resolve_lookahead(robot_x, robot_y, target_x, target_y)
        carrot = self._pick_carrot(robot_x, robot_y, lookahead)
        if carrot is None:
            self._publish_twist(0.0, 0.0)
            return
        carrot_x, carrot_y, carrot_index = carrot

        # 5.5) 卡死检测 + 倒车自救：仅在正常行进状态下参与判定（reached/零速时上面已 return）。
        if self._update_stuck_and_maybe_reverse(robot_x, robot_y):
            return

        err = _normalize_angle(math.atan2(carrot_y - robot_y, carrot_x - robot_x) - yaw)

        # 6) 生成并发布速度指令。
        angular_z = max(-self._max_angular_vel, min(self._max_angular_vel,
                                                    self._k_angular * err))
        speed_scale = max(0.0, 1.0 - abs(err) / math.pi)
        linear_x = self._max_linear_vel * speed_scale
        # 减速段：按到最终目标（target_dist 已在上文算过）的距离限速。
        distance_scale = min(1.0, target_dist / max(self._slowdown_dist, 1e-3))
        linear_x = min(linear_x, self._max_linear_vel * distance_scale)
        # 最小线速度下限：阿克曼底盘不能原地转向，必须保留线速度才能转向（目标在正后方时尤为关键）。
        linear_x = max(self._min_linear_vel, linear_x)

        # 6.5) 曲率限速（新增）：从当前前视索引往前取约 0.5m 路径段，用 _path_yaw 的
        #      相邻差估计局部曲率 kappa = |wrap(yaw[j]-yaw[i])| / ds，
        #      v_curv = sqrt(a_lat_max / kappa)，再把线速度钳到 [traj_v_min, traj_v_max]。
        #      注意：倒车段（轨迹里的 v<0）不经过 Path 消息传递，倒车只由现有脱困/自救
        #      状态机负责，这里只处理前进限速。
        if (
            carrot_index >= 0
            and self._path_s is not None
            and self._path_yaw is not None
            and len(self._path_s) == len(self._path_yaw)
        ):
            linear_x = apply_curvature_speed_limit(
                linear_x,
                self._path_s,
                self._path_yaw,
                carrot_index,
                self._a_lat_max,
                self._traj_v_min,
                self._traj_v_max,
            )

        # 7) 可达曲率钳制：必须在所有限速逻辑之后、发布之前。
        #    插件按 steer = atan(L*w/v) 反解转角，|w| 超过 v*tan(max_steer)/L 时转角被
        #    max_steer 截断，实际曲率到不了指令曲率 -> 横向误差不收敛 -> 满舵绕圈。
        angular_z = self._apply_curvature_guard(linear_x, angular_z)
        self._publish_twist(linear_x, angular_z)

    # ------------------------------------------------------------------ 生命周期
    def destroy_node(self) -> bool:
        # 退出前补发一次零 Twist，确保底盘停下。
        try:
            self._publish_twist(0.0, 0.0)
        except Exception:
            pass
        return super().destroy_node()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
