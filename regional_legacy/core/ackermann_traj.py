"""阿克曼底盘轨迹生成：把 A* 的栅格折线变成车真正开得过去的等弧长轨迹。

为什么需要这个模块（设计依据）
--------------------------------
A* 输出的是**栅格折线**，与"阿克曼车能开的轨迹"差四件事：

1. 曲率不可行：折线转角处曲率无穷大，而车有最小转弯半径
   ``R_min = L / tan(max_steer)``。本项目真机 ``L = 0.144 m``、``max_steer = 0.489 rad``
   → 几何值 ``R_min = 0.2706 m``、``kappa_max = 3.695 1/m``。**默认**取真机固件记录的
   保守值 ``min_turn_radius_m = 0.35``（``kappa_max = 2.857``）；需要几何值时显式传 0.0。
2. 折角只有 90°/45°（4/8 邻接），车需要连续转向。
3. 没有速度剖面：弯道必须减速（``v <= sqrt(a_lat / kappa)``）。
4. 起步姿态不匹配：车头朝东、路径首段朝西时，阿克曼**不能原地转**，只能开弧线或
   三点掉头 —— 这正是历史"目标在背后 → 满舵绕圈"的根因。

本模块的流水线（与需求逐步对应）：
  步骤 1  8 邻接 + 未知格硬障碍 + 膨胀的 A*；
  步骤 2  Bresenham 视线捷径（整条线满足安全距离、不穿越未知格）；
  步骤 3  圆弧倒角（**构造上保证曲率可行**：只用半径 >= R_min 的圆弧替换折角）；
  步骤 4  按等弧长 ``ds_m`` 重采样，直线段 ``kappa = 0``、圆弧段 ``kappa = sign(phi)/R``；
  步骤 5  起步对齐：先试纯前进弧，失败再试三点掉头（可倒车）；
  步骤 6  曲率限速 + 纵向加速度前后扫描；
  步骤 7  全局安全/曲率复核。

车辆模型约定（控制器依赖，务必一致）
------------------------------------
后轴中心为参考点的运动学自行车模型::

    kappa   = tan(delta) / L            # delta 为前轮转角, |delta| <= max_steer
    yaw_rate= v * kappa                 # v 为带符号线速度, v < 0 表示倒车
    yaw_{i+1} = yaw_i + v_sign * |ds| * kappa_i

**倒车段为什么 kappa 必须与 v 反号**：
  倒车时 ``v < 0``，若希望 yaw 继续朝同一方向累积（例如三点掉头里每段都朝 -dyaw
  方向转），则 ``yaw_rate = v * kappa`` 必须保持该符号，因此 ``kappa`` 与 ``v`` 反号。
  本模块把 ``kappa`` 定义为**物理转向曲率**（控制器直接 ``tan(delta)=L*kappa``），
  倒车机动段用 ``kappa = -sign(目标转角)/R``。

**倒车段为什么不参与 yaw 积分方向翻转**：
  几何路径本身按"从起点开向终点"的方向生成；倒车机动是**额外前置**在几何路径之前
  的另一段运动，拼接处 yaw 连续。几何路径上的 yaw 始终按 ``yaw += kappa*ds`` 正向
  积分；倒车只在机动段通过 ``v_sign = -1`` 体现。

坐标换算完全由调用方传入的 callable 决定，模块内部**不依赖 ROS / map origin**，
只用栅格做碰撞、捷径与未知格判断，可离线单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from core.a_star import AStarPlanner
from core.obstacle_distance import distance_to_obstacle


# ---------------------------------------------------------------------------
# 配置与输出
# ---------------------------------------------------------------------------


@dataclass
class AckermannTrajConfig:
    """轨迹生成参数。

    默认值**安全优先**（与项目安全口径对齐）：

      * ``safety_cells = 5.0``：0.25 m / 0.05 m = 5 格，等于
        ``regional_legacy/config.py::SAFE_ENVELOPE_CELLS``。复杂建筑场景若确需更贴近
        墙面，可由调用方（launch/节点参数）显式传 ``3.0`` 覆盖；不要直接改默认值。
      * ``min_turn_radius_m = 0.35``：真机固件记录的保守最小转弯半径，
        对应 ``kappa_max = 1/0.35 = 2.857 1/m``。若想使用纯几何值
        ``L/tan(max_steer) = 0.2706 m``（``kappa_max = 3.695``），显式传 ``0.0``。

    车辆几何基准（仅用于解释）：轴距 0.144 m、前轮最大转角 0.489 rad(28°)。
    """

    wheelbase_m: float = 0.144          # 轴距 L
    max_steer_rad: float = 0.489        # 前轮最大转角
    min_turn_radius_m: float = 0.35     # 安全优先：真机固件保守值；传 0.0 用 L/tan(max_steer)
    res_m: float = 0.05                 # 栅格分辨率
    safety_cells: float = 5.0           # 轨迹点到最近障碍的最小栅格距离（对齐 SAFE_ENVELOPE_CELLS）
    ds_m: float = 0.05                  # 轨迹重采样步长（圆弧另有角步长上限）
    fillet_step_rad: float = 0.15       # 倒角圆弧的离散角步长（还会被 _MAX_ARC_ANGLE_STEP_RAD 收紧）
    align_tol_rad: float = 0.0175       # 1°：仅容忍数值噪声；超过就必须生成对齐弧
    align_thresh_rad: float = 1.0472    # 已废弃：保留字段仅为兼容旧调用，不参与任何判定
    allow_reverse_align: bool = True    # 起步对齐是否允许倒车（三点掉头）
    v_max: float = 0.30
    v_min: float = 0.08
    a_lat_max: float = 0.60             # 横向加速度上限（决定弯道限速）
    a_long_max: float = 0.25            # 纵向加速度上限（速度剖面前后扫描）
    max_reverse_len_m: float = 0.60     # 三点掉头中倒车段的最大弧长
    shortcut_max_iters: int = 200


@dataclass
class Trajectory:
    """等弧长离散的阿克曼轨迹。所有数组长度一致（N >= 1；失败时 N == 0）。

    失败时各数组一律为空数组（``np.zeros(0)``），绝不再返回"部分轨迹"，避免调用方
    按长度判断失败时把不安全轨迹拿去执行。
    """

    x: np.ndarray          # (N,) 世界坐标米
    y: np.ndarray
    yaw: np.ndarray        # (N,) 切线朝向（含倒车段的实际朝向）
    kappa: np.ndarray      # (N,) 物理转向曲率 1/m（yaw_rate = v*kappa）
    v: np.ndarray          # (N,) 带符号速度 m/s（负=倒车；换向点/首尾为 0）
    s: np.ndarray          # (N,) 累计弧长（按采样顺序累加 |ds|）
    feasible: bool         # 是否满足全部曲率/碰撞约束
    reason: str            # 失败原因（成功时为空串）
    warnings: List[str] = field(default_factory=list)  # 非致命提示（如起点在安全包络内）


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

_EPS = 1e-9

# 圆弧构建余量（栅格）。轨迹首点会被平移到车辆真实位置 / 机动终点，与栅格中心
# 最多差半格对角线（约 0.71 格）。圆弧是路径中唯一"主动向障碍内侧切"的部分，
# 若恰好压在 safety_cells 边界上，这半步平移就会把它压到阈值以下。
# 因此圆弧按 safety_cells + 0.5 检查（单次流程内的构建余量，不再依赖"整体失败后
# 把 safety_cells+1 重试"这种翻倍开销）；直线段仍沿用捷径的安全检查结果。
_ARC_BUILD_MARGIN_CELLS = 0.5

# 圆弧采样的**角步长上限**（弧度）。验收要求相邻点弦向与点航向的偏差 <= 0.05 rad；
# 弦向对应圆弧区间的中点切线，与端点切线的差约为角步长的一半，因此角步长必须
# <= 0.1。这里取 0.07 留出数值余量（对应 R_min=0.35 时弧长步长约 0.0245 m）。
_MAX_ARC_ANGLE_STEP_RAD = 0.07

# ---- 倒角鲁棒性（问题 4）----
# 半径候选档数：从 2.0*R_min 线性降到 1.0*R_min（含两端），共 8 档。
# 旧实现只有 4 档（2.0/1.5/1.2/1.0），在窄通道里常出现"大半径放不下、下一档直接
# 跳到 1.0*R_min 又因离散采样/切点预算不通过"的跳档失败；加密档位可以更细地
# 匹配可用切点长度，减少"折点 N 无法在安全区内倒角"。
_FILLET_RADIUS_LEVELS = 8
# 折点外移步长（格）：0.5 格比整格更细，窄通道里"整格会撞墙、半格刚好"很常见。
_OUTWARD_STEP_CELLS = 0.5
# 单个折点的外移次数上限（旧实现 10 次）。
_MAX_OUTWARD_MOVES = 20


class _StartZone:
    """起点本体占位圈（世界坐标，修复 A）。

    机器人物理上占据以真实起点为圆心、``radius_m`` 为半径的圆。建图抖动/贴墙时，
    这个圈内的格子常被判成障碍或落在障碍膨胀区里；A* 已用 ``start_clear_cells``
    清空同一圈的阻挡（robot footprint clearing）。轨迹复核如果仍按普通安全距离
    检查圈内采样点，就会出现"A* 能算出来、最终复核又把整条轨迹判死"的矛盾。

    因此本类只做一件事：判断某个世界坐标点是否落在本体占位圈内，供
    ``_point_safe`` / ``_line_safe_cells`` 等复核函数豁免**安全距离与障碍**判定。

    **未知格绝不能豁免**：未知区域本来就不允许穿越，这条硬约束在任何情况下成立
    （调用方必须先判 ``explored != -1``）。
    """

    __slots__ = ("x", "y", "radius_m")

    def __init__(self, x: float, y: float, radius_m: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.radius_m = max(float(radius_m), 0.0)

    def contains(self, x: float, y: float) -> bool:
        """世界坐标点是否在本体占位圈内（欧氏距离，含边界）。"""
        if self.radius_m <= 0.0:
            return False
        return math.hypot(float(x) - self.x, float(y) - self.y) <= self.radius_m + 1e-9


def _start_clear_cells(config: "AckermannTrajConfig") -> int:
    """本体占位圈半径（栅格）= ceil(safety_cells) + 1，与 A* 的 start_clear_cells 同口径。

    安全膨胀是 ceil(safety_cells)（构建余量另算），本体再外扩 1 格：车体中心在
    safety 边界上时，整车仍会压到膨胀区里，多留一格给建图抖动/离散化误差。
    """
    return int(math.ceil(float(config.safety_cells))) + 1


def _start_zone_from_pose(
    start_pose_world: tuple, config: "AckermannTrajConfig"
) -> _StartZone:
    """按真实起点位姿与配置构造本体占位圈（半径 = start_clear_cells * res_m）。"""
    radius_m = float(_start_clear_cells(config)) * float(config.res_m)
    return _StartZone(float(start_pose_world[0]), float(start_pose_world[1]), radius_m)


def _wrap_angle(angle: float) -> float:
    """把角度归一化到 (-pi, pi]，用于计算两个朝向之间的最短夹角。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _build_safety_cells(config: "AckermannTrajConfig") -> float:
    """构建阶段的栅格安全距离 = ``safety_cells + 0.5``（H4）。

    为什么不直接用 ``safety_cells`` 构建：倒角圆弧会**向转角内侧切**，
    直边若恰好压在安全边界上，圆弧就没有任何余量（H2 的 300x300 墙场景就是
    这样整条失败）。构建时多留 0.5 格，最后再按 ``safety_cells`` 做一次严格复核，
    既不会降低最终安全口径，也不需要用"整体失败后 safety_cells+1 重试"的翻倍开销。
    """
    return float(config.safety_cells) + float(_ARC_BUILD_MARGIN_CELLS)


def resolve_min_turn_radius(config: AckermannTrajConfig) -> float:
    """返回实际使用的最小转弯半径 R_min。

    ``min_turn_radius_m > 0`` 时直接用它（真机固件口径更保守），否则按
    运动学 ``R_min = L / tan(max_steer)`` 计算。
    """
    if config.min_turn_radius_m is not None and config.min_turn_radius_m > 0.0:
        return float(config.min_turn_radius_m)
    steer = float(config.max_steer_rad)
    if steer <= 0.0:
        return float("inf")
    return float(config.wheelbase_m) / math.tan(steer)


def resolve_max_curvature(config: AckermannTrajConfig) -> float:
    """返回允许的最大曲率 ``kappa_max = 1 / R_min``。"""
    radius = resolve_min_turn_radius(config)
    if not np.isfinite(radius) or radius <= 0.0:
        return 0.0
    return 1.0 / radius


def _norm(vx: float, vy: float) -> Tuple[float, float, float]:
    """二维向量归一化，返回 (ux, uy, 长度)；零向量返回 (0, 0, 0)。"""
    length = math.hypot(vx, vy)
    if length < _EPS:
        return 0.0, 0.0, 0.0
    return vx / length, vy / length, length


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """整数 Bresenham 直线，返回连线经过的所有栅格（含两端点）。"""
    points: List[Tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


def _to_grid_rc(world_to_grid_rc: Callable, x: float, y: float) -> Tuple[int, int]:
    """调用方的世界->栅格换算，四舍五入到整数格。"""
    r, c = world_to_grid_rc(float(x), float(y))
    return int(round(float(r))), int(round(float(c)))


# ---------------------------------------------------------------------------
# 安全判定（障碍距离 + 未知格）
# ---------------------------------------------------------------------------


def _point_safe(
    x: float,
    y: float,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    safety_cells: float,
    start_zone: Optional[_StartZone] = None,
) -> bool:
    """单个世界坐标点是否安全。

    判定三件事：在地图内、所在格不是未知(-1)、到最近障碍的栅格距离 >= safety_cells。
    注意这里刻意**不**把"到未知格的距离"当安全距离：探索任务的目标就是未知前沿，
    若要求离未知 >= safety_cells，任何前沿都不可达。未知格只做"不可进入"的硬判定。

    ``start_zone`` 非空时（修复 A）：落在机器人本体占位圈内的点豁免"障碍 / 安全距离"
    检查（本体占位本就是自由，建图抖动/贴墙时常被误判），但**未知格仍然禁止**。
    """
    r, c = _to_grid_rc(world_to_grid_rc, x, y)
    h, w = explored.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    # 未知格是硬约束：任何豁免都不能放开这一条。
    if explored[r, c] == -1:
        return False
    # 本体占位圈内：障碍/膨胀误判一律豁免（圈外的检查不受影响）。
    if start_zone is not None and start_zone.contains(x, y):
        return True
    if explored[r, c] == 1:
        return False
    return float(distance[r, c]) >= float(safety_cells)


def _line_safe_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    explored: np.ndarray,
    distance: np.ndarray,
    safety_cells: float,
    dense_step_cells: float = 0.25,
    start_zone: Optional[_StartZone] = None,
    grid_to_world_xy: Optional[Callable] = None,
) -> bool:
    """视线捷径的整条连线是否安全。

    两层检查：
      1. Bresenham 经过的每个栅格都安全；
      2. **连续直线细采样复核**（0.25 格步长）。原因：Bresenham 只保证"经过的格子"
         安全，但两格中心之间的连续直线可能贴着障碍角切过去——当路径恰好以
         ``dist == safety_cells`` 绕过凸角时，这个差异会让最终轨迹侵入安全区。
         细采样把"栅格安全"提升为"连续线段安全"，与步骤 7 的全局复核口径一致。

    修复 A：传入 ``start_zone`` 与 ``grid_to_world_xy`` 时，落在本体占位圈内的格子
    豁免障碍/安全距离检查（未知格仍硬禁止）。否则起点贴墙时视线捷径的第一段必然
    不满足安全距离，整条路线会被误判为 "A* 路径不满足安全距离"。
    """
    h, w = explored.shape

    def cell_ok(r: int, c: int) -> bool:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if explored[r, c] == -1:
            return False
        if start_zone is not None and grid_to_world_xy is not None:
            gx, gy = grid_to_world_xy(r, c)
            if start_zone.contains(gx, gy):
                return True
        if explored[r, c] == 1:
            return False
        return float(distance[r, c]) >= float(safety_cells)

    for r, c in _bresenham(r0, c0, r1, c1):
        if not cell_ok(r, c):
            return False

    length = math.hypot(r1 - r0, c1 - c0)
    n = max(1, int(math.ceil(length / max(dense_step_cells, _EPS))))
    for k in range(n + 1):
        t = k / n
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        if not cell_ok(r, c):
            return False
    return True


# ---------------------------------------------------------------------------
# 轨迹片段累加器
# ---------------------------------------------------------------------------


class _PathBuilder:
    """按顺序累积采样点，并记录每个**区间**（i -> i+1）的曲率/弧长/运动方向。

    为什么要按区间记录而不是按点记录：
      速度剖面与 yaw 积分都作用在相邻点之间。把 ``kappa/ds/v_sign`` 绑定到区间上，
      可以精确保证 ``yaw_{i+1} = yaw_i + v_sign * ds * kappa`` 逐区间成立；拼接处
      只共享一个点、不跨越两个片段，因此不会出现"一个区间跨两种曲率"的误差。
    """

    def __init__(self) -> None:
        self.x: List[float] = []
        self.y: List[float] = []
        self.kappa: List[float] = []   # 区间曲率（物理转向曲率）
        self.dsl: List[float] = []     # 区间弧长（正数）
        self.vsign: List[float] = []   # 区间运动方向：+1 前进 / -1 倒车

    @property
    def n(self) -> int:
        return len(self.x)

    def push(self, x: float, y: float) -> None:
        self.x.append(float(x))
        self.y.append(float(y))

    def _commit(self, x: float, y: float, kappa: float, dsl: float, vsign: float) -> None:
        """追加一个新点，并把区间参数记到"当前最后一点"上。"""
        self.kappa.append(float(kappa))
        self.dsl.append(float(dsl))
        self.vsign.append(float(vsign))
        self.push(x, y)

    def add_line_to(self, x1: float, y1: float, step: float) -> None:
        """从当前最后一点沿直线走到 (x1, y1)，按 step 等分，kappa=0。"""
        x0, y0 = self.x[-1], self.y[-1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < _EPS:
            return
        n = max(1, int(math.ceil(length / max(step, _EPS))))
        ds = length / n
        for k in range(1, n + 1):
            t = k / n
            self._commit(x0 + t * dx, y0 + t * dy, 0.0, ds, +1.0)

    def add_arc_center(
        self,
        cx: float,
        cy: float,
        radius: float,
        a0: float,
        sweep: float,
        step_rad: float,
        step_ds: float,
    ) -> None:
        """沿给定圆心/半径的圆弧采样（前进方向），kappa = sign(sweep)/R。

        这是几何路径的倒角圆弧：圆弧由构造保证半径 >= R_min，因此曲率天然可行。
        """
        if radius <= 0.0 or abs(sweep) < _EPS:
            return
        kappa = math.copysign(1.0, sweep) / radius
        n = max(
            1,
            int(math.ceil(abs(sweep) / max(step_rad, _EPS))),
            # 角步长上限：保证弦向（区间中点切线）与端点切线的差 <= 0.035 rad。
            int(math.ceil(abs(sweep) / _MAX_ARC_ANGLE_STEP_RAD)),
            int(math.ceil(radius * abs(sweep) / max(step_ds, _EPS))),
        )
        ds = radius * abs(sweep) / n
        for k in range(1, n + 1):
            angle = a0 + sweep * k / n
            self._commit(
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle),
                kappa,
                ds,
                +1.0,
            )

    def add_arc_motion(
        self,
        pose: Tuple[float, float, float],
        delta: float,
        radius: float,
        v_sign: float,
        step: float,
    ) -> None:
        """从 pose 出发做一段"航向变化 delta、半径 radius"的运动弧。

        delta 是**航向变化**（带符号），不含倒车方向；运动方向由 v_sign 决定：

          * 前进 (v_sign=+1)：位移方向 = 起始航向 + delta/2；
          * 倒车 (v_sign=-1)：位移方向 = 起始航向 + pi + delta/2。

        物理转向曲率 ``kappa = delta / (v_sign * R * |delta|)``，因此倒车段的 kappa
        与 v 反号，保证 ``yaw += v_sign*ds*kappa = ds*delta/L`` 朝目标方向累积。
        """
        if abs(delta) < _EPS or radius <= 0.0:
            return
        x0, y0, yaw0 = pose
        kappa = delta / (v_sign * radius * abs(delta))
        length = radius * abs(delta)
        n = max(
            1,
            int(math.ceil(length / max(step, _EPS))),
            # 与倒角圆弧同样收紧角步长，保证离散弦向与航向自洽。
            int(math.ceil(abs(delta) / _MAX_ARC_ANGLE_STEP_RAD)),
        )
        ds = length / n
        for k in range(1, n + 1):
            du = delta * k / n
            # 圆弧位移大小 = 2R sin(|du|/2)，方向为平均速度方向。
            mag = 2.0 * radius * math.sin(abs(du) * 0.5)
            ang = yaw0 + du * 0.5 + (0.0 if v_sign > 0 else math.pi)
            self._commit(
                x0 + mag * math.cos(ang),
                y0 + mag * math.sin(ang),
                kappa,
                ds,
                v_sign,
            )

    def extend(self, other: "_PathBuilder") -> None:
        """把另一段轨迹接到当前末尾（要求首点与当前末点重合，调用方保证）。"""
        if other.n == 0:
            return
        if self.n == 0:
            self.x = list(other.x)
            self.y = list(other.y)
            self.kappa = list(other.kappa)
            self.dsl = list(other.dsl)
            self.vsign = list(other.vsign)
            return
        # 跳过重合的首点，区间参数整体保留。
        self.x.extend(other.x[1:])
        self.y.extend(other.y[1:])
        self.kappa.extend(other.kappa)
        self.dsl.extend(other.dsl)
        self.vsign.extend(other.vsign)

    def point_arrays(self):
        """返回点坐标与"按点"的区间参数（最后一点沿用最后一个区间）。"""
        n = self.n
        kappa_pt = np.zeros(n, dtype=float)
        vsign_pt = np.ones(n, dtype=float)
        if n >= 2:
            kappa_pt[:-1] = np.asarray(self.kappa, dtype=float)
            kappa_pt[-1] = self.kappa[-1]
            vsign_pt[:-1] = np.asarray(self.vsign, dtype=float)
            vsign_pt[-1] = self.vsign[-1]
        ds_int = np.asarray(self.dsl, dtype=float)
        return (
            np.asarray(self.x, dtype=float),
            np.asarray(self.y, dtype=float),
            kappa_pt,
            vsign_pt,
            ds_int,
        )


# ---------------------------------------------------------------------------
# 步骤 1：A* 栅格路径
# ---------------------------------------------------------------------------


def _plan_grid_path(
    explored: np.ndarray,
    start_rc,
    goal_rc,
    config,
    planner,
    distance_field: Optional[np.ndarray] = None,
):
    """8 邻接 + 未知格硬障碍 + 障碍膨胀的 A*。

    未知格当硬障碍是安全的：上游候选前沿的可达性只在已知自由格上算
    （`real_work/light_qifa.py::_reachable_distances`），因此不会把合法前沿判死。

    `distance_field` 一律透传给 A*（H4）：同一次轨迹生成里 A* 可能被调用多次
    （起步机动后重规划），复用同一份距离场，避免重复做整图距离变换。
    """
    if planner is None:
        planner = AStarPlanner()
    return planner.plan(
        explored,
        start_rc,
        goal_rc,
        # 构建余量取整到上界（5.0 -> 6 格），保证捷径至少能保留 5.5 格的连续间隙。
        inflation_radius_cells=int(math.ceil(_build_safety_cells(config))),
        use_eight_neighbors=True,
        treat_unknown_as_obstacle=True,
        distance_field=distance_field,
        # 修复 A（核心）：清空"机器人本体占位圈"（安全膨胀 + 1 格）。
        # 轨迹模块的 A* 起点就是机器人所在格，一旦该格被膨胀/未知挡住，A* 会直接
        # 返回 None（实测 201/533 的回退原因）。机器人物理上占着这一圈，清空它
        # 是 costmap 的标准做法，不会放宽圈外的安全距离。
        start_clear_cells=_start_clear_cells(config),
    )


# ---------------------------------------------------------------------------
# 步骤 2：视线捷径
# ---------------------------------------------------------------------------


def _shortcut_grid_path(
    path,
    explored,
    distance,
    safety_cells,
    max_iters,
    start_zone: Optional[_StartZone] = None,
    grid_to_world_xy: Optional[Callable] = None,
):
    """用 Bresenham 视线捷径压缩栅格折线。

    为什么用栅格而不是世界坐标判定：碰撞与安全距离本来就是栅格度量；把候选端点
    仍限制在 A* 的栅格点上，可以保证捷径结果不会因为坐标换算的舍入而"擦边"。

    迭代上限由 `shortcut_max_iters` 控制；若某个点无法直达更远点，则退化为相邻点
    （相邻点仍不安全时说明 A* 路径本身贴障碍，返回 None）。

    **迭代预算按"每个接受段"独立计算**：旧实现用全局计数器，长路径上一旦累计
    超过上限，后续所有段都永久退化为逐格 ``i+1``，在转角后留下大量 5 cm 共线点
    和 1 格阶梯连接（H2 的根因之一）。按段独立预算既保留总工作量可控，
    又不会让一次局部搜索拖垮整条路线。
    """
    n = len(path)
    if n <= 1:
        return [(int(p[0]), int(p[1])) for p in path]

    accepted = [0]
    i = 0

    while i < n - 1:
        found: Optional[int] = None
        # 每个接受段重新获得一份预算。
        iters = 0
        for j in range(n - 1, i, -1):
            iters += 1
            if _line_safe_cells(
                int(path[i][0]), int(path[i][1]),
                int(path[j][0]), int(path[j][1]),
                explored, distance, safety_cells,
                start_zone=start_zone,
                grid_to_world_xy=grid_to_world_xy,
            ):
                found = j
                break
            if iters >= int(max_iters):
                break
        if found is None:
            if not _line_safe_cells(
                int(path[i][0]), int(path[i][1]),
                int(path[i + 1][0]), int(path[i + 1][1]),
                explored, distance, safety_cells,
                start_zone=start_zone,
                grid_to_world_xy=grid_to_world_xy,
            ):
                return None
            found = i + 1
        accepted.append(found)
        i = found

    return [(int(path[k][0]), int(path[k][1])) for k in accepted]


# ---------------------------------------------------------------------------
# 步骤 2.5：共线点合并（必须在倒角之前）
# ---------------------------------------------------------------------------


def _merge_collinear_points(
    points: List[Tuple[float, float]],
    cross_tol: float = 1e-6,
) -> List[Tuple[float, float]]:
    """删除"同向共线"的中间折点（H2）。

    为什么必须做：视线捷径在某段退化成逐格 ``i -> i+1`` 后，转角之后会留下一串
    5 cm 的共线点。每个共线点在倒角预算里只贡献 1 格边长，导致紧跟着的真实折角
    连 ``R_min`` 的切点都放不下，整条普通路线被误判为"折点在安全距离内无法倒角"。
    先把同向共线的中间点删掉，让真实折角两侧获得完整边长。

    判定必须**同时**满足：
      * ``cross(u_in, u_out) ≈ 0``：方向平行；
      * ``dot(u_in, u_out) > 0``：方向相同。
    只判 cross 会把 180° 回折（U 形掉头）也合并掉，必须靠 dot 保留回折点。
    """
    if len(points) <= 2:
        return list(points)

    merged: List[Tuple[float, float]] = [points[0]]
    for i in range(1, len(points) - 1):
        px, py = merged[-1]
        ix, iy = points[i]
        nx, ny = points[i + 1]

        uix, uiy, lin = _norm(ix - px, iy - py)
        uox, uoy, lout = _norm(nx - ix, ny - iy)
        if lin < _EPS or lout < _EPS:
            # 重复点：直接丢弃中间点。
            continue

        cross = uix * uoy - uiy * uox
        dot = uix * uox + uiy * uoy
        if abs(cross) <= cross_tol and dot > 0.0:
            # 同向共线：points[i] 不是折点，删除。
            continue
        merged.append((float(ix), float(iy)))

    merged.append(points[-1])
    return merged


# ---------------------------------------------------------------------------
# 步骤 3：圆弧倒角
# ---------------------------------------------------------------------------


def _arc_cells_safe(
    cx: float,
    cy: float,
    radius: float,
    a0: float,
    sweep: float,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    safety_cells: float,
    step_rad: float,
    start_zone: Optional[_StartZone] = None,
) -> bool:
    """按角步长采样整段圆弧，逐点检查安全（含两端切点）。

    ``start_zone`` 非空时圆内点豁免安全距离（修复 A，仍禁止未知格）。
    """
    n = max(1, int(math.ceil(abs(sweep) / max(step_rad, _EPS))))
    for k in range(n + 1):
        angle = a0 + sweep * k / n
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        if not _point_safe(
            x, y, explored, distance, world_to_grid_rc, safety_cells,
            start_zone=start_zone,
        ):
            return False
    return True


def _segment_safe_world(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    safety_cells: float,
    step_m: float,
    start_zone: Optional[_StartZone] = None,
) -> bool:
    """连续直线段（世界坐标）逐点安全检查。

    用于折点外移后的相邻直边复检（M3）：外移折点会改变两侧直线段的位置，
    只复检圆弧是不够的——外移方向恰好可能把直线段推进障碍/未知格。

    ``start_zone`` 非空时圆内点豁免安全距离（修复 A，仍禁止未知格）。
    """
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(math.ceil(length / max(step_m, _EPS))))
    for k in range(n + 1):
        t = k / n
        if not _point_safe(
            x0 + (x1 - x0) * t,
            y0 + (y1 - y0) * t,
            explored, distance, world_to_grid_rc, safety_cells,
            start_zone=start_zone,
        ):
            return False
    return True


def _fillet_once(
    points: List[Tuple[float, float]],
    config: AckermannTrajConfig,
    r_min: float,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    start_zone: Optional[_StartZone] = None,
):
    """尝试给所有内部折点倒角。成功返回 (builder, None)，失败返回 (None, 折点下标)。

    **构造上保证曲率可行**：每个折点只用半径序列
    ``linspace(2R_min, R_min, 8)``（从大往小试，含两端）中的某个圆弧替换，
    圆弧曲率恒为 ``sign(phi)/R``，因此 ``|kappa| <= 1/R_min = kappa_max``。
    不存在"先连折线再平滑"的近似步骤，曲率上限是数学上保证的，而不是事后
    检查出来的。候选从 4 档加密到 8 档（问题 4）：窄通道里可用切点预算往往
    只落在两档之间，粗档位会直接跳档失败。8 档贪心失败时再按旧 4 档完整重跑
    一次作为保底（旧 4 档是 8 档的子集），避免"加密候选反而多占共享直边、
    饿死下一折点"的回归。

    倒角几何（务必按叉积符号取圆心方向）::

        phi   = atan2(cross(u_in, u_out), dot(u_in, u_out))
        t     = R * tan(|phi|/2)
        T1    = Pi - t*u_in
        T2    = Pi + t*u_out
        b     = normalize(u_out - u_in)          # 指向转角内侧
        C     = Pi + (R / cos(|phi|/2)) * b
        arc   = 从 T1 绕 C 到 T2，扫角 phi，kappa = sign(phi)/R

    半径从大往小试，并保证相邻两个倒角在同一条直边上的切点长度之和不超过该边长
    （``t_prev + t_i <= 边长``）。这样既不会让两个圆弧重叠，也不会像"每角最多占用
    一半"那样，在绕凸角的 U 形里把本来装得下的两个倒角误判为不可行。
    所有半径都不满足时由调用方沿角平分线外移折点。
    """
    if len(points) < 3:
        builder = _PathBuilder()
        builder.push(points[0][0], points[0][1])
        builder.add_line_to(points[-1][0], points[-1][1], config.ds_m)
        return builder, None

    # 预计算每个内部折点用 R_min 倒角时**至少**要占用的切点长度 t_min。
    # 相邻两个倒角共享同一条直边，必须把这条边的长度按 t_prev + t_i <= 边长来分配；
    # 只给每个角固定"不超过一半"会在两个折点靠得很近（例如绕凸角的 U 形）时误判
    # "无法倒角"——实际上两角的切点之和可能仍然装得下。
    t_min = [0.0] * len(points)
    for j in range(1, len(points) - 1):
        ux0, uy0, l0 = _norm(points[j][0] - points[j - 1][0], points[j][1] - points[j - 1][1])
        ux1, uy1, l1 = _norm(points[j + 1][0] - points[j][0], points[j + 1][1] - points[j][1])
        if l0 < _EPS or l1 < _EPS:
            continue
        phi_j = math.atan2(ux0 * uy1 - uy0 * ux1, ux0 * ux1 + uy0 * uy1)
        if abs(phi_j) >= 1e-3:
            t_min[j] = r_min * math.tan(abs(phi_j) * 0.5)

    # 半径候选：2.0*R_min -> 1.0*R_min 线性降（含两端，共 8 档），从大往小试。
    # 用 set 去重只是防御性写法（linspace 各档互不相同）。
    radii = sorted(
        {float(scale) * r_min
         for scale in np.linspace(2.0, 1.0, _FILLET_RADIUS_LEVELS)},
        reverse=True,
    )
    builder, _ = _fillet_pass(
        points, config, t_min, radii, explored, distance, world_to_grid_rc,
        start_zone=start_zone,
    )
    if builder is not None:
        return builder, None

    # 加密档位的**保底**：8 档失败时再按旧的 4 档（2.0/1.5/1.2/1.0）完整重跑一次。
    # 为什么需要：8 档里更大的中间半径可能通过本折点的检查，却多占了与下一折点
    # 共享的直边，使下一个折点预算不足（实测 U/Z 形窄通道回归）。用旧 4 档重跑
    # 可以保证"加密候选"绝不会让原本能倒角的路线反而失败（旧 4 档是 8 档的子集，
    # 且档位顺序不变，因此旧实现能成功的情形在这里一定能成功）。
    legacy_radii = sorted(
        {2.0 * r_min, 1.5 * r_min, 1.2 * r_min, 1.0 * r_min}, reverse=True,
    )
    return _fillet_pass(
        points, config, t_min, legacy_radii, explored, distance, world_to_grid_rc,
        start_zone=start_zone,
    )


def _fillet_pass(
    points: List[Tuple[float, float]],
    config: AckermannTrajConfig,
    t_min: List[float],
    radii: List[float],
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    start_zone: Optional[_StartZone] = None,
):
    """按给定半径候选序列，从大往小一次性贪心倒角（`_fillet_once` 的单次实现）。

    返回 ``(builder, None)`` 成功，``(None, 折点下标)`` 失败。
    """
    builder = _PathBuilder()
    builder.push(points[0][0], points[0][1])
    used_prev = 0.0   # 上一个折角在这条入边上已经占用的长度

    for i in range(1, len(points) - 1):
        px, py = points[i - 1]
        ix, iy = points[i]
        nx, ny = points[i + 1]

        uix, uiy, lin = _norm(ix - px, iy - py)
        uox, uoy, lout = _norm(nx - ix, ny - iy)
        if lin < _EPS or lout < _EPS:
            # 重复点：不加线段，直接看下一个折点。
            continue

        cross = uix * uoy - uiy * uox
        dot = uix * uox + uiy * uoy
        phi = math.atan2(cross, dot)
        if abs(phi) < 1e-3:
            # 近似共线，不需要倒角。
            builder.add_line_to(ix, iy, config.ds_m)
            used_prev = 0.0
            continue

        # 角平分线方向（指向转角内侧）；零向量说明两边完全反向，跳过。
        bx, by, blen = _norm(uox - uix, uoy - uiy)
        if blen < _EPS:
            builder.add_line_to(ix, iy, config.ds_m)
            used_prev = 0.0
            continue

        cos_half = math.cos(abs(phi) * 0.5)
        # 本角可用的切点长度预算：
        #   入边剩余 = 边长 - 上一个角已占用；
        #   出边剩余 = 边长 - 下一个角用 R_min 时至少要占用的长度。
        t_budget = min(lin - used_prev, lout - t_min[i + 1])
        if t_budget < -1e-9:
            return None, i
        chosen = None
        for radius in radii:
            if radius <= 0.0:
                continue
            t = radius * math.tan(abs(phi) * 0.5)
            if t > t_budget + 1e-9:
                continue
            t1x, t1y = ix - t * uix, iy - t * uiy
            t2x, t2y = ix + t * uox, iy + t * uoy
            cx = ix + (radius / cos_half) * bx
            cy = iy + (radius / cos_half) * by
            a0 = math.atan2(t1y - cy, t1x - cx)
            # 碰撞检查比轨迹采样更密：空间步长取 1/4 格。否则圆弧最接近障碍的
            # 那一点可能落在两个采样点之间，出现"检查通过、最终轨迹压线"。
            check_step = min(
                config.fillet_step_rad,
                0.25 * config.res_m / max(radius, _EPS),
            )
            if not _arc_cells_safe(
                cx, cy, radius, a0, phi, explored, distance,
                world_to_grid_rc,
                config.safety_cells + _ARC_BUILD_MARGIN_CELLS,
                check_step,
                start_zone=start_zone,
            ):
                continue
            chosen = (radius, t1x, t1y, t2x, t2y, cx, cy, a0)
            break

        if chosen is None:
            return None, i

        radius, t1x, t1y, t2x, t2y, cx, cy, a0 = chosen
        # prev_end 一定落在 P_{i-1}->Pi 这条直线上（它是上一个角倒角的 T2），
        # 所以这里连到 T1 的线段与 u_in 共线。
        builder.add_line_to(t1x, t1y, config.ds_m)
        builder.add_arc_center(
            cx, cy, radius, a0, phi,
            config.fillet_step_rad, config.ds_m,
        )
        # 数值上强制圆弧终点等于解析切点 T2，避免累计舍入造成微小折角。
        builder.x[-1] = t2x
        builder.y[-1] = t2y
        used_prev = t   # 本角占用的出边长度，留给下一个角做入边预算

    builder.add_line_to(points[-1][0], points[-1][1], config.ds_m)
    return builder, None


def _outward_shifted_point(
    points: List[Tuple[float, float]], index: int, step_m: float
) -> Optional[Tuple[float, float]]:
    """返回把 ``points[index]`` 沿自身角平分线**向外**平移 ``step_m`` 后的坐标。

    "向外"= 离开转角内侧，即 ``-bisector`` 方向（与 `_fillet_once` 里圆弧圆心的
    方向相反）。沿该方向移动会同时**加长**相邻两条直边、并让倒角圆弧远离内侧墙，
    这正是"让形状更缓"的几何手段。

    端点（首/尾）是路线固定锚点，不参与外移，返回 None；相邻边退化（重复点）或
    两边完全反向（角平分线为零向量）时也返回 None。
    """
    if index <= 0 or index >= len(points) - 1:
        return None
    px, py = points[index - 1]
    ix, iy = points[index]
    nx, ny = points[index + 1]
    uix, uiy, lin = _norm(ix - px, iy - py)
    uox, uoy, lout = _norm(nx - ix, ny - iy)
    if lin < _EPS or lout < _EPS:
        return None
    bx, by, blen = _norm(uox - uix, uoy - uiy)
    if blen < _EPS:
        return None
    # 沿 -bisector（离开转角内侧）向外移动。
    return (ix - bx * step_m, iy - by * step_m)


def _try_outward_shift(
    points: List[Tuple[float, float]],
    index: int,
    step_m: float,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    build_safety: float,
    check_step_m: float,
    start_zone: Optional[_StartZone] = None,
) -> bool:
    """在 ``points``（通常是副本）上把 ``points[index]`` 外移 ``step_m``。

    只有**相邻两条直线段**在构建安全距离下都安全时才真正写回并返回 True；
    圆弧本身不在这一步检查（下一轮 `_fillet_once` 会统一按更密的采样复检）。
    失败时不修改 ``points``，保证调用方可以安全放弃这次尝试。
    """
    shifted = _outward_shifted_point(points, index, step_m)
    if shifted is None:
        return False
    # M3：外移会改变两侧直边位置，必须复检（可能被推进障碍/未知格）。
    if not _segment_safe_world(
        points[index - 1], shifted,
        explored, distance, world_to_grid_rc, build_safety, check_step_m,
        start_zone=start_zone,
    ):
        return False
    if not _segment_safe_world(
        shifted, points[index + 1],
        explored, distance, world_to_grid_rc, build_safety, check_step_m,
        start_zone=start_zone,
    ):
        return False
    points[index] = shifted
    return True


def _build_geometry(
    points: List[Tuple[float, float]],
    config: AckermannTrajConfig,
    r_min: float,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    start_zone: Optional[_StartZone] = None,
):
    """整体倒角构建；折点失败时沿角平分线向外移动半格后重试。

    搜索策略（问题 4：提鲁棒性；不改动整体流水线与安全口径）：
      1) **单点外移**：失败折点沿角平分线向外移动 0.5 格，单点最多 20 次；
      2) **让形状更缓**：某折点**首次**失败时，除移动它自己，还同时把相邻的
         ``i-1`` / ``i+1`` 各向外移 0.5 格（仅内部折点），整体重试一次。
         相邻折点外移会加长共享直边、并让它们自己的倒角更远离内侧墙，能救活
         "切点预算不够 / 内侧贴墙"的折点。每个失败索引只做一轮，避免组合爆炸。
      3) 若新的 0.5 格搜索失败，**再按改造前的搜索策略兜底一次**（整格外移、
         单点最多 10 次），保证加密档位/细化步长不会让原本能倒角的路线反而失败；
      4) 仍然全部失败 -> 返回失败，绝不放宽 ``safety_cells``、绝不跳过碰撞检查；
         宁可失败，也不生成不安全轨迹。

    失败返回 ``(None, 折点下标)``，reason 由调用方拼装。
    """
    work = [(float(p[0]), float(p[1])) for p in points]
    moves = [0] * len(work)
    # 每次外层循环至少推进一次（单点外移），外加每点一次"相邻外移"尝试。
    max_iter = _MAX_OUTWARD_MOVES * len(work) + len(work) + 8
    fail_index = 0
    # 已经对哪些失败索引尝试过"相邻折点一起外移"（每索引只做一轮）。
    spread_tried = set()

    step_m = float(_OUTWARD_STEP_CELLS) * float(config.res_m)
    build_safety = _build_safety_cells(config)
    check_step_m = 0.25 * float(config.res_m)

    for _ in range(max_iter):
        builder, fail_index = _fillet_once(
            work, config, r_min, explored, distance, world_to_grid_rc,
            start_zone=start_zone,
        )
        if builder is not None:
            return builder, None

        # 端点不属于可移动折点 / 单点外移次数已达上限：结束新搜索，交给旧策略兜底。
        if fail_index <= 0 or fail_index >= len(work) - 1:
            break
        if moves[fail_index] >= _MAX_OUTWARD_MOVES:
            break

        # ---- 先尝试"让形状更缓"：相邻两个折点也各向外 0.5 格（只做一轮）----
        if fail_index not in spread_tried:
            spread_tried.add(fail_index)
            candidate = list(work)
            shifted_ok = True
            # 先动相邻内部折点，再动失败折点自己（后者用更新后的邻边重算角平分线）。
            for neighbor in (fail_index - 1, fail_index + 1):
                if 0 < neighbor < len(candidate) - 1:
                    if not _try_outward_shift(
                        candidate, neighbor, step_m,
                        explored, distance, world_to_grid_rc,
                        build_safety, check_step_m,
                        start_zone=start_zone,
                    ):
                        shifted_ok = False
                        break
            if shifted_ok:
                shifted_ok = _try_outward_shift(
                    candidate, fail_index, step_m,
                    explored, distance, world_to_grid_rc,
                    build_safety, check_step_m,
                    start_zone=start_zone,
                )
            if shifted_ok:
                # 整批外移都安全：提交并重试倒角（不额外消耗"单点预算"以外的东西，
                # 但仍为每个被移动的折点记一次外移，保证总移动次数有界）。
                work = candidate
                for idx in (fail_index - 1, fail_index, fail_index + 1):
                    if 0 < idx < len(moves) - 1:
                        moves[idx] += 1
                continue
            # 相邻外移不可行（会撞墙/推进未知）：退回只移动当前折点。

        # ---- 兜底：只外移失败折点自己 ----
        if not _try_outward_shift(
            work, fail_index, step_m,
            explored, distance, world_to_grid_rc,
            build_safety, check_step_m,
            start_zone=start_zone,
        ):
            break
        moves[fail_index] += 1

    # ---- 最后兜底：完全按改造前的搜索策略再跑一次 ----
    # 为什么需要：0.5 格外移与整格外移走的折点轨迹不同，个别几何里 0.5 格会
    # 在安全区内"绕死"（每一步都合法，但收敛到无法倒角的位形），而整格步长反而
    # 能一步跳到可行位形。兜底保留旧搜索（整格外移、每点最多 10 次、无相邻扩散），
    # 保证改造后**至少不会比改造前更差**；仍只接受满足安全距离的几何，宁可失败
    # 也不放宽 safety_cells 或跳过碰撞检查。
    legacy_builder, _legacy_fail = _build_geometry_legacy(
        points, config, r_min, explored, distance, world_to_grid_rc,
        start_zone=start_zone,
    )
    if legacy_builder is not None:
        return legacy_builder, None
    return None, fail_index


def _build_geometry_legacy(
    points: List[Tuple[float, float]],
    config: AckermannTrajConfig,
    r_min: float,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    start_zone: Optional[_StartZone] = None,
):
    """改造前的倒角搜索（整格外移、单点最多 10 次、不做相邻折点扩散）。

    仅作为 `_build_geometry` 失败后的兜底，保证新的 0.5 格搜索不会让原本能倒角的
    路线反而失败。安全口径与主路径完全一致（`_build_safety_cells` + 相邻直边复检）。
    """
    work = [(float(p[0]), float(p[1])) for p in points]
    moves = [0] * len(work)
    max_iter = 10 * len(work) + len(work) + 8
    fail_index = 0

    step_m = float(config.res_m)
    build_safety = _build_safety_cells(config)
    check_step_m = 0.25 * float(config.res_m)

    for _ in range(max_iter):
        builder, fail_index = _fillet_once(
            work, config, r_min, explored, distance, world_to_grid_rc,
            start_zone=start_zone,
        )
        if builder is not None:
            return builder, None
        if fail_index <= 0 or fail_index >= len(work) - 1:
            return None, fail_index
        if moves[fail_index] >= 10:
            return None, fail_index
        if not _try_outward_shift(
            work, fail_index, step_m,
            explored, distance, world_to_grid_rc,
            build_safety, check_step_m,
            start_zone=start_zone,
        ):
            return None, fail_index
        moves[fail_index] += 1

    return None, fail_index


# ---------------------------------------------------------------------------
# 路线构建（步骤 1~3：A* -> 捷径 -> 倒角）
# ---------------------------------------------------------------------------


def _build_route(
    explored: np.ndarray,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    config: AckermannTrajConfig,
    planner,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    grid_to_world_xy: Callable,
    distance_field: Optional[np.ndarray] = None,
    start_zone: Optional[_StartZone] = None,
):
    """给定栅格起终点，产出倒角后的几何轨迹。

    返回 ``(builder, dir0, reason)``：失败时 builder 为 None；起终点同格时 builder
    只有一个点且 dir0 为 None。

    `distance_field` 透传给 A*（H4）；`distance` 用于本函数的碰撞/捷径判定。
    """
    grid_path = _plan_grid_path(
        explored, start_rc, goal_rc, config, planner,
        distance_field=distance_field,
    )
    if grid_path is None:
        return None, 0.0, "A* 无路"

    # H1：A* 在起点=终点时返回 [[sx, sy]]，每个元素必须按 (row, col) 取；
    # 单点走"原地不动"的早退分支，绝不把 grid_path[0][0] 当标量解包。
    if len(grid_path) <= 1:
        if grid_path:
            r, c = int(grid_path[0][0]), int(grid_path[0][1])
        else:
            r, c = int(start_rc[0]), int(start_rc[1])
        x, y = grid_to_world_xy(r, c)
        builder = _PathBuilder()
        builder.push(float(x), float(y))
        return builder, None, ""

    # 捷径按"构建安全距离"（safety_cells + 0.5）压缩，给倒角圆弧留余量；
    # 最终仍会按 safety_cells 做一次严格复核。
    short_path = _shortcut_grid_path(
        grid_path, explored, distance, _build_safety_cells(config),
        config.shortcut_max_iters,
        # 修复 A：起点本体占位圈内的格子豁免安全距离（未知格仍禁止），
        # 否则贴墙起步时捷径第一段必然不满足构建安全距离。
        start_zone=start_zone,
        grid_to_world_xy=grid_to_world_xy,
    )
    if short_path is None:
        return None, 0.0, "A* 路径不满足安全距离"

    points: List[Tuple[float, float]] = []
    for r, c in short_path:
        x, y = grid_to_world_xy(r, c)
        x, y = float(x), float(y)
        if not points or math.hypot(x - points[-1][0], y - points[-1][1]) > 1e-9:
            points.append((x, y))
    if len(points) < 2:
        builder = _PathBuilder()
        builder.push(points[0][0], points[0][1])
        return builder, None, ""

    # H2：倒角**之前**先合并同向共线点，避免逐格共线链把倒角预算吃光。
    points = _merge_collinear_points(points)
    if len(points) < 2:
        builder = _PathBuilder()
        builder.push(points[0][0], points[0][1])
        return builder, None, ""

    geometry, fail_index = _build_geometry(
        points, config, resolve_min_turn_radius(config),
        explored, distance, world_to_grid_rc,
        start_zone=start_zone,
    )
    if geometry is None:
        return None, 0.0, f"折点 {fail_index} 无法在安全区内倒角"

    return geometry, _first_direction(geometry, 0.0), ""


# ---------------------------------------------------------------------------
# 步骤 5：起步对齐（前进弧 / 三点掉头）
# ---------------------------------------------------------------------------


def _first_direction(builder: _PathBuilder, fallback_yaw: float) -> float:
    """取几何路径第一个非零区间的切线方向。"""
    for i in range(builder.n - 1):
        dx = builder.x[i + 1] - builder.x[i]
        dy = builder.y[i + 1] - builder.y[i]
        if math.hypot(dx, dy) > 1e-6:
            return math.atan2(dy, dx)
    return float(fallback_yaw)


def _all_points_safe(
    xs: Sequence[float],
    ys: Sequence[float],
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    safety_cells: float,
    skip_first: bool = False,
    start_zone: Optional[_StartZone] = None,
) -> bool:
    """逐个采样点判安全；skip_first 用于放行"车已经停着"的起点（M5）。

    ``start_zone`` 非空时，本体占位圈内的点豁免安全距离/障碍检查（未知仍禁止）。
    """
    for idx, (x, y) in enumerate(zip(xs, ys)):
        if skip_first and idx == 0:
            continue
        if not _point_safe(
            x, y, explored, distance, world_to_grid_rc, safety_cells,
            start_zone=start_zone,
        ):
            return False
    return True


def _start_cell_known_free(
    x: float,
    y: float,
    explored: np.ndarray,
    world_to_grid_rc: Callable,
) -> bool:
    """起点所在格是否为地图内的**已知自由格**（-1 未知 / 1 障碍都不放行）。"""
    r, c = _to_grid_rc(world_to_grid_rc, x, y)
    h, w = explored.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    return int(explored[r, c]) == 0


def _start_cell_known(
    x: float,
    y: float,
    explored: np.ndarray,
    world_to_grid_rc: Callable,
) -> bool:
    """起点所在格是否为地图内的**已知格**（未知 -1 不放行；障碍 1 按本体占位放行）。

    修复 A 的配套判定：机器人物理上就压在那一格上，建图抖动可能把它判成障碍；
    只要不是未知格（未知区域表示"从未观测"，不可能有车），就允许起步。
    """
    r, c = _to_grid_rc(world_to_grid_rc, x, y)
    h, w = explored.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    return int(explored[r, c]) >= 0


def _try_arc(
    pose: Tuple[float, float, float],
    delta: float,
    radius: float,
    v_sign: float,
    config: AckermannTrajConfig,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    skip_first_safety: bool = False,
    start_zone: Optional[_StartZone] = None,
):
    """构造一段运动弧并检查全部采样点；安全返回 builder，否则 None。

    `skip_first_safety=True` 仅用于**第一段**机动：它的首点就是车辆当前停的位置，
    即使落在安全包络内也允许（M5），但该格必须是已知自由格。
    """
    builder = _PathBuilder()
    builder.push(pose[0], pose[1])
    # 机动段很短，用半格步长采样：既能更密地做碰撞检查，也不会让点数膨胀。
    builder.add_arc_motion(
        pose, delta, radius, v_sign,
        min(config.ds_m, 0.5 * config.res_m),
    )
    if skip_first_safety:
        # 修复 A：本体占位圈内允许起点格被建图判成障碍（只禁止未知格）；
        # 圈外仍要求起点是已知自由格。
        if start_zone is not None and start_zone.contains(pose[0], pose[1]):
            if not _start_cell_known(pose[0], pose[1], explored, world_to_grid_rc):
                return None
        elif not _start_cell_known_free(
            pose[0], pose[1], explored, world_to_grid_rc,
        ):
            return None
    if not _all_points_safe(
        builder.x, builder.y, explored, distance,
        world_to_grid_rc, config.safety_cells,
        skip_first=skip_first_safety,
        start_zone=start_zone,
    ):
        return None
    return builder


def _alignment_turn_candidates(need: float) -> List[float]:
    """返回需要尝试的航向变化量（M6）。

    同一最终航向有两种绕法：``need``（较短方向）与 ``need ∓ 2π``（另一方向）。
    夹角接近 ±π 时两者长度几乎相同、但转弯方向相反；此时两个都试，取第一个可行。
    其它情况只试较短的 ``need``。
    """
    candidates = [float(need)]
    if abs(abs(need) - math.pi) <= 0.35:
        alt = float(need) - math.copysign(2.0 * math.pi, need) if need != 0.0 else 0.0
        if abs(alt - need) > 1e-9:
            candidates.append(alt)
    # 先短后长；长度相同时 need 优先（稳定排序）。
    candidates.sort(key=lambda d: abs(d))
    return candidates


def _build_alignment(
    start_pose: Tuple[float, float, float],
    dir0: float,
    config: AckermannTrajConfig,
    r_min: float,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    start_zone: Optional[_StartZone] = None,
):
    """生成起步对齐机动。

    返回 ``(builder, end_pose, n_segments, reason)``；builder 的 n==1 表示不需要机动。
    失败时 builder 为 None。

    **H3：取消"60° 以内容忍"**。只要
    ``|wrap(dir0 - start_yaw)| > align_tol_rad``（默认 1°，仅容忍数值噪声）就必须
    生成对齐弧，否则几何路径沿 ``dir0`` 走、yaw 仍从车头开始，违反
    ``x' = v*cos(yaw)``、``y' = v*sin(yaw)``。

    策略（每个候选转向都按下面顺序尝试，取第一个可行）：
      1. 纯前进弧：R=R_min，扫角 = 候选航向变化，一次到位；
      2. 纯前进弧碰撞时，若允许倒车，做三段交替(前进/倒车/前进)的三点掉头：
         每段 |kappa| = 1/R_min；倒车段 kappa 与 v 反号，保证每段都把 yaw 朝
         目标方向累积；每段弧长受 max_reverse_len_m 限制（前进段可放宽到 2 倍）。
      3. 都不行 → 返回 None，原因唯一："起步姿态无法在对齐弧内对齐"。
    """
    x0, y0, yaw0 = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
    need = _wrap_angle(dir0 - yaw0)

    base = _PathBuilder()
    base.push(x0, y0)
    if abs(need) <= float(config.align_tol_rad):
        # 仅数值噪声级别：保留原几何路径，yaw 与首段切线的残差 <= align_tol_rad。
        return base, (x0, y0, yaw0), 0, ""

    # --- 候选转向：先短弧、再另一绕法（接近 ±π 时两者都试） ---
    for delta in _alignment_turn_candidates(need):
        # --- 策略 A：纯前进弧 ---
        forward = _try_arc(
            (x0, y0, yaw0), delta, r_min, +1.0,
            config, explored, distance, world_to_grid_rc,
            skip_first_safety=True,
            start_zone=start_zone,
        )
        if forward is not None:
            end_pose = (forward.x[-1], forward.y[-1], yaw0 + delta)
            return forward, end_pose, 1, ""

        if not config.allow_reverse_align:
            continue

        # --- 策略 B：三点掉头 ---
        # 三段转角按不同比例分配，任何一段都能单独被 max_reverse_len_m 截断。
        patterns = [
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.25, 0.50, 0.25),
            (0.40, 0.40, 0.20),
            (0.20, 0.30, 0.50),
        ]

        for pattern in patterns:
            if abs(sum(pattern) - 1.0) > 1e-9:
                continue
            builder = _PathBuilder()
            builder.push(x0, y0)
            pose = (x0, y0, yaw0)
            ok = True
            for seg_idx, (v_sign, frac) in enumerate(zip((+1.0, -1.0, +1.0), pattern)):
                seg_delta = delta * frac
                if abs(seg_delta) < _EPS:
                    continue
                # 逐段弧长上限：倒车段 max_reverse_len，前进段 2*max_reverse_len。
                max_len = (2.0 if v_sign > 0.0 else 1.0) * float(config.max_reverse_len_m)
                if r_min * abs(seg_delta) > max_len + 1e-9:
                    ok = False
                    break
                seg = _try_arc(
                    pose, seg_delta, r_min, v_sign,
                    config, explored, distance, world_to_grid_rc,
                    # 只有整条机动的第一段首点可以落在安全包络内（车就在那儿）。
                    skip_first_safety=(seg_idx == 0),
                    start_zone=start_zone,
                )
                if seg is None:
                    ok = False
                    break
                builder.extend(seg)
                pose = (seg.x[-1], seg.y[-1], pose[2] + seg_delta)
            if ok:
                return builder, pose, 3, ""

    return None, (x0, y0, yaw0), 0, "起步姿态无法在对齐弧内对齐"


# ---------------------------------------------------------------------------
# 步骤 6：速度剖面
# ---------------------------------------------------------------------------


def _speed_profile(
    kappa_pt: np.ndarray,
    vsign_pt: np.ndarray,
    ds_int: np.ndarray,
    config: AckermannTrajConfig,
) -> np.ndarray:
    """曲率限速 + 纵向加速度前后扫描，返回带符号速度。

    横向限速：``v_curv = sqrt(a_lat_max / |kappa|)``，再取 ``min(v_max, .)`` 与
    ``max(., v_min)``。若 ``v_min`` 本身就让 ``v*|kappa| > a_lat_max``，仍保留
    ``v_min``（按需求不做失败判定）。

    纵向：以相邻点为单位做后向（提前减速）与前向（起步加速）扫描。使用
    ``v_allowed = sqrt(v_next^2 + 2*a_long*ds)`` —— 这恰是恒定纵向加速度下的
    物理可达速度，比 "|dv| <= a*dt" 的线性近似更严格，因此不会超限。
    不同运动方向（前进/倒车）之间不互相传播速度：倒车段与前进段在拼接点处
    由控制器停车换向，速度剖面按连续同号区间分段处理。

    M1：``v_sign`` 变化处（含起点、终点）把 ``|v|`` 置 0。否则拼接点会出现
    ``+0.3 -> -0.3`` 的瞬时跳变。先置零再前后扫描，保证 0 值能向两侧传播成
    "减速到 0 -> 换向 -> 重新加速"的过渡。
    """
    n = len(kappa_pt)
    if n == 0:
        return np.zeros(0, dtype=float)

    kappa_abs = np.maximum(np.abs(kappa_pt), 1e-6)
    v_mag = np.minimum(float(config.v_max), np.sqrt(float(config.a_lat_max) / kappa_abs))
    v_mag = np.maximum(v_mag, float(config.v_min))

    # M1：起点、终点必须静止。
    v_mag[0] = 0.0
    if n >= 2:
        v_mag[-1] = 0.0
        # 区间 k-1 与 k 的运动方向不同 -> 点 k 是换向点，必须停车。
        for k in range(1, n - 1):
            if vsign_pt[k - 1] != vsign_pt[k]:
                v_mag[k] = 0.0

    if n >= 2:
        a_long = max(float(config.a_long_max), _EPS)
        i = 0
        # 按"区间运动方向"分连续段；一段 [i, j+1] 内的区间 i..j 同号。
        while i < n - 1:
            j = i
            while j + 1 < n - 1 and vsign_pt[j + 1] == vsign_pt[i]:
                j += 1
            # 后向扫描：前方有慢点，提前减速。
            for _ in range(2):
                for k in range(j, i - 1, -1):
                    allowed = math.sqrt(v_mag[k + 1] ** 2 + 2.0 * a_long * max(ds_int[k], 0.0))
                    if allowed < v_mag[k]:
                        v_mag[k] = allowed
                # 前向扫描：从低速点起步，受纵向加速度限制。
                for k in range(i + 1, j + 2):
                    allowed = math.sqrt(v_mag[k - 1] ** 2 + 2.0 * a_long * max(ds_int[k - 1], 0.0))
                    if allowed < v_mag[k]:
                        v_mag[k] = allowed
            i = j + 1

    return v_mag * vsign_pt


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def _empty_trajectory(reason: str, warnings: Optional[List[str]] = None) -> Trajectory:
    """失败结果：**所有数组长度 0**（H5：绝不再返回部分轨迹）。"""
    empty = np.zeros(0, dtype=float)
    return Trajectory(
        x=empty.copy(),
        y=empty.copy(),
        yaw=empty.copy(),
        kappa=empty.copy(),
        v=empty.copy(),
        s=empty.copy(),
        feasible=False,
        reason=reason,
        warnings=list(warnings) if warnings else [],
    )


def _single_point_trajectory(
    start_pose_world: tuple,
    explored: np.ndarray,
    distance: np.ndarray,
    world_to_grid_rc: Callable,
    config: AckermannTrajConfig,
) -> Trajectory:
    """起点=终点（同一格）时的"原地不动"轨迹（H1）。

    N=1、各数组长度 1、速度为 0。起点若落在安全包络内（距离 < safety_cells），
    按 M5 放行并记录 warning。修复 A：起点的栅格允许是**已知障碍**（机器人本体
    占位，建图抖动常把它判成障碍），但未知格（-1）仍然不放行。
    """
    x0 = float(start_pose_world[0])
    y0 = float(start_pose_world[1])
    yaw0 = float(start_pose_world[2])

    warnings: List[str] = []
    if not _point_safe(x0, y0, explored, distance, world_to_grid_rc, config.safety_cells):
        if not _start_cell_known(x0, y0, explored, world_to_grid_rc):
            return _empty_trajectory("轨迹平移后进入不安全或未知区域", warnings)
        warnings.append("起点在安全包络内，已放行")

    return Trajectory(
        x=np.array([x0], dtype=float),
        y=np.array([y0], dtype=float),
        yaw=np.array([yaw0], dtype=float),
        kappa=np.array([0.0], dtype=float),
        v=np.array([0.0], dtype=float),
        s=np.array([0.0], dtype=float),
        feasible=True,
        reason="",
        warnings=warnings,
    )


def _plan_ackermann_trajectory_impl(
    explored_map: np.ndarray,
    grid_to_world_xy: Callable,
    world_to_grid_rc: Callable,
    start_pose_world: tuple,
    goal_world_xy: tuple,
    config: AckermannTrajConfig,
    planner=None,
    distance_field: Optional[np.ndarray] = None,
) -> Trajectory:
    """把 A* 栅格路径转成可执行的阿克曼轨迹。

    :param explored_map: (H,W) int8，-1 未知 / 0 自由 / 1 障碍。
    :param grid_to_world_xy: callable (row, col) -> (x_m, y_m)。
    :param world_to_grid_rc: callable (x_m, y_m) -> (row, col)。
    :param start_pose_world: (x, y, yaw)。
    :param goal_world_xy: (x, y)。
    :param config: AckermannTrajConfig。
    :param planner: 现有 AStarPlanner 实例；None 时内部新建。
    :param distance_field: 可选，预先算好的"到最近已知障碍距离场"。传入则整条流程
        （含起步机动后的重规划）复用这一份，不再计算；不传则本函数**只算一次**。

    坐标换算由调用方决定，模块不依赖 ROS/map origin，可离线单测。

    位置连续性说明：几何路径按 A* 栅格中心生成；实际轨迹必须从车辆真实位置 /
    机动终点起步。无机动时把几何路径平移 ``车辆位置 - 起点格中心``（不超过半格
    对角线）；有起步机动时先拼接机动段，再从机动终点重新规划几何路径并做同样的
    微小平移。平移不改变切线与曲率，拼接处 yaw 精确连续。倒角圆弧保留
    ``_ARC_BUILD_MARGIN_CELLS``（0.5 格）的构建余量吸收这半步平移。

    **失败策略（H5）**：任何失败都返回全空数组 + 单一原因；不再"提高安全距离整体
    重试一次"，也不再返回部分轨迹。
    """
    explored = np.asarray(explored_map)
    if explored.ndim != 2 or explored.size == 0:
        return _empty_trajectory("地图为空")

    cfg = config
    r_min = resolve_min_turn_radius(cfg)
    kappa_max = resolve_max_curvature(cfg)
    if not np.isfinite(r_min) or r_min <= 0.0 or kappa_max <= 0.0:
        return _empty_trajectory("最小转弯半径无效")

    # ---- H4：整次调用最多算一次距离场 ----
    # 只对**已知障碍**计算（未知格不参与安全距离，否则探索前沿永远不可达）；
    # 未知格的不可通行由 A* 的 treat_unknown_as_obstacle 与 _point_safe 硬判定。
    if distance_field is not None:
        distance = np.asarray(distance_field)
        if distance.shape != explored.shape:
            # 尺寸不匹配的复用会索引越界，这里退回自算一次（仍然只算一次）。
            distance = distance_to_obstacle(explored == 1)
    else:
        distance = distance_to_obstacle(explored == 1)

    start_r, start_c = _to_grid_rc(world_to_grid_rc, start_pose_world[0], start_pose_world[1])
    goal_r, goal_c = _to_grid_rc(world_to_grid_rc, goal_world_xy[0], goal_world_xy[1])
    h, w = explored.shape
    if not (0 <= start_r < h and 0 <= start_c < w):
        return _empty_trajectory("起点越界")
    if not (0 <= goal_r < h and 0 <= goal_c < w):
        return _empty_trajectory("终点越界")

    # ---- 修复 A：起点本体占位圈（以**机器人真实位置**为圆心）----
    # A* 用 start_clear_cells 清空同圈阻挡；轨迹的捷径/倒角/最终复核也必须对同圈
    # 采样点豁免安全距离检查，否则会出现"A* 能算、复核判死"的矛盾。圈内仍禁止未知格。
    start_zone = _start_zone_from_pose(start_pose_world, cfg)

    # ---- 步骤 1~3 + 步骤 5：规划 -> （必要时）机动 -> 从机动终点重新规划 ----
    # 为什么机动后要重新规划：
    #   阿克曼的三点掉头必然带来位置偏移（最多约 R_min*pi）。若把原几何路径整体
    #   刚性平移到机动终点，窄通道里整条路线会跟着平移，容易擦墙。正确做法是从
    #   机动终点的新位姿重新做一次 A* -> 捷径 -> 倒角，只保留"机动段 + 新路线"。
    # 最多 3 轮：初始路线 + 两次"机动后重规划"。仍对不齐就判失败，不无限迭代。
    max_rounds = 3
    pose = (
        float(start_pose_world[0]),
        float(start_pose_world[1]),
        float(start_pose_world[2]),
    )
    maneuvers = None
    route = None
    dir0 = None

    for attempt in range(max_rounds):
        start_rc = _to_grid_rc(world_to_grid_rc, pose[0], pose[1])
        if not (0 <= start_rc[0] < h and 0 <= start_rc[1] < w):
            return _empty_trajectory("起点越界")

        route, dir0, reason = _build_route(
            explored, start_rc, (goal_r, goal_c), cfg, planner,
            distance, world_to_grid_rc, grid_to_world_xy,
            distance_field=distance,
            start_zone=start_zone,
        )
        if route is None:
            return _empty_trajectory(reason)

        if dir0 is None:
            # 本轮起点已经落在终点格，无需再对齐。
            if maneuvers is None:
                maneuvers = _PathBuilder()
                maneuvers.push(pose[0], pose[1])
            break

        alignment, end_pose, n_segments, align_reason = _build_alignment(
            pose, dir0, cfg, r_min, explored, distance, world_to_grid_rc,
            start_zone=start_zone,
        )
        if alignment is None:
            return _empty_trajectory(align_reason or "起步姿态无法在对齐弧内对齐")

        if n_segments == 0:
            # |wrap(dir0 - yaw)| <= align_tol_rad：数值噪声级，直接沿用几何路径。
            if maneuvers is None:
                maneuvers = _PathBuilder()
                maneuvers.push(pose[0], pose[1])
            break

        # 需要机动：先接上机动段，再以机动终点为新位姿重规划一次。
        if maneuvers is None:
            maneuvers = alignment
        else:
            maneuvers.extend(alignment)
        pose = (float(end_pose[0]), float(end_pose[1]), float(end_pose[2]))
        if attempt == max_rounds - 1:
            # 重规划后的新路线首段仍与当前朝向差太多：不再无限迭代，判失败。
            return _empty_trajectory("起步姿态无法在对齐弧内对齐")

    if route is None:
        return _empty_trajectory("A* 无路")

    if route.n < 2:
        # 已到终点格：有机动就只返回机动段，否则返回车辆当前位置的单点轨迹。
        if maneuvers is not None and maneuvers.n > 1:
            full = maneuvers
        else:
            return _single_point_trajectory(
                start_pose_world, explored, distance, world_to_grid_rc, cfg,
            )
    else:
        # 只平移到当前位姿（无机动时是车辆真实位置，有动机动时是机动终点）。
        # 平移不改变切线/曲率，因此拼接处 yaw 精确连续。
        base_x, base_y = float(pose[0]), float(pose[1])
        shift_x = base_x - route.x[0]
        shift_y = base_y - route.y[0]
        for k in range(route.n):
            route.x[k] += shift_x
            route.y[k] += shift_y
        route.x[0] = base_x
        route.y[0] = base_y

        # ---- 步骤 4：拼接并按区间记录等弧长采样 ----
        if maneuvers is not None:
            full = _PathBuilder()
            full.extend(maneuvers)
            full.extend(route)
        else:
            full = route

    if full.n < 2:
        return _empty_trajectory("轨迹点不足")

    x_arr, y_arr, kappa_pt, vsign_pt, ds_int = full.point_arrays()

    # ---- 步骤 7（H5）：平移后对**整条轨迹**重新做安全复核 ----
    # 口径两条同时成立：到已知障碍距离 >= safety_cells，且该格不是未知(-1)。
    #
    # 修复 A：落在**起点本体占位圈**内的采样点豁免安全距离/障碍检查。机器人物理上
    # 占着这一圈（A* 也已用 start_clear_cells 清空同圈阻挡），贴墙/建图抖动时圈内
    # 的格子必然不满足 safety_cells；若不放行，A* 算出来的路会被最终复核判死。
    # **未知格绝不豁免**：未知区域不允许穿越，这条硬约束始终成立。
    # 豁免点只记录一条汇总 warning，避免日志刷屏。
    warnings: List[str] = []
    exempt_count = 0
    for k in range(full.n):
        xk = float(x_arr[k])
        yk = float(y_arr[k])
        # 先按严格口径（含障碍/安全距离）检查。
        if _point_safe(xk, yk, explored, distance, world_to_grid_rc, cfg.safety_cells):
            continue
        # 严格检查失败：判断是否只是"本体占位圈内"的误判。
        rk, ck = _to_grid_rc(world_to_grid_rc, xk, yk)
        in_bounds = 0 <= rk < h and 0 <= ck < w
        cell_is_unknown = in_bounds and int(explored[rk, ck]) == -1
        if start_zone.contains(xk, yk) and not cell_is_unknown:
            exempt_count += 1
            if k == 0:
                # 保留原有起点提示，方便现场判断"车贴墙起步"。
                warnings.append("起点在安全包络内，已放行")
            continue
        if k == 0 and _start_cell_known_free(xk, yk, explored, world_to_grid_rc):
            warnings.append("起点在安全包络内，已放行")
            continue
        return _empty_trajectory("轨迹平移后进入不安全或未知区域", warnings)

    if exempt_count > 0:
        # 只记一条汇总（需求要求），不逐点刷日志。
        warnings.append(
            f"起点占位圈内 {exempt_count} 个采样点豁免安全距离检查"
        )

    # yaw 积分：yaw_{i+1} = yaw_i + v_sign * ds * kappa。
    # 起点用车辆真实 yaw；有起步机动时，机动末段自然精确落到 dir0。
    yaw = np.empty(full.n, dtype=float)
    yaw[0] = float(start_pose_world[2])
    for k in range(full.n - 1):
        yaw[k + 1] = yaw[k] + float(vsign_pt[k]) * float(ds_int[k]) * float(kappa_pt[k])

    s = np.concatenate(([0.0], np.cumsum(ds_int))) if len(ds_int) else np.zeros(1)

    # ---- 步骤 6：速度剖面（M1：换向点/首尾速度为 0） ----
    v = _speed_profile(kappa_pt, vsign_pt, ds_int, cfg)

    # ---- 曲率复核（构造上应恒可行，这里只做数值兜底） ----
    if np.max(np.abs(kappa_pt)) > kappa_max + 1e-6:
        return _empty_trajectory("曲率超限", warnings)

    return Trajectory(
        x=x_arr, y=y_arr, yaw=yaw, kappa=kappa_pt, v=v, s=s,
        feasible=True, reason="", warnings=warnings,
    )


def plan_ackermann_trajectory(
    explored_map: np.ndarray,
    grid_to_world_xy: Callable,
    world_to_grid_rc: Callable,
    start_pose_world: tuple,
    goal_world_xy: tuple,
    config: AckermannTrajConfig,
    planner=None,
    distance_field: Optional[np.ndarray] = None,
) -> Trajectory:
    """对外入口：见 `_plan_ackermann_trajectory_impl` 的完整说明。

    :param distance_field: 可选，预先算好的"到最近已知障碍距离场"（形状与地图一致）。
        传入时本函数**不再调用** `distance_to_obstacle`；不传时整次调用也只算一次。
        内部的 A*（含起步机动后的重规划）全部复用这份距离场（H4）。
    """
    return _plan_ackermann_trajectory_impl(
        explored_map,
        grid_to_world_xy,
        world_to_grid_rc,
        start_pose_world,
        goal_world_xy,
        config,
        planner=planner,
        distance_field=distance_field,
    )
