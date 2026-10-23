from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

# 与决策侧（real_work/light_qifa.py）共用同一份"到最近障碍距离"实现。
# 必须共用，才能保证候选过滤条件（dist >= R 放行）与膨胀阻挡条件（dist < R 阻挡）
# 在 dist == R 时判定一致，不会出现"决策放行、A* 阻挡"的重叠。
from core.obstacle_distance import distance_to_obstacle


class AStarPlanner:
    """A* 栅格规划器（默认 4 邻域；可选 8 邻接），支持对障碍做硬膨胀（车体安全包络）。

    关键参数 `inflation_radius_cells`：障碍硬膨胀半径（单位：栅格）。规划前会一次性
    把所有到最近障碍的**栅格距离严格小于**该值的格子视为阻挡（`dist < R` 阻挡，
    `dist == R` 放行），从而禁止 "贴墙一格走"，并让目标点不再被允许落在墙边。

    度量口径：本类的膨胀与决策侧 `real_work/light_qifa.py` 的候选过滤共用
    `core/obstacle_distance.distance_to_obstacle()`（有 scipy 时精确欧氏，否则
    chamfer 3-4 近似欧氏）。候选条件 `dist >= obstacle_safe_distance` 与本类
    `dist < inflation_radius_cells` 在 `dist == R` 时一致放行，不会再出现
    "决策侧放行、A* 侧把目标格当阻挡"的重叠。

    取值来源（真机基准）：车体安全包络半径约 0.25 m，栅格分辨率 0.05 m/格，
    故 0.25 / 0.05 = 5 格（与 `regional_legacy/config.py` 的 `SAFE_ENVELOPE_CELLS` 一致）。

    默认值：`plan()` 的 `inflation_radius_cells` 默认为 0（不膨胀），以保持既有离线/
    训练/评测调用（它们都不传该参数）行为不变；部署侧（ROS 探索器）需显式传 5。

    失败策略：起点或终点落在膨胀阻挡里导致规划失败时**直接返回 None**，不再在内部
    静默退回未膨胀障碍重试。是否退回由调用方（如 frontier_explorer_node 用
    `inflation_radius_cells=0` 显式重试）统一决定，便于调用方区分"膨胀暂时封路"与
    "真不可达"，也避免内部 warning 刷屏。

    新增开关（默认全部关闭，保证既有离线训练/评测行为逐位不变）：
      * `use_eight_neighbors`：8 邻接搜索。对角步代价 sqrt(2)，且**禁止穿角**（对角
        扩展时要求两个正交邻格都可通行），避免贴着障碍角斜穿过去。
      * `treat_unknown_as_obstacle`：把 `-1`（未知）当硬障碍。用于"规划时未知即不可
        穿越"的场合（阿克曼轨迹生成）。上游候选前沿的 BFS 可达性只在已知自由格上
        计算，因此这个开关不会制造"合法前沿不可达"。

        **重要（问题 1 修复）**：未知格只作为"硬阻挡"并入，**绝不参与膨胀**。
        原因：frontier 的定义是"自由格且邻域含未知"，合法目标格天然紧贴未知格；
        若把未知格并入障碍后一起膨胀 R 格，目标格必然落在"未知膨胀圈"内，自己
        变成阻挡 -> A* 无路（实测复杂建筑里 45 次 `A* 无路` + 12 次
        `A* 路径不满足安全距离` 的直接根因）。膨胀（安全包络）是为已知墙体的
        几何碰撞留余量，未知区域本来就不允许进入，对它膨胀没有意义。
    """

    def __init__(
        self,
        fog_penalty=0.8,
        use_eight_neighbors: bool = False,
        treat_unknown_as_obstacle: bool = False,
    ):
        # 4方向移动：上下左右；8 邻接时额外加入四条对角。
        self.fog_penalty = fog_penalty
        self.use_eight_neighbors = bool(use_eight_neighbors)
        self.treat_unknown_as_obstacle = bool(treat_unknown_as_obstacle)
        self.motions = self._build_motions(self.use_eight_neighbors)

    @staticmethod
    def _build_motions(eight: bool):
        """构造运动基元表。默认 4 邻接，顺序与改动前完全一致。"""
        if eight:
            return [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def plan(
        self,
        explored_map,
        start,
        goal,
        inflation_radius_cells: int = 0,
        use_eight_neighbors: bool = None,
        treat_unknown_as_obstacle: bool = None,
        distance_field: Optional[np.ndarray] = None,
        start_clear_cells: int = 0,
        clearance_target_cells: float = 0.0,
        clearance_weight: float = 0.0,
    ):
        """在当前探索地图上规划从 start 到 goal 的 A* 路径。

        :param inflation_radius_cells: 障碍硬膨胀半径（栅格）。**默认 0 = 不膨胀**，保持
            既有离线/训练/评测调用（env/grid_env.py、tools/eval_policy.py、
            real_work/compare_light.py、real_work/test_light_qifa.py、
            core/separating_runtime.py 等均未显式传该参数）的行为与改动前一致，避免静默
            改变训练/离线指标。**部署侧（真机/仿真探索器）需显式传值**：真机安全包络约
            0.25 m、栅格分辨率 0.05 m/格，故应传 5（与 regional_legacy/config.py 的
            SAFE_ENVELOPE_CELLS 一致）。
        :param use_eight_neighbors: 单次调用覆盖实例默认的邻接方式；None 表示用实例设置。
        :param treat_unknown_as_obstacle: 单次调用覆盖实例默认的未知格策略；None 表示用
            实例设置。True 时 `-1` 视为**硬阻挡但不参与膨胀**（见下方掩膜构造说明），
            且终点为未知格直接判失败。
        :param distance_field: 可选，预先算好的"到最近**已知障碍**的栅格距离场"（形状须与
            explored_map 一致）。传入时直接用它做膨胀，**不再内部调用
            `distance_to_obstacle`**；不传时保持原行为（radius>0 才计算一次）。阿克曼轨迹
            生成会把这个场同时传给多次 A* 调用（起步机动后重规划），确保整条流水线
            只算一次距离变换。
        :param start_clear_cells: 起点本体占位圈半径（栅格，Chebyshev 距离）。大于 0 时把
            起点格周围该圈内的阻挡掩膜清零（**只清阻挡掩膜，不改地图/终点判定**）。
            默认 0 = 完全跳过，行为与改造前逐位一致。详见下方掩膜构造处的说明。
        :param clearance_target_cells: **软性**期望净空（栅格）。距离小于该值的格子会
            附加二次代价 `weight * (target - d)^2`，让路径主动往开阔侧偏，而不是贴着
            硬膨胀边界走。默认 0 = 关闭（既有离线/训练调用行为逐位不变）。
            只对**已知障碍**生效（未知格仍只做硬阻挡），因此不会把前沿目标格"推走"。
        :param clearance_weight: 软性净空代价权重（每格）。默认 0 = 关闭。
            经验取值：target=10 格、weight=0.05 时，贴墙 30 格的额外代价约 37，
            足以让规划选择多绕 10~15% 的路径（与本项目的绕行预算一致）。
            该代价恒非负，不破坏启发式可采纳性。
        :return: 栅格路径 [[x, y], ...]（起点在前、终点在后）；规划失败返回 None。
        """
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])

        eight = self.use_eight_neighbors if use_eight_neighbors is None else bool(use_eight_neighbors)
        unknown_block = (
            self.treat_unknown_as_obstacle
            if treat_unknown_as_obstacle is None
            else bool(treat_unknown_as_obstacle)
        )

        if not (0 <= gx < explored_map.shape[0] and 0 <= gy < explored_map.shape[1]):
            return None
        if explored_map[gx, gy] == 1:
            return None
        if unknown_block and explored_map[gx, gy] == -1:
            return None
        if sx == gx and sy == gy:
            return [[sx, sy]]

        # ------------------------------------------------------------------
        # 阻挡掩膜构造（问题 1 修复）：已知障碍"膨胀" + 未知格"硬阻挡"，两者分开。
        #
        # 为什么未知格**绝不能**参与膨胀：
        #   frontier 的定义就是"自由格且邻域含未知"，合法目标格天然紧贴未知格。
        #   若把未知格并入障碍掩膜后一起膨胀 R 格（旧实现正是如此），目标格必然落进
        #   "未知格膨胀圈"内 -> 目标格自己变成阻挡 -> A* 无路。实测复杂建筑里
        #   45 次 `A* 无路` 与 12 次 `A* 路径不满足安全距离` 就是这条链式误判。
        #   未知格只需要"不可进入"这一条硬约束；安全包络是为已知墙体的几何碰撞留的
        #   余量，对"本来就不许进"的未知区域做膨胀没有任何意义。
        #
        # 已知障碍仍照旧用统一的 `distance_to_obstacle` 做硬膨胀：
        #   `dist < radius` 阻挡，`dist == radius` 放行，与决策侧候选过滤口径一致。
        #   `distance_to_obstacle` 的语义不变，只对已知障碍算距离。
        # treat_unknown_as_obstacle=False 时完全不并入未知格，行为与改动前逐位一致。
        # ------------------------------------------------------------------
        known_obstacles = explored_map == 1
        # radius=0 时直接返回已知障碍掩膜（行为不变）。
        # 传入 distance_field 时不重复做距离变换（H4：一次轨迹生成只算一次距离场）。
        inflated_block = self._inflation_block_mask(
            known_obstacles, int(inflation_radius_cells), distance_field=distance_field
        )
        if unknown_block:
            # 未知格（-1）只做硬阻挡并入，不参与膨胀（原因见上面的注释）。
            block_mask = inflated_block | (explored_map < 0)
        else:
            # 默认路径：与改动前完全一致，只有已知障碍参与的膨胀掩膜。
            block_mask = inflated_block

        # ------------------------------------------------------------------
        # 起点本体占位圈清空（修复 A：A* 无路的首要根因）
        #
        # 机器人物理上就占了这一圈：建图抖动/贴墙时，它自己的格常被判为障碍，或落在
        # 障碍膨胀区/未知格里，于是 A* 一开算就发现"起点四周全被挡住" -> 立即返回
        # None（实测复杂建筑 59 次机器人处在非自由格，201/533 的回退原因是 A* 无路）。
        # 清空本体占位是 costmap 的标准做法（robot footprint clearing：车体占位格
        # 不参与障碍膨胀），这里只对**起点周围一圈**做同样的处理。
        #
        # 边界：只清 block_mask，不修改 explored_map；终点检查、圈外的膨胀与未知硬阻挡
        # 全部保持原样。`start_clear_cells=0` 时整段跳过，保证既有离线回归逐位不变。
        # ------------------------------------------------------------------
        clear_radius = int(start_clear_cells)
        if clear_radius > 0:
            height, width = explored_map.shape
            r0 = max(0, sx - clear_radius)
            r1 = min(height, sx + clear_radius + 1)
            c0 = max(0, sy - clear_radius)
            c1 = min(width, sy + clear_radius + 1)
            if r0 < r1 and c0 < c1:
                rows = np.arange(r0, r1)[:, None]
                cols = np.arange(c0, c1)[None, :]
                # Chebyshev 距离 <= clear_radius 的正方形邻域（与 8 邻接的"格距"一致）。
                chebyshev = np.maximum(np.abs(rows - sx), np.abs(cols - sy))
                block_mask[r0:r1, c0:c1][chebyshev <= clear_radius] = False

        # ------------------------------------------------------------------
        # 软性净空代价（2026-10-22 决策）：硬膨胀只能保证"不贴墙"，但掩膜外所有格子
        # 代价相同，最短路径必然压在膨胀边界上（距障碍恰好 R 格）。这里叠加
        # `weight * (target - d)^2`，让路径主动往开阔侧偏，给阿克曼倒角
        # （R=0.35m 时内切约 2.9 格）和循迹误差留出余量。
        # 未知格不参与本代价：前沿目标天然紧贴未知，若把未知计入会把目标推开。
        # ------------------------------------------------------------------
        soft_target = max(float(clearance_target_cells), 0.0)
        soft_weight = max(float(clearance_weight), 0.0)
        soft_distance = None
        if soft_weight > 0.0 and soft_target > 0.0:
            if distance_field is not None and distance_field.shape == explored_map.shape:
                # 调用方已算好距离场（需保证其截断上限 >= soft_target，见调用侧说明）。
                soft_distance = distance_field
            else:
                soft_distance = distance_to_obstacle(
                    known_obstacles, max_distance=soft_target + 2.0
                )

        # 按上述硬阻挡规划；失败直接返回 None，由调用方决定是否用未膨胀参数重试。
        return self._search(
            explored_map,
            block_mask,
            sx,
            sy,
            gx,
            gy,
            motions=self._build_motions(eight),
            eight=eight,
            soft_distance=soft_distance,
            soft_target=soft_target,
            soft_weight=soft_weight,
        )

    def _search(
        self,
        explored_map,
        block_mask,
        sx,
        sy,
        gx,
        gy,
        motions=None,
        eight=False,
        soft_distance=None,
        soft_target=0.0,
        soft_weight=0.0,
    ):
        """在给定阻挡掩膜（block_mask）下做 A*，返回路径或 None。

        启发式、单步代价（含迷雾惩罚）与路径回溯逻辑与原始实现保持一致，唯一的区别是
        邻接判定由"该格是障碍"改为"该格在膨胀后仍不是阻挡"。

        ``soft_distance`` 非空时，对离障碍距离小于 ``soft_target`` 的格子附加
        ``soft_weight * (soft_target - d)^2`` 的软代价；启发式不包含该项，故仍可采纳。

        `eight=True` 时：对角步代价 sqrt(2)、启发式用可采纳的 octile 距离，并且**禁止
        穿角**——对角扩展要求当前格的两个正交邻格都不阻挡，否则会从两个障碍的缝隙
        里"斜穿"过去，生成的折线物理上不可通行。
        """
        H, W = explored_map.shape
        if not (0 <= sx < H and 0 <= sy < W):
            return None
        if motions is None:
            motions = self._build_motions(eight)
        # 注意：起点自身不做阻挡判定（与原始实现一致，原始实现也不校验起点是否为障碍）；
        # 阻挡判定只作用于"从当前格扩展出去的邻格"。
        open_set = []
        heapq.heappush(open_set, (self._heuristic(sx, sy, gx, gy, eight), 0, sx, sy))

        came_from = {}
        g_costs = {(sx, sy): 0}
        closed_set = set()

        while open_set:
            _, g, curr_x, curr_y = heapq.heappop(open_set)

            if curr_x == gx and curr_y == gy:
                return self._reconstruct_path(came_from, curr_x, curr_y)

            if (curr_x, curr_y) in closed_set:
                continue
            closed_set.add((curr_x, curr_y))

            for dx, dy in motions:
                nx, ny = curr_x + dx, curr_y + dy

                if not (0 <= nx < H and 0 <= ny < W):
                    continue
                # 邻接判定：该格在"膨胀后"仍不能是阻挡（原实现这里只判"该格是障碍"）。
                if block_mask[nx, ny]:
                    continue
                # 禁止穿角：对角移动必须两侧正交邻格都通行。
                if dx != 0 and dy != 0:
                    if block_mask[curr_x + dx, curr_y] or block_mask[curr_x, curr_y + dy]:
                        continue

                step_cost = math.sqrt(2.0) if (dx != 0 and dy != 0) else 1.0
                if explored_map[nx, ny] == -1:
                    step_cost += self.fog_penalty
                # 软性净空代价：越靠近已知障碍越贵，使路径主动居中/走开阔侧。
                if soft_distance is not None and soft_weight > 0.0:
                    clearance = float(soft_distance[nx, ny])
                    if clearance < soft_target:
                        gap = soft_target - clearance
                        step_cost += soft_weight * gap * gap

                new_g = g + step_cost
                if (nx, ny) not in g_costs or new_g < g_costs[(nx, ny)]:
                    g_costs[(nx, ny)] = new_g
                    f = new_g + self._heuristic(nx, ny, gx, gy, eight)
                    heapq.heappush(open_set, (f, new_g, nx, ny))
                    came_from[(nx, ny)] = (curr_x, curr_y)

        return None

    @staticmethod
    def _inflation_block_mask(
        obstacles: np.ndarray,
        radius: int,
        distance_field: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """返回膨胀后的阻挡掩膜：到最近障碍的栅格距离严格小于 radius 的格子视为阻挡。

        入参 `obstacles` **只应包含已知障碍**（`explored_map == 1`）。未知格不参与
        本函数（它由 `plan()` 单独做不膨胀的硬阻挡合并），否则 frontier 目标格会被
        未知格的膨胀圈封死（见 `plan()` 中的掩膜构造说明）。

        判定规则：`distance < radius` 为阻挡，`distance == radius` 放行。
        距离由 `core/obstacle_distance.distance_to_obstacle()` 统一计算，与决策侧
        （light_qifa 的候选条件 `distance >= obstacle_safe_distance`）完全同一度量，
        因此不会出现"候选放行、目标格却被膨胀盖住"的重叠。

        膨胀只标记"靠近真实障碍"的格子，天然做边界裁剪，不会把地图边缘当障碍。
        radius <= 0 时直接返回原始障碍掩膜（行为与改动前一致）。

        `distance_field` 非空且形状匹配时直接复用（不再调用距离变换）；这保证
        `plan_ackermann_trajectory` 里多次（含重试）A* 只用同一份距离场。
        """
        radius = int(radius)
        if radius <= 0:
            return obstacles.copy()

        # 障碍格距离=0，其余格为到最近障碍的距离（单位：栅格）。
        # 严格小于才阻挡：dist == radius 与决策侧的 `dist >= R` 放行保持一致。
        if distance_field is not None and distance_field.shape == obstacles.shape:
            return distance_field < float(radius)
        distance = distance_to_obstacle(obstacles)
        return distance < float(radius)

    def _heuristic(self, x1, y1, x2, y2, eight=False):
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        if eight:
            # octile 距离：8 邻接下的可采纳启发式（对角代价 sqrt(2)）。
            return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)
        return dx + dy

    def _reconstruct_path(self, came_from, curr_x, curr_y):
        path = [[curr_x, curr_y]]
        while (curr_x, curr_y) in came_from:
            curr_x, curr_y = came_from[(curr_x, curr_y)]
            path.append([curr_x, curr_y])
        path.reverse()
        return path
