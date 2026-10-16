from __future__ import annotations

from pathlib import Path

if __package__:
    from .light_qifa import LightHeuristicConfig
else:
    from light_qifa import LightHeuristicConfig


# real_work 下验证脚本的默认参数。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAP_SIZE = 200
SENSOR_RANGE = 12
MAP_GENERATOR = "easy_big_room"
SEED = 150
MAX_ROUNDS = 200
MAX_MACRO_STEPS = 40
SHOW_VISUALIZATION = True

# 纯几何启发式权重，距离单位为栅格。
# 调参结论（easy_big_room, 200x200, sensor_range=12, 10 个种子全部探索完成）：
#   平均物理步数 1386.6，优于旧配置(1.0/0.35/2.0) 的 1527.3，也优于 BFS 基线 1581.1。
HEURISTIC_CONFIG = LightHeuristicConfig(
    bfs_weight=8.0,
    turn_weight=0.35,
    obstacle_weight=1.0,
    obstacle_safe_distance=6.0,
)


# 方便后续接入真实地图时统一管理路径。
REAL_MAP_ROOT = PROJECT_ROOT / "standard_test" / "maps"


def validate_config() -> None:
    valid_generators = {"easy_big_room", "fish_bone"}
    if MAP_GENERATOR not in valid_generators:
        raise ValueError(
            f"MAP_GENERATOR 必须是 {sorted(valid_generators)} 之一，当前为 {MAP_GENERATOR!r}"
        )
    if MAP_SIZE <= 0:
        raise ValueError("MAP_SIZE 必须大于 0")
    if SENSOR_RANGE <= 0:
        raise ValueError("SENSOR_RANGE 必须大于 0")
    if MAX_ROUNDS <= 0 or MAX_MACRO_STEPS <= 0:
        raise ValueError("MAX_ROUNDS 和 MAX_MACRO_STEPS 必须大于 0")
