# 这个文件是常数配置文件，修改这里就可以修改当前启发式版本的参数

from pathlib import Path

# 实验命名
EXPERIMENT_PREFIX = "heuristic_hybrid"

# ---------------------------------------------------------------------------
# 项目基础路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
CHECKPOINTS_ROOT = ARTIFACTS_ROOT / "checkpoints"
FUSION_WEIGHTS_ROOT = ARTIFACTS_ROOT / "fusion_weights"

# ---------------------------------------------------------------------------
# 环境与地图基础配置
# ---------------------------------------------------------------------------
# 修改 `SENSOR_RANGE` 可以调整雷达/视野照射范围。
# 修改 `TEST_MAP_SIZE` / `TRAIN_MAP_SIZE` 可以调整测试与训练地图尺寸。
# 修改 `MAP_GENERATOR` 可以切换地图生成器。

TRAIN_MAP_SIZE = 200
TEST_MAP_SIZE = 200
SENSOR_RANGE = 12
TEST_SEED = 114
MAP_GENERATOR = "easy_big_room"
TRAIN_MAX_MACRO_STEPS = 60

# ---------------------------------------------------------------------------
# 启发式权重配置
# ---------------------------------------------------------------------------
# 下面三个值分别对应：
# - `HEURISTIC_GAIN_WEIGHT`：信息增益权重
# - `HEURISTIC_COST_WEIGHT`：可达性距离代价权重
# - `HEURISTIC_TURN_WEIGHT`：掉头惩罚权重
# 这三个权重越大，对应因素对最终选择的影响越强。

HEURISTIC_GAIN_WEIGHT = 0
HEURISTIC_COST_WEIGHT = 20
HEURISTIC_TURN_WEIGHT = 2

# ---------------------------------------------------------------------------
# 环境奖励/代价
# ---------------------------------------------------------------------------
# 这些常量主要被 `grid_env.py` 和旧的环境逻辑引用；当前启发式主流程中不一定都会直接使用。

AREA_REWARD_SCALE = 0.035
SIGMA = 3.0
STEP_PENALTY = 0.9
SUCCESS_BONUS_PER_AREA = 0.0008
COLLISION_PENALTY = 40.0
REPEAT_PENALTY_SCALE = 1.0
RECENT_REPEAT_PENALTY_SCALE = 2.2
EARLY_EXIT_PENALTY_BASE = 10.0
EARLY_EXIT_PENALTY_UNFINISHED = 30.0
EARLY_EXIT_PENALTY_FINISHED = 3.0
PATH_KNOWN_CELL_PENALTY_SCALE = 0.35
ROOM_COMPLETION_POTENTIAL_SCALE = 1.8
ROOM_COMPLETION_POTENTIAL_K1 = 10.0
ROOM_COMPLETION_POTENTIAL_K2 = 70.0
ROOM_COMPLETION_POTENTIAL_K3 = 180.0
ROOM_COMPLETION_POTENTIAL_T1 = 0.90
ROOM_COMPLETION_POTENTIAL_T2 = 0.95
ROOM_COMPLETION_BONUS = 180.0
ROOM_COMPLETION_BONUS_EXTRA = 50.0

# ---------------------------------------------------------------------------
# 地图生成相关
# ---------------------------------------------------------------------------

CORRIDOR_WIDTH = 5
EASY_BIG_ROOM_CORRIDOR_WIDTH = 4
ROOM_MIN_SIZE = 22
ROOM_MAX_SIZE = 32
ROOM_PADDING = 3
ROOM_PLACE_ATTEMPTS = 200
ROOMS_BASE_COUNT = 4
ROOM_OBSERVABLE_AREA_RATIO = 0.16
ROOM_GENERATION_ATTEMPTS = 80
ROOM_AREA_ADJUST_ATTEMPTS = 300

OBS_MIN_SIZE = 2
OBS_MAX_SIZE = 4
OBSTACLES_PER_ROOM_MIN = 1
OBSTACLES_PER_ROOM_MAX = 5
OBSTACLE_PLACE_ATTEMPTS = 30
OBSTACLE_WALL_MARGIN = 2
SPAWN_MARGIN = 2

DYNAMIC_ROOM_MAX_SIZE_RATIO = 0.30
DYNAMIC_ROOM_MAX_SIZE_CAP = 80
DYNAMIC_ROOM_LARGE_THRESHOLD = 50
DYNAMIC_ROOM_LARGE_ROOM_PROB = 0.45
DYNAMIC_ROOM_OBSERVABLE_AREA_RATIO = 0.20
DYNAMIC_ROOM_GENERATION_ATTEMPTS = 100
DYNAMIC_ROOM_OBSTACLE_DENSITY = 0.0035
DYNAMIC_ROOM_PARTITION_MIN_LEN = 10
DYNAMIC_ROOM_PARTITION_MAX_LEN_RATIO = 0.55
DYNAMIC_ROOM_PARTITION_THICKNESS_MIN = 2
DYNAMIC_ROOM_PARTITION_THICKNESS_MAX = 4
DYNAMIC_ROOM_PARTITIONS_PER_LARGE_ROOM_MIN = 1
DYNAMIC_ROOM_PARTITIONS_PER_LARGE_ROOM_MAX = 3

MAP_TEMPLATE_SIZE = 100
WALL_WIDTH = 2
DOOR_WIDTH = 6
DOOR_PROTECT_MARGIN = 3
NUM_OBSTACLES_BASE = 45

ROW_MID1 = 45
ROW_MID2 = 70
COL_MID1 = 35
COL_MID2 = 65

DOOR_POSITIONS = [
    (ROW_MID1, 15),
    (ROW_MID1, 80),
    (ROW_MID2, 30),
    (ROW_MID2, 85),
    (20, COL_MID1),
    (80, COL_MID2),
]

SPAWN_X_MIN = 5
SPAWN_X_MAX = 30
SPAWN_Y_MIN = 5
SPAWN_Y_MAX = 30

# ---------------------------------------------------------------------------
# 旧训练/评估接口仍会间接调用到的基础函数
# 这里只保留不会影响启发式运行的必要项。
# ---------------------------------------------------------------------------

REPLAY_BUFFER_BASE_MAP_SIZE = 100
REPLAY_BUFFER_BASE_SIZE = 2000
REPLAY_BUFFER_MIN_SIZE = 2000
TRAIN_TOTAL_EPISODES_OVERRIDE = 2000


def rooms_for(map_size: int, base_count: int = ROOMS_BASE_COUNT) -> int:
    ratio = (map_size / TRAIN_MAP_SIZE) ** 2
    return max(2, int(base_count * ratio))


def num_obstacles_for(map_size: int, base_count: int = NUM_OBSTACLES_BASE) -> int:
    return int(base_count * map_size / MAP_TEMPLATE_SIZE)


def replay_buffer_size_for(map_size: int) -> int:
    scale = (map_size / REPLAY_BUFFER_BASE_MAP_SIZE) ** 2
    return max(REPLAY_BUFFER_MIN_SIZE, int(round(REPLAY_BUFFER_BASE_SIZE * scale)))


def max_episodes_for(map_size: int, base_size: int = 100, base_episodes: int = TRAIN_TOTAL_EPISODES_OVERRIDE) -> int:
    return int(base_episodes)


def max_macro_steps_for(map_size: int, base_size: int = 100, base_steps: int = TRAIN_MAX_MACRO_STEPS) -> int:
    scale = map_size / base_size
    return max(base_steps, int(round(base_steps * scale)))
