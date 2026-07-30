# 这个文件是常数配置文件，修改这里就可以修改训练和测试的参数

from pathlib import Path
from datetime import datetime

# 实验命名会自动拼接这些信息，便于区分不同训练任务
EXPERIMENT_PREFIX = "spatial_fcn_rl"

# ---------------------------------------------------------------------------
# 项目内统一输出目录
# ---------------------------------------------------------------------------
# 说明：regional_legacy 现在只使用本文件夹下的 results 目录保存训练结果。
# 过去 base_on_new 的 artifacts 结构不再作为默认输出位置。
# 使用方式：
#   1) 每次训练会自动生成一个新的 run 子目录：results/runs/<experiment>_<timestamp>/
#   2) 当前 run 的 checkpoint / logs / metrics / configs 都写入该子目录
#   3) LATEST_* 与 BEST_* 指向 results 根目录下的聚合权重，便于评估脚本直接加载
#   4) 如果你想手动指定某次训练结果，直接改下面的 TEST_*/DERIVED_* 路径即可

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
CHECKPOINTS_ROOT = RESULTS_ROOT / "checkpoints"
LOGS_ROOT = RESULTS_ROOT / "logs"
METRICS_ROOT = RESULTS_ROOT / "metrics"
CONFIG_SNAPSHOTS_ROOT = RESULTS_ROOT / "configs"
CURRENT_RUN_NAME = f"{EXPERIMENT_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
CURRENT_RUN_DIR = RUNS_ROOT / CURRENT_RUN_NAME
CHECKPOINTS_DIR = CURRENT_RUN_DIR / "checkpoints"
LOGS_DIR = CURRENT_RUN_DIR / "logs"
METRICS_DIR = CURRENT_RUN_DIR / "metrics"
CONFIG_SNAPSHOTS_DIR = CURRENT_RUN_DIR / "configs"
LATEST_CHECKPOINT_PATH = CHECKPOINTS_ROOT / f"{EXPERIMENT_PREFIX}_latest.pth"
BEST_CHECKPOINT_PATH = CHECKPOINTS_ROOT / f"{EXPERIMENT_PREFIX}_best.pth"

# ---------------------------------------------------------------------------
# derived_train.py 派生训练初始化接口
# 默认继承 results 下全局 latest 主策略权重。
# 如果希望从某个 run 的 best 权重派生训练，可改成：
# RUNS_ROOT / "某次run目录" / "checkpoints" / "best_checkpoint.pth"
# ---------------------------------------------------------------------------

# 探索期占总 Episode 的比例。epsilon 会在前 int(max_episodes * EPSILON_CONVERGE_RATIO) 个 Episode
# 从 epsilon_start 衰减到 epsilon_end；修改这里即可统一控制 train.py 和 derived_train.py 的探索期长度。
EPSILON_CONVERGE_RATIO = 0.3
# 稳定训练期起点占总 Episode 的比例。到达该比例后，epsilon 会从 epsilon_end 继续下降到
# EPSILON_STABILIZE_END，便于后期用更低探索率稳定 Q 网络参数。例如 0.8 表示最后 20% 训练进入稳定期。
EPSILON_STABILIZE_RATIO = 0.8
EPSILON_STABILIZE_END = 0.05
# 回放池尺寸按地图面积缩放：buffer_size = max(REPLAY_BUFFER_MIN_SIZE,
# int(REPLAY_BUFFER_BASE_SIZE * (map_size / REPLAY_BUFFER_BASE_MAP_SIZE) ** 2))。
REPLAY_BUFFER_BASE_MAP_SIZE = 100
REPLAY_BUFFER_BASE_SIZE = 2000
REPLAY_BUFFER_MIN_SIZE = 2000
TRAIN_TOTAL_EPISODES_OVERRIDE = 2000

DERIVED_INITIAL_CHECKPOINT_PATH = LATEST_CHECKPOINT_PATH
# DERIVED_INITIAL_CHECKPOINT_PATH = RUNS_ROOT / "spatial_fcn_rl_20260714_232906" / "checkpoints" / "best_checkpoint.pth"
DERIVED_EPSILON_START = 0.6
DERIVED_EPSILON_END = 0.1
# 派生训练默认沿用全局 EPSILON_CONVERGE_RATIO，保持只改一处即可控制探索期占比。
DERIVED_EPSILON_CONVERGE_RATIO = EPSILON_CONVERGE_RATIO
# 派生训练从已有策略继续微调，学习率应低于从零训练，避免探索期快速破坏原策略。
DERIVED_LEARNING_RATE = 5e-5

# ---------------------------------------------------------------------------
# 运行 / 可视化 / 评估脚本权重接口
# 说明：
#   1) 默认都指向 results 根目录下的 latest 权重
#   2) 如果要手动指定某次 run，只需把下面的 *_PATH 改成对应的 run 路径
#   3) 运行脚本和可视化脚本都应优先调用 resolve_*_path()，不要在别处写死路径
# ---------------------------------------------------------------------------

# 主干 checkpoint：默认 latest；如果你想固定某次实验，改成 RUNS_ROOT / "xxx" / "checkpoints" / "best_checkpoint.pth"
TEST_CHECKPOINT_PATH = LATEST_CHECKPOINT_PATH


def resolve_test_checkpoint_path(explicit_path=None):
    """统一解析测试/可视化所用的 checkpoint 路径。"""
    if explicit_path is not None:
        return explicit_path
    return TEST_CHECKPOINT_PATH


LEGACY_ROOT = PROJECT_ROOT / "legacy"
LEGACY_CODE_DIR = LEGACY_ROOT / "prior_rubbish"

# 训练集固定尺寸（训练权重时使用，改这一处即可）
TRAIN_MAP_SIZE = 200

# 测试/部署地图尺寸（真实地图尺寸，改这一处即可）
TEST_MAP_SIZE = 200

# 统一地图尺寸入口
# 如果你的训练和测试都希望用同一个尺寸，优先直接改下面这个值。
# 代码中凡是需要“当前默认地图尺寸”的地方，都应优先读取它，而不是写死常数。
DEFAULT_MAP_SIZE = TEST_MAP_SIZE

# 传感器视距（栅格），不随地图尺寸变化
SENSOR_RANGE = 12

# 状态张量中小车热力图的 sigma，保留用于兼容旧接口；当前使用归一化曼哈顿距离热力图
SIGMA = 3.0

# 奖励函数：以结构进度为主、面积增益为辅，轻惩罚重复震荡和过早离开。
# 这些常量对应 step_macro 中的默认 reward shaping 权重。
AREA_REWARD_SCALE = 0.035
ROOM_COMPLETION_POTENTIAL_SCALE = 1.8
ROOM_COMPLETION_POTENTIAL_K1 = 10.0
ROOM_COMPLETION_POTENTIAL_K2 = 70.0
ROOM_COMPLETION_POTENTIAL_K3 = 180.0
ROOM_COMPLETION_POTENTIAL_T1 = 0.90
ROOM_COMPLETION_POTENTIAL_T2 = 0.95
ROOM_COMPLETION_BONUS = 180.0
ROOM_COMPLETION_BONUS_EXTRA = 50.0
STEP_PENALTY = 0.9
# 路径级已知区域惩罚：每经过一个已知空地就给轻微额外惩罚，抑制绕路和回踩。
PATH_KNOWN_CELL_PENALTY_SCALE = 0.35
COLLISION_PENALTY = 40.0
REPEAT_PENALTY_SCALE = 1.0
EARLY_EXIT_PENALTY_BASE = 10.0
EARLY_EXIT_PENALTY_UNFINISHED = 30.0
EARLY_EXIT_PENALTY_FINISHED = 3.0
# UNFINISHED_FRONTIER_PENALTY_BASE = 6.0
# UNFINISHED_FRONTIER_PENALTY_PER_FRONTIER = 1.6
# UNFINISHED_FRONTIER_GATED_SCALE = 0.75
# FRONTIER_JUMP_PENALTY_BASE = 5.0
# FRONTIER_JUMP_PENALTY_SCALE = 0.22
# COMPLETED_ROOM_REVISIT_PENALTY = 4.0
# ABANDONED_RETURN_PENALTY = 8.0
RECENT_REPEAT_PENALTY_SCALE = 2.2
# ROOM_REPEAT_PENALTY_PROGRESS_SCALE = 3.0
# ROOM_REPEAT_PENALTY_POTENTIAL_SCALE = 4.0
# 按地图面积缩放成功奖励，并在训练入口处设置上限，避免终点奖励主导 TD target.
SUCCESS_BONUS_PER_AREA = 0.0008

# ---------------------------------------------------------------------------
# 传统启发式前沿选择权重
# ---------------------------------------------------------------------------
# 仅用于 legacy/core/heuristic.py 中的 frontier 评分融合。
HEURISTIC_BFS_WEIGHT = 10
HEURISTIC_FRONTIER_CLASS_PENALTY = 100
HEURISTIC_FRONTIER_CLASS_COVERAGE_THRESHOLD = 0.93
HEURISTIC_TURN_WEIGHT = 1
HEURISTIC_OBS_WEIGHT = 8

TRAIN_MAX_MACRO_STEPS = 60
TEST_SEED = 41
# 探索完成阈值：已探索自由栅格占 ground_truth 中自由栅格的比例达到该值时，认为本轮探索完成
EXPLORATION_SUCCESS_THRESHOLD = 0.995
# 效果好的随机种子：55010，1,4(双好),7，8，9，10（双好）11（双好）1-11全胜收官，43（双好）
# 地图生成器：运行前修改此项切换逻辑
#   "room"            -> 固定尺寸房间 + 走廊 + 房间内零星障碍物
#   "dynamic_room"    -> 房间尺寸随地图变大，较大房间内含桌子/隔断类障碍物
#   "easy_big_room"   -> 3 个大房间，拓扑更简单，少量桌椅式障碍物
#   "random"          -> 固定拓扑 + 全图随机散落障碍物
MAP_GENERATOR = "easy_big_room"


# ---------------------------------------------------------------------------
# 房间-走廊地图生成
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

# ---------------------------------------------------------------------------
# 动态房间地图生成：下限固定为 ROOM_MIN_SIZE，上限随地图尺寸增长
# 200 尺寸地图中，房间上限约为 60，因而可随机出现 50-60 的大房间
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 随机散落障碍物地图（固定拓扑 + 全图随机障碍物）
# ---------------------------------------------------------------------------

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


def rooms_for(map_size: int, base_count: int = ROOMS_BASE_COUNT) -> int:
    """地图变大时，按面积比例增加房间数量"""
    ratio = (map_size / TRAIN_MAP_SIZE) ** 2
    return max(2, int(base_count * ratio))


def num_obstacles_for(map_size: int, base_count: int = NUM_OBSTACLES_BASE) -> int:
    """随机地图：障碍物数量随地图尺寸等比增加"""
    return int(base_count * map_size / MAP_TEMPLATE_SIZE)


def replay_buffer_size_for(map_size: int) -> int:
    """回放池容量随地图面积放大，避免大地图训练时过快遗忘早期经验。"""
    scale = (map_size / REPLAY_BUFFER_BASE_MAP_SIZE) ** 2
    return max(REPLAY_BUFFER_MIN_SIZE, int(round(REPLAY_BUFFER_BASE_SIZE * scale)))


def max_episodes_for(map_size: int, base_size: int = 100, base_episodes: int = TRAIN_TOTAL_EPISODES_OVERRIDE) -> int:
    """训练 Episode 数默认按全局目标覆盖为 2000；如需恢复线性缩放可改回旧逻辑。"""
    return int(base_episodes)


def max_macro_steps_for(map_size: int, base_size: int = 100, base_steps: int = TRAIN_MAX_MACRO_STEPS) -> int:
    """每个 Episode 的最大宏动作数随地图边长线性放大，100 尺寸对应 60 步。"""
    scale = map_size / base_size
    return max(base_steps, int(round(base_steps * scale)))
