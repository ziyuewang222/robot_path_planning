# 这个文件是常数配置文件，修改这里就可以修改训练和测试的参数

from pathlib import Path
from datetime import datetime

# 实验命名会自动拼接这些信息，便于区分不同训练任务
EXPERIMENT_PREFIX = "spatial_fcn_rl"

# ---------------------------------------------------------------------------
# 项目内统一输出目录
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
CHECKPOINTS_ROOT = ARTIFACTS_ROOT / "checkpoints"
FUSION_WEIGHTS_ROOT = ARTIFACTS_ROOT / "fusion_weights"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"
METRICS_ROOT = ARTIFACTS_ROOT / "metrics"
CONFIG_SNAPSHOTS_ROOT = ARTIFACTS_ROOT / "configs"
CURRENT_RUN_NAME = f"{EXPERIMENT_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
CURRENT_RUN_DIR = RUNS_ROOT / CURRENT_RUN_NAME
CHECKPOINTS_DIR = CURRENT_RUN_DIR / "checkpoints"
FUSION_WEIGHTS_DIR = CURRENT_RUN_DIR / "fusion_weights"
LOGS_DIR = CURRENT_RUN_DIR / "logs"
METRICS_DIR = CURRENT_RUN_DIR / "metrics"
CONFIG_SNAPSHOTS_DIR = CURRENT_RUN_DIR / "configs"
LATEST_CHECKPOINT_PATH = CHECKPOINTS_ROOT / f"{EXPERIMENT_PREFIX}_latest.pth"
BEST_CHECKPOINT_PATH = CHECKPOINTS_ROOT / f"{EXPERIMENT_PREFIX}_best.pth"
LATEST_FUSION_WEIGHTS_PATH = FUSION_WEIGHTS_ROOT / f"{EXPERIMENT_PREFIX}_fusion_latest.pth"
BEST_FUSION_WEIGHTS_PATH = FUSION_WEIGHTS_ROOT / f"{EXPERIMENT_PREFIX}_fusion_best.pth"

# ---------------------------------------------------------------------------
# derived_train.py 派生训练初始化接口
# 默认继承 artifacts 下全局 latest 主策略权重与 latest 融合权重。
# 如果希望从某个 run 的 best 权重派生训练，可改成：
# RUNS_ROOT / "某次run目录" / "checkpoints" / "best_checkpoint.pth"
# RUNS_ROOT / "某次run目录" / "fusion_weights" / "best_fusion_weights.pth"
# ---------------------------------------------------------------------------

DERIVED_INITIAL_CHECKPOINT_PATH = LATEST_CHECKPOINT_PATH
DERIVED_INITIAL_FUSION_WEIGHTS_PATH = LATEST_FUSION_WEIGHTS_PATH
DERIVED_EPSILON_START = 0.5
DERIVED_EPSILON_END = 0.1
DERIVED_EPSILON_CONVERGE_RATIO = 0.3

# ---------------------------------------------------------------------------
# 测试/评估脚本权重接口
# compare_batch.py、compare_eva.py、eval_policy.py 默认都从这里读取权重。
# 默认使用 artifacts 下全局 latest；如需评估 best 或某个 run 的权重，改这里即可。
# ---------------------------------------------------------------------------

TEST_CHECKPOINT_PATH = LATEST_CHECKPOINT_PATH
TEST_FUSION_WEIGHTS_PATH = LATEST_FUSION_WEIGHTS_PATH

LEGACY_ROOT = PROJECT_ROOT / "legacy"
LEGACY_CODE_DIR = LEGACY_ROOT / "prior_rubbish"

# 训练集固定尺寸（训练权重时使用，改这一处即可）
TRAIN_MAP_SIZE = 200

# 测试/部署地图尺寸（真实地图尺寸，改这一处即可）
TEST_MAP_SIZE = 300

# 传感器视距（栅格），不随地图尺寸变化
SENSOR_RANGE = 12

# 状态张量中小车高斯热力图的 sigma，不随地图尺寸变化
SIGMA = 3.0

# 奖励函数：弱化面积收益，强化少走路，成功到达 95% 探索率给一次性奖励
AREA_REWARD_SCALE = 0.1
STEP_PENALTY = 0.5
COLLISION_PENALTY = 50.0
# 按地图面积缩放成功奖励；0.075 在 200x200 地图上对应 3000 的一次性奖励。
SUCCESS_BONUS_PER_AREA = 0.075

TRAIN_MAX_MACRO_STEPS = 60
TEST_SEED = 4

# 地图生成器：运行前修改此项切换逻辑
#   "room"         -> 固定尺寸房间 + 走廊 + 房间内零星障碍物
#   "dynamic_room" -> 房间尺寸随地图变大，较大房间内含桌子/隔断类障碍物
#   "random"       -> 固定拓扑 + 全图随机散落障碍物
MAP_GENERATOR = "dynamic_room"

# ---------------------------------------------------------------------------
# 房间-走廊地图生成
# ---------------------------------------------------------------------------

CORRIDOR_WIDTH = 5
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


def max_episodes_for(map_size: int, base_size: int = 100, base_episodes: int = 120) -> int:
    """训练 Episode 数随地图边长线性放大，100 尺寸对应 120 次。"""
    scale = map_size / base_size
    return max(base_episodes, int(round(base_episodes * scale)))


def max_macro_steps_for(map_size: int, base_size: int = 100, base_steps: int = TRAIN_MAX_MACRO_STEPS) -> int:
    """每个 Episode 的最大宏动作数随地图边长线性放大，100 尺寸对应 60 步。"""
    scale = map_size / base_size
    return max(base_steps, int(round(base_steps * scale)))
