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
LEGACY_ROOT = PROJECT_ROOT / "legacy"
LEGACY_CODE_DIR = LEGACY_ROOT / "prior_rubbish"

# 训练集固定尺寸（训练权重时使用，改这一处即可）
TRAIN_MAP_SIZE = 200

# 测试/部署地图尺寸（真实地图尺寸，改这一处即可）
TEST_MAP_SIZE = 500

# 传感器视距（栅格），不随地图尺寸变化
SENSOR_RANGE = 12

# 状态张量中小车高斯热力图的 sigma，不随地图尺寸变化
SIGMA = 3.0

TRAIN_MAX_MACRO_STEPS = 60
TEST_SEED = 888

# 地图生成器：运行前修改此项切换逻辑
#   "room"   -> 房间 + 走廊 + 房间内零星障碍物
#   "random" -> 固定拓扑 + 全图随机散落障碍物
MAP_GENERATOR = "room"

# ---------------------------------------------------------------------------
# 房间-走廊地图生成（固定像素尺寸，不随地图等比放大）
# 地图变大时仅增加房间数量；房间/走廊/障碍物尺寸保持不变
# ---------------------------------------------------------------------------

CORRIDOR_WIDTH = 5
ROOM_MIN_SIZE = 22
ROOM_MAX_SIZE = 32
ROOM_PADDING = 3
ROOM_PLACE_ATTEMPTS = 200
ROOMS_BASE_COUNT = 4

OBS_MIN_SIZE = 2
OBS_MAX_SIZE = 4
OBSTACLES_PER_ROOM_MIN = 1
OBSTACLES_PER_ROOM_MAX = 5
OBSTACLE_PLACE_ATTEMPTS = 30
OBSTACLE_WALL_MARGIN = 2
SPAWN_MARGIN = 2

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


def max_macro_steps_for(map_size: int, base_steps: int = TRAIN_MAX_MACRO_STEPS) -> int:
    """地图变大时，宏观步数上限按比例放大（探索面积增加，步数上限随之增加）"""
    return int(base_steps * (map_size / TRAIN_MAP_SIZE) ** 2)
