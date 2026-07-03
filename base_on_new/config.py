# 训练集固定尺寸（训练权重时使用，改这一处即可）
TRAIN_MAP_SIZE = 100

# 测试/部署地图尺寸（真实地图尺寸，改这一处即可）
TEST_MAP_SIZE = 200

# 传感器视距（栅格），不随地图尺寸变化
SENSOR_RANGE = 12

# 状态张量中小车高斯热力图的 sigma，不随地图尺寸变化
SIGMA = 3.0

TRAIN_MAX_MACRO_STEPS = 60
TEST_SEED = 888

# ---------------------------------------------------------------------------
# 地图拓扑固定参数（像素级常量，地图变大时结构不变，仅外围空地扩大）
# 以下坐标均基于 MAP_TEMPLATE_SIZE 网格设计
# ---------------------------------------------------------------------------
MAP_TEMPLATE_SIZE = 100

WALL_WIDTH = 2
DOOR_WIDTH = 6
DOOR_PROTECT_MARGIN = 3

OBS_MIN_SIZE = 2
OBS_MAX_SIZE = 4
NUM_OBSTACLES = 45

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


def max_macro_steps_for(map_size: int, base_steps: int = TRAIN_MAX_MACRO_STEPS) -> int:
    """地图变大时，宏观步数上限按比例放大（探索面积增加，步数上限随之增加）"""
    return int(base_steps * (map_size / TRAIN_MAP_SIZE) ** 2)
