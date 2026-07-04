# 这个文件是常数配置文件，修改这里就可以修改训练和测试的参数

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
# 房间-走廊地图生成（固定像素尺寸，不随地图等比放大）
# 地图变大时仅增加房间数量；房间/走廊/障碍物尺寸保持不变
# ---------------------------------------------------------------------------

# 走廊宽度（栅格）
CORRIDOR_WIDTH = 3

# 房间内部尺寸范围（不含墙体，单位：栅格）
ROOM_MIN_SIZE = 12
ROOM_MAX_SIZE = 22

# 房间放置：bounding box 之间的最小间距
ROOM_PADDING = 3
ROOM_PLACE_ATTEMPTS = 200

# 基准地图 (TRAIN_MAP_SIZE) 上的目标房间数；地图面积翻倍则房间数约按面积比增加
ROOMS_BASE_COUNT = 6

# 每个房间内的零星障碍物（尺寸固定，不随地图缩放）
OBS_MIN_SIZE = 2
OBS_MAX_SIZE = 4
OBSTACLES_PER_ROOM_MIN = 1
OBSTACLES_PER_ROOM_MAX = 5
OBSTACLE_PLACE_ATTEMPTS = 30
OBSTACLE_WALL_MARGIN = 2

# 出生点距离房间内壁的最小留白
SPAWN_MARGIN = 2


def rooms_for(map_size: int, base_count: int = ROOMS_BASE_COUNT) -> int:
    """地图变大时，按面积比例增加房间数量"""
    ratio = (map_size / TRAIN_MAP_SIZE) ** 2
    return max(2, int(base_count * ratio))


def max_macro_steps_for(map_size: int, base_steps: int = TRAIN_MAX_MACRO_STEPS) -> int:
    """地图变大时，宏观步数上限按比例放大（探索面积增加，步数上限随之增加）"""
    return int(base_steps * (map_size / TRAIN_MAP_SIZE) ** 2)
