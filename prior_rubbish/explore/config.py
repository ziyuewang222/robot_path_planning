# 默认环境与地图参数；单次训练超参由 scripts/train.py + runs/<name>/config.json 覆盖

TRAIN_MAP_SIZE = 100
TEST_MAP_SIZE = 100
SENSOR_RANGE = 12
SIGMA = 3.0

TRAIN_MAX_MACRO_STEPS = 60
TEST_SEED = 888

MAP_GENERATOR = "room"

# --- room map ---
CORRIDOR_WIDTH = 3
ROOM_MIN_SIZE = 12
ROOM_MAX_SIZE = 22
ROOM_PADDING = 3
ROOM_PLACE_ATTEMPTS = 200
ROOMS_BASE_COUNT = 6

OBS_MIN_SIZE = 2
OBS_MAX_SIZE = 4
OBSTACLES_PER_ROOM_MIN = 1
OBSTACLES_PER_ROOM_MAX = 5
OBSTACLE_PLACE_ATTEMPTS = 30
OBSTACLE_WALL_MARGIN = 2
SPAWN_MARGIN = 2

# --- random map ---
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
    ratio = (map_size / TRAIN_MAP_SIZE) ** 2
    return max(2, int(base_count * ratio))


def num_obstacles_for(map_size: int, base_count: int = NUM_OBSTACLES_BASE) -> int:
    return int(base_count * map_size / MAP_TEMPLATE_SIZE)


def max_macro_steps_for(map_size: int, base_steps: int = TRAIN_MAX_MACRO_STEPS) -> int:
    return int(base_steps * (map_size / TRAIN_MAP_SIZE) ** 2)
