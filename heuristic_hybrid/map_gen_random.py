import numpy as np

from config import (
    MAP_TEMPLATE_SIZE,
    WALL_WIDTH,
    DOOR_WIDTH,
    DOOR_PROTECT_MARGIN,
    OBS_MIN_SIZE,
    OBS_MAX_SIZE,
    NUM_OBSTACLES_BASE,
    ROW_MID1,
    ROW_MID2,
    COL_MID1,
    COL_MID2,
    DOOR_POSITIONS,
    SPAWN_X_MIN,
    SPAWN_X_MAX,
    SPAWN_Y_MIN,
    SPAWN_Y_MAX,
    num_obstacles_for,
)


def build(map_size):
    """生成固定拓扑 + 随机散落障碍物的地图。"""
    ground_truth = np.zeros((map_size, map_size), dtype=np.int8)
    template = min(MAP_TEMPLATE_SIZE, map_size)

    ground_truth[0, :] = 1
    ground_truth[-1, :] = 1
    ground_truth[:, 0] = 1
    ground_truth[:, -1] = 1

    ground_truth[ROW_MID1:ROW_MID1 + WALL_WIDTH, 1:template - 1] = 1
    ground_truth[ROW_MID2:ROW_MID2 + WALL_WIDTH, 1:template - 1] = 1
    ground_truth[1:ROW_MID1, COL_MID1:COL_MID1 + WALL_WIDTH] = 1
    ground_truth[ROW_MID1:template - 1, COL_MID2:COL_MID2 + WALL_WIDTH] = 1

    protected_mask = np.zeros((map_size, map_size), dtype=bool)
    for r, c in DOOR_POSITIONS:
        if r >= template or c >= template:
            continue
        r_start = max(1, r - DOOR_WIDTH // 2)
        r_end = min(template - 1, r + DOOR_WIDTH)
        c_start = max(1, c - DOOR_WIDTH // 2)
        c_end = min(template - 1, c + DOOR_WIDTH)
        ground_truth[r_start:r_end, c_start:c_end] = 0
        protected_mask[
            max(0, r_start - DOOR_PROTECT_MARGIN):min(map_size, r_end + DOOR_PROTECT_MARGIN),
            max(0, c_start - DOOR_PROTECT_MARGIN):min(map_size, c_end + DOOR_PROTECT_MARGIN),
        ] = True

    obs_lo = OBS_MAX_SIZE + 2
    obs_hi = map_size - OBS_MAX_SIZE - 2
    if obs_hi > obs_lo:
        for _ in range(num_obstacles_for(map_size)):
            rx = np.random.randint(obs_lo, obs_hi)
            ry = np.random.randint(obs_lo, obs_hi)
            w = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
            h = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
            if rx + w >= map_size or ry + h >= map_size:
                continue
            target_slice = protected_mask[rx:rx + w, ry:ry + h]
            if not np.any(target_slice) and np.all(ground_truth[rx:rx + w, ry:ry + h] == 0):
                ground_truth[rx:rx + w, ry:ry + h] = 1

    return ground_truth, {}


def sample_spawn(ground_truth, meta, map_size):
    x_lo = max(1, SPAWN_X_MIN)
    x_hi = min(map_size - 2, SPAWN_X_MAX)
    y_lo = max(1, SPAWN_Y_MIN)
    y_hi = min(map_size - 2, SPAWN_Y_MAX)

    while True:
        x = np.random.randint(x_lo, x_hi)
        y = np.random.randint(y_lo, y_hi)
        if np.all(ground_truth[x - 1:x + 2, y - 1:y + 2] == 0):
            return x, y
