import numpy as np

from config import (
    CORRIDOR_WIDTH,
    ROOM_MIN_SIZE,
    ROOM_MAX_SIZE,
    ROOM_PADDING,
    ROOM_PLACE_ATTEMPTS,
    OBS_MIN_SIZE,
    OBS_MAX_SIZE,
    OBSTACLES_PER_ROOM_MIN,
    OBSTACLES_PER_ROOM_MAX,
    OBSTACLE_PLACE_ATTEMPTS,
    OBSTACLE_WALL_MARGIN,
    SPAWN_MARGIN,
    rooms_for,
)


def build(map_size):
    """生成房间-走廊地图，返回 ground_truth 与 rooms 元数据。"""
    ground_truth = np.ones((map_size, map_size), dtype=np.int8)

    rooms = _generate_rooms(ground_truth, map_size)
    for room in rooms:
        _carve_room(ground_truth, room)
    _connect_rooms(ground_truth, map_size, rooms)
    for room in rooms:
        _place_room_obstacles(ground_truth, room)

    return ground_truth, {"rooms": rooms}


def sample_spawn(ground_truth, meta):
    rooms = meta.get("rooms", [])
    if not rooms:
        raise RuntimeError("房间地图生成失败：未放置任何房间")

    for _ in range(500):
        room = rooms[np.random.randint(len(rooms))]
        x, y, w, h = room
        x_lo = x + 1 + SPAWN_MARGIN
        x_hi = x + w - SPAWN_MARGIN
        y_lo = y + 1 + SPAWN_MARGIN
        y_hi = y + h - SPAWN_MARGIN
        if x_lo >= x_hi or y_lo >= y_hi:
            continue

        px = np.random.randint(x_lo, x_hi)
        py = np.random.randint(y_lo, y_hi)
        if np.all(ground_truth[px - 1:px + 2, py - 1:py + 2] == 0):
            return px, py

    x, y, w, h = rooms[0]
    return x + 1 + w // 2, y + 1 + h // 2


def _generate_rooms(ground_truth, map_size):
    target = rooms_for(map_size)
    rooms = []
    for _ in range(target):
        for _ in range(ROOM_PLACE_ATTEMPTS):
            w = np.random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE + 1)
            h = np.random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE + 1)
            x_max = map_size - w - 3
            y_max = map_size - h - 3
            if x_max < 1 or y_max < 1:
                break

            x = np.random.randint(1, x_max + 1)
            y = np.random.randint(1, y_max + 1)
            room = (x, y, w, h)
            if not _room_overlaps(room, rooms):
                rooms.append(room)
                break
    return rooms


def _room_bbox(room):
    x, y, w, h = room
    return x, y, x + w + 2, y + h + 2


def _room_overlaps(room, rooms):
    ax1, ay1, ax2, ay2 = _room_bbox(room)
    pad = ROOM_PADDING
    for other in rooms:
        bx1, by1, bx2, by2 = _room_bbox(other)
        if not (ax2 + pad <= bx1 or bx2 + pad <= ax1 or ay2 + pad <= by1 or by2 + pad <= ay1):
            return True
    return False


def _carve_room(ground_truth, room):
    x, y, w, h = room
    ground_truth[x + 1:x + 1 + w, y + 1:y + 1 + h] = 0


def _room_center(room):
    x, y, w, h = room
    return x + 1 + w // 2, y + 1 + h // 2


def _connect_rooms(ground_truth, map_size, rooms):
    if len(rooms) < 2:
        return

    centers = [_room_center(room) for room in rooms]
    in_tree = {0}
    while len(in_tree) < len(rooms):
        best_dist = None
        best_pair = None
        for i in in_tree:
            for j in range(len(rooms)):
                if j in in_tree:
                    continue
                dist = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        i, j = best_pair
        _carve_corridor(ground_truth, map_size, centers[i], centers[j])
        in_tree.add(j)


def _carve_corridor(ground_truth, map_size, start, end):
    x1, y1 = start
    x2, y2 = end
    half = CORRIDOR_WIDTH // 2
    if np.random.rand() < 0.5:
        _carve_horizontal_corridor(ground_truth, map_size, x1, x2, y1, half)
        _carve_vertical_corridor(ground_truth, map_size, y1, y2, x2, half)
    else:
        _carve_vertical_corridor(ground_truth, map_size, y1, y2, x1, half)
        _carve_horizontal_corridor(ground_truth, map_size, x1, x2, y2, half)


def _carve_horizontal_corridor(ground_truth, map_size, x1, x2, y, half):
    xa, xb = sorted((x1, x2))
    y0 = max(1, y - half)
    y1 = min(map_size - 1, y - half + CORRIDOR_WIDTH)
    ground_truth[xa:xb + 1, y0:y1] = 0


def _carve_vertical_corridor(ground_truth, map_size, y1, y2, x, half):
    ya, yb = sorted((y1, y2))
    x0 = max(1, x - half)
    x1 = min(map_size - 1, x - half + CORRIDOR_WIDTH)
    ground_truth[x0:x1, ya:yb + 1] = 0


def _place_room_obstacles(ground_truth, room):
    x, y, w, h = room
    n_obs = np.random.randint(OBSTACLES_PER_ROOM_MIN, OBSTACLES_PER_ROOM_MAX + 1)
    ix1 = x + 1 + OBSTACLE_WALL_MARGIN
    iy1 = y + 1 + OBSTACLE_WALL_MARGIN
    ix2 = x + w - OBSTACLE_WALL_MARGIN
    iy2 = y + h - OBSTACLE_WALL_MARGIN
    if ix2 <= ix1 or iy2 <= iy1:
        return

    for _ in range(n_obs):
        for _ in range(OBSTACLE_PLACE_ATTEMPTS):
            ow = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
            oh = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
            if ix2 - ow + 1 < ix1 or iy2 - oh + 1 < iy1:
                break

            rx = np.random.randint(ix1, ix2 - ow + 2)
            ry = np.random.randint(iy1, iy2 - oh + 2)
            if np.all(ground_truth[rx:rx + ow, ry:ry + oh] == 0):
                ground_truth[rx:rx + ow, ry:ry + oh] = 1
                break
