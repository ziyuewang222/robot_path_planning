# 新建地图逻辑：动态房间

import numpy as np

from config import (
    MAP_GENERATOR,
    CORRIDOR_WIDTH,
    ROOM_MIN_SIZE,
    ROOM_PADDING,
    ROOM_PLACE_ATTEMPTS,
    OBS_MIN_SIZE,
    OBS_MAX_SIZE,
    OBSTACLE_PLACE_ATTEMPTS,
    OBSTACLE_WALL_MARGIN,
    SPAWN_MARGIN,
    DYNAMIC_ROOM_MAX_SIZE_RATIO,
    DYNAMIC_ROOM_MAX_SIZE_CAP,
    DYNAMIC_ROOM_LARGE_THRESHOLD,
    DYNAMIC_ROOM_LARGE_ROOM_PROB,
    DYNAMIC_ROOM_OBSTACLE_DENSITY,
    DYNAMIC_ROOM_PARTITION_MIN_LEN,
    DYNAMIC_ROOM_PARTITION_MAX_LEN_RATIO,
    DYNAMIC_ROOM_PARTITION_THICKNESS_MIN,
    DYNAMIC_ROOM_PARTITION_THICKNESS_MAX,
    DYNAMIC_ROOM_PARTITIONS_PER_LARGE_ROOM_MIN,
    DYNAMIC_ROOM_PARTITIONS_PER_LARGE_ROOM_MAX,
    DYNAMIC_ROOM_GENERATION_ATTEMPTS,
    DYNAMIC_ROOM_OBSERVABLE_AREA_RATIO,
    rooms_for,
)
import map_gen_random
import map_gen_room

GENERATORS = {
    "room": {
        "build": map_gen_room.build,
        "sample_spawn": map_gen_room.sample_spawn,
    },
    "dynamic_room": {
        "build": lambda map_size: _build_dynamic_room(map_size),
        "sample_spawn": lambda ground_truth, meta: _sample_dynamic_room_spawn(ground_truth, meta),
    },
    "random": {
        "build": map_gen_random.build,
        "sample_spawn": map_gen_random.sample_spawn,
    },
}


def available_generators():
    return tuple(GENERATORS.keys())


def resolve_generator(name=None):
    generator = name or MAP_GENERATOR
    if generator not in GENERATORS:
        raise ValueError(
            f"未知地图生成器 '{generator}'，可选: {', '.join(available_generators())}"
        )
    return generator


def build_map(map_size, generator=None):
    name = resolve_generator(generator)
    ground_truth, meta = GENERATORS[name]["build"](map_size)
    return ground_truth, meta, name


def sample_spawn(ground_truth, meta, map_size, generator=None):
    name = resolve_generator(generator)
    spawn_fn = GENERATORS[name]["sample_spawn"]
    if name == "random":
        return spawn_fn(ground_truth, meta, map_size)
    return spawn_fn(ground_truth, meta)


def dynamic_room_max_size(map_size):
    scaled = int(round(map_size * DYNAMIC_ROOM_MAX_SIZE_RATIO))
    return max(ROOM_MIN_SIZE, min(DYNAMIC_ROOM_MAX_SIZE_CAP, scaled))


def _build_dynamic_room(map_size):
    """生成尺寸自适应房间地图，返回 ground_truth 与 rooms 元数据。"""
    target_observable_area = _target_dynamic_room_observable_area(map_size)
    best_candidate = None
    best_diff = None

    for _ in range(DYNAMIC_ROOM_GENERATION_ATTEMPTS):
        ground_truth = np.ones((map_size, map_size), dtype=np.int8)

        rooms = _generate_dynamic_rooms(map_size)
        for room in rooms:
            _carve_dynamic_room(ground_truth, room)

        room_graph = _connect_dynamic_rooms(ground_truth, map_size, rooms)
        for room in rooms:
            _place_dynamic_room_obstacles(ground_truth, room)

        observable_area = _dynamic_observable_area(ground_truth)
        free_area = int(np.sum(ground_truth == 0))
        diff = abs(observable_area - target_observable_area)
        candidate = (ground_truth, rooms, room_graph, observable_area, free_area)
        if best_diff is None or diff < best_diff:
            best_candidate = candidate
            best_diff = diff
        if diff == 0:
            break

    ground_truth, rooms, room_graph, observable_area, free_area = best_candidate
    for room in rooms:
        room.setdefault("degree", 0)
        room.setdefault("neighbors", set())

    return ground_truth, {
        "rooms": rooms,
        "room_graph": room_graph,
        "target_observable_area": target_observable_area,
        "observable_area": observable_area,
        "free_area": free_area,
        "room_max_size": dynamic_room_max_size(map_size),
    }


def _sample_dynamic_room_spawn(ground_truth, meta):
    rooms = meta.get("rooms", [])
    if not rooms:
        raise RuntimeError("动态房间地图生成失败：未放置任何房间")

    preferred_rooms = [room for room in rooms if room.get("degree", 0) >= 2]
    candidate_rooms = preferred_rooms if preferred_rooms else rooms

    for _ in range(1000):
        room = candidate_rooms[np.random.randint(len(candidate_rooms))]
        x, y, w, h = room["bbox"]
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

    free_x, free_y = np.where(ground_truth == 0)
    if len(free_x) == 0:
        raise RuntimeError("动态房间地图生成失败：没有可出生的自由区域")
    idx = np.random.randint(len(free_x))
    return int(free_x[idx]), int(free_y[idx])


def _target_dynamic_room_observable_area(map_size):
    return int(round(map_size * map_size * DYNAMIC_ROOM_OBSERVABLE_AREA_RATIO))


def _dynamic_observable_area(ground_truth):
    free = ground_truth == 0
    observable = free.copy()
    observable[1:, :] |= free[:-1, :]
    observable[:-1, :] |= free[1:, :]
    observable[:, 1:] |= free[:, :-1]
    observable[:, :-1] |= free[:, 1:]
    return int(np.sum(observable))


def _generate_dynamic_rooms(map_size):
    target = rooms_for(map_size)
    rooms = []
    max_size = dynamic_room_max_size(map_size)
    large_min = min(max_size, max(ROOM_MIN_SIZE, DYNAMIC_ROOM_LARGE_THRESHOLD))

    for _ in range(target):
        placed = False
        for _ in range(ROOM_PLACE_ATTEMPTS):
            room = _sample_dynamic_room(map_size, max_size, large_min)
            if room is None:
                break
            if not _dynamic_room_overlaps(room, rooms):
                rooms.append(room)
                placed = True
                break
        if not placed and rooms:
            continue
    return rooms


def _sample_dynamic_room(map_size, max_size, large_min):
    use_large = max_size >= DYNAMIC_ROOM_LARGE_THRESHOLD and np.random.rand() < DYNAMIC_ROOM_LARGE_ROOM_PROB
    low = large_min if use_large else ROOM_MIN_SIZE
    high = max_size if use_large else max(ROOM_MIN_SIZE, min(max_size, large_min - 1))
    if high < low:
        low, high = ROOM_MIN_SIZE, max_size

    w = np.random.randint(low, high + 1)
    h = np.random.randint(low, high + 1)
    if max(w, h) / max(1, min(w, h)) > 1.35:
        side = np.random.randint(low, high + 1)
        if np.random.rand() < 0.5:
            w = side
        else:
            h = side

    x_max = map_size - w - 3
    y_max = map_size - h - 3
    if x_max < 1 or y_max < 1:
        return None

    x = np.random.randint(1, x_max + 1)
    y = np.random.randint(1, y_max + 1)
    return {
        "bbox": (x, y, w, h),
        "degree": 0,
        "neighbors": set(),
        "is_large": max(w, h) >= DYNAMIC_ROOM_LARGE_THRESHOLD,
    }


def _dynamic_room_bbox(room):
    x, y, w, h = room["bbox"]
    return x, y, x + w + 2, y + h + 2


def _dynamic_room_overlaps(room, rooms):
    ax1, ay1, ax2, ay2 = _dynamic_room_bbox(room)
    for other in rooms:
        bx1, by1, bx2, by2 = _dynamic_room_bbox(other)
        if not (
            ax2 + ROOM_PADDING <= bx1
            or bx2 + ROOM_PADDING <= ax1
            or ay2 + ROOM_PADDING <= by1
            or by2 + ROOM_PADDING <= ay1
        ):
            return True
    return False


def _carve_dynamic_room(ground_truth, room):
    x, y, w, h = room["bbox"]
    ground_truth[x + 1:x + 1 + w, y + 1:y + 1 + h] = 0


def _dynamic_room_center(room):
    x, y, w, h = room["bbox"]
    return x + 1 + w // 2, y + 1 + h // 2


def _connect_dynamic_rooms(ground_truth, map_size, rooms):
    if len(rooms) < 2:
        return []

    centers = [_dynamic_room_center(room) for room in rooms]
    edges = set()
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
        _carve_dynamic_corridor(ground_truth, map_size, centers[i], centers[j])
        _add_dynamic_room_edge(rooms, edges, i, j)
        in_tree.add(j)

    extra_edges = max(1, len(rooms) // 3)
    candidates = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if (i, j) not in edges:
                dist = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
                candidates.append((dist, i, j))
    candidates.sort(key=lambda item: item[0])

    added = 0
    for _, i, j in candidates:
        if added >= extra_edges:
            break
        if abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1]) > map_size // 2:
            continue
        _carve_dynamic_corridor(ground_truth, map_size, centers[i], centers[j])
        _add_dynamic_room_edge(rooms, edges, i, j)
        added += 1

    return list(edges)


def _add_dynamic_room_edge(rooms, edges, i, j):
    rooms[i]["degree"] += 1
    rooms[j]["degree"] += 1
    rooms[i]["neighbors"].add(j)
    rooms[j]["neighbors"].add(i)
    edges.add(tuple(sorted((i, j))))


def _carve_dynamic_corridor(ground_truth, map_size, start, end):
    x1, y1 = start
    x2, y2 = end
    half = CORRIDOR_WIDTH // 2
    if np.random.rand() < 0.5:
        _carve_dynamic_horizontal_corridor(ground_truth, map_size, x1, x2, y1, half)
        _carve_dynamic_vertical_corridor(ground_truth, map_size, y1, y2, x2, half)
    else:
        _carve_dynamic_vertical_corridor(ground_truth, map_size, y1, y2, x1, half)
        _carve_dynamic_horizontal_corridor(ground_truth, map_size, x1, x2, y2, half)


def _carve_dynamic_horizontal_corridor(ground_truth, map_size, x1, x2, y, half):
    xa, xb = sorted((x1, x2))
    y0 = max(1, y - half)
    y1 = min(map_size - 1, y - half + CORRIDOR_WIDTH)
    ground_truth[xa:xb + 1, y0:y1] = 0


def _carve_dynamic_vertical_corridor(ground_truth, map_size, y1, y2, x, half):
    ya, yb = sorted((y1, y2))
    x0 = max(1, x - half)
    x1 = min(map_size - 1, x - half + CORRIDOR_WIDTH)
    ground_truth[x0:x1, ya:yb + 1] = 0


def _place_dynamic_room_obstacles(ground_truth, room):
    x, y, w, h = room["bbox"]
    area = w * h
    n_obs = max(1, int(round(area * DYNAMIC_ROOM_OBSTACLE_DENSITY)))
    if room.get("is_large", False):
        n_partitions = np.random.randint(
            DYNAMIC_ROOM_PARTITIONS_PER_LARGE_ROOM_MIN,
            DYNAMIC_ROOM_PARTITIONS_PER_LARGE_ROOM_MAX + 1,
        )
        for _ in range(n_partitions):
            _place_dynamic_partition(ground_truth, room)

    for _ in range(n_obs):
        _place_dynamic_block_obstacle(ground_truth, room)


def _place_dynamic_partition(ground_truth, room):
    x, y, w, h = room["bbox"]
    horizontal = np.random.rand() < 0.5
    length_cap = int(round((w if horizontal else h) * DYNAMIC_ROOM_PARTITION_MAX_LEN_RATIO))
    max_len = max(DYNAMIC_ROOM_PARTITION_MIN_LEN, length_cap)
    thickness = np.random.randint(
        DYNAMIC_ROOM_PARTITION_THICKNESS_MIN,
        DYNAMIC_ROOM_PARTITION_THICKNESS_MAX + 1,
    )

    for _ in range(OBSTACLE_PLACE_ATTEMPTS):
        if horizontal:
            length = np.random.randint(DYNAMIC_ROOM_PARTITION_MIN_LEN, max_len + 1)
            if w - length <= 2 * OBSTACLE_WALL_MARGIN:
                continue
            rx = np.random.randint(x + 1 + OBSTACLE_WALL_MARGIN, x + w - length - OBSTACLE_WALL_MARGIN + 2)
            ry = np.random.randint(y + 1 + OBSTACLE_WALL_MARGIN, y + h - thickness - OBSTACLE_WALL_MARGIN + 2)
            patch = ground_truth[rx:rx + length, ry:ry + thickness]
        else:
            length = np.random.randint(DYNAMIC_ROOM_PARTITION_MIN_LEN, max_len + 1)
            if h - length <= 2 * OBSTACLE_WALL_MARGIN:
                continue
            rx = np.random.randint(x + 1 + OBSTACLE_WALL_MARGIN, x + w - thickness - OBSTACLE_WALL_MARGIN + 2)
            ry = np.random.randint(y + 1 + OBSTACLE_WALL_MARGIN, y + h - length - OBSTACLE_WALL_MARGIN + 2)
            patch = ground_truth[rx:rx + thickness, ry:ry + length]

        if patch.size and np.all(patch == 0):
            patch[:, :] = 1
            return


def _place_dynamic_block_obstacle(ground_truth, room):
    x, y, w, h = room["bbox"]
    ix1 = x + 1 + OBSTACLE_WALL_MARGIN
    iy1 = y + 1 + OBSTACLE_WALL_MARGIN
    ix2 = x + w - OBSTACLE_WALL_MARGIN
    iy2 = y + h - OBSTACLE_WALL_MARGIN
    if ix2 <= ix1 or iy2 <= iy1:
        return

    for _ in range(OBSTACLE_PLACE_ATTEMPTS):
        ow = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
        oh = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
        if ix2 - ow + 1 < ix1 or iy2 - oh + 1 < iy1:
            break

        rx = np.random.randint(ix1, ix2 - ow + 2)
        ry = np.random.randint(iy1, iy2 - oh + 2)
        if np.all(ground_truth[rx:rx + ow, ry:ry + oh] == 0):
            ground_truth[rx:rx + ow, ry:ry + oh] = 1
            return


def _mark_dynamic_line_protected(mask, start, end, width):
    x1, y1 = start
    x2, y2 = end
    half = width // 2
    for x, y in _dynamic_bresenham_line(x1, y1, x2, y2):
        x0 = max(0, x - half)
        x1_lim = min(mask.shape[0], x + half + 1)
        y0 = max(0, y - half)
        y1_lim = min(mask.shape[1], y + half + 1)
        mask[x0:x1_lim, y0:y1_lim] = True


def _dynamic_bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points
