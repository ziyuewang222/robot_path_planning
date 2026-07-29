import numpy as np

from config import (
    EASY_BIG_ROOM_CORRIDOR_WIDTH,
    OBS_MIN_SIZE,
    OBS_MAX_SIZE,
    OBSTACLE_PLACE_ATTEMPTS,
    OBSTACLE_WALL_MARGIN,
    ROOM_PADDING,
    SPAWN_MARGIN,
)

EASY_BIG_ROOM_COUNT = 3
EASY_BIG_ROOM_MIN_SIZE = 50
EASY_BIG_ROOM_MAX_SIZE = 70
EASY_BIG_ROOM_ATTEMPTS = 160
EASY_BIG_ROOM_TARGET_OBSTACLE_DENSITY = 0.0028
EASY_BIG_ROOM_MAX_EXTRA_CONNECT_DIST_RATIO = 0.72


def build(map_size):
    """生成 3 个大房间 + 简单走廊 + 少量家具式障碍物的地图。"""
    best_candidate = None
    best_score = None

    for _ in range(EASY_BIG_ROOM_ATTEMPTS):
        ground_truth = np.ones((map_size, map_size), dtype=np.int8)
        rooms = _generate_rooms(map_size)
        if len(rooms) < EASY_BIG_ROOM_COUNT:
            continue

        for room in rooms:
            _carve_room(ground_truth, room)

        room_graph = _connect_rooms(ground_truth, map_size, rooms)
        for room in rooms:
            _place_room_obstacles(ground_truth, room)

        free_area = int(np.sum(ground_truth == 0))
        corridor_penalty = _corridor_complexity_score(room_graph)
        size_penalty = _room_size_variance_score(rooms)
        obstacle_density = _obstacle_density_score(ground_truth, rooms)
        score = corridor_penalty + size_penalty + obstacle_density

        candidate = (ground_truth, rooms, room_graph, free_area)
        if best_score is None or score < best_score:
            best_candidate = candidate
            best_score = score

        if corridor_penalty == 0:
            break

    ground_truth, rooms, room_graph, free_area = best_candidate
    for room in rooms:
        room.setdefault("degree", 0)
        room.setdefault("neighbors", set())

    return ground_truth, {
        "rooms": rooms,
        "room_graph": room_graph,
        "free_area": free_area,
        "room_count": len(rooms),
        "room_size_range": (EASY_BIG_ROOM_MIN_SIZE, EASY_BIG_ROOM_MAX_SIZE),
        "target_obstacle_density": EASY_BIG_ROOM_TARGET_OBSTACLE_DENSITY,
        "corridor_width": EASY_BIG_ROOM_CORRIDOR_WIDTH,
    }


def sample_spawn(ground_truth, meta):
    rooms = meta.get("rooms", [])
    if not rooms:
        raise RuntimeError("easy_big_room 地图生成失败：未放置任何房间")

    candidate_rooms = [room for room in rooms if room.get("degree", 0) >= 1] or rooms
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

    x, y, w, h = candidate_rooms[0]["bbox"]
    return x + 1 + w // 2, y + 1 + h // 2


def _generate_rooms(map_size):
    rooms = []
    for _ in range(EASY_BIG_ROOM_COUNT):
        placed = False
        for _ in range(ROOM_PADDING * 20):
            w = np.random.randint(EASY_BIG_ROOM_MIN_SIZE, EASY_BIG_ROOM_MAX_SIZE + 1)
            h = np.random.randint(EASY_BIG_ROOM_MIN_SIZE, EASY_BIG_ROOM_MAX_SIZE + 1)
            x_max = map_size - w - 3
            y_max = map_size - h - 3
            if x_max < 1 or y_max < 1:
                break
            x = np.random.randint(1, x_max + 1)
            y = np.random.randint(1, y_max + 1)
            room = {"bbox": (x, y, w, h), "degree": 0, "neighbors": set()}
            if not _room_overlaps(room, rooms):
                rooms.append(room)
                placed = True
                break
        if not placed:
            return rooms
    return rooms


def _room_bbox(room):
    x, y, w, h = room["bbox"]
    return x, y, x + w + 2, y + h + 2


def _room_overlaps(room, rooms):
    ax1, ay1, ax2, ay2 = _room_bbox(room)
    for other in rooms:
        bx1, by1, bx2, by2 = _room_bbox(other)
        if not (ax2 + ROOM_PADDING <= bx1 or bx2 + ROOM_PADDING <= ax1 or ay2 + ROOM_PADDING <= by1 or by2 + ROOM_PADDING <= ay1):
            return True
    return False


def _carve_room(ground_truth, room):
    x, y, w, h = room["bbox"]
    ground_truth[x + 1:x + 1 + w, y + 1:y + 1 + h] = 0


def _room_center(room):
    x, y, w, h = room["bbox"]
    return x + 1 + w // 2, y + 1 + h // 2


def _connect_rooms(ground_truth, map_size, rooms):
    if len(rooms) < 2:
        return []

    centers = [_room_center(room) for room in rooms]
    edges = set()

    # 简化拓扑：先按左右顺序串联，再少量补边，避免 dynamic_room 那种较复杂连接
    order = sorted(range(len(rooms)), key=lambda idx: centers[idx][1])
    for i in range(len(order) - 1):
        a = order[i]
        b = order[i + 1]
        _carve_corridor(ground_truth, map_size, centers[a], centers[b])
        _add_edge(rooms, edges, a, b)

    return list(edges)


def _add_edge(rooms, edges, i, j):
    rooms[i]["degree"] += 1
    rooms[j]["degree"] += 1
    rooms[i]["neighbors"].add(j)
    rooms[j]["neighbors"].add(i)
    edges.add(tuple(sorted((i, j))))


def _carve_corridor(ground_truth, map_size, start, end):
    x1, y1 = start
    x2, y2 = end
    half = EASY_BIG_ROOM_CORRIDOR_WIDTH // 2
    if np.random.rand() < 0.5:
        _carve_horizontal_corridor(ground_truth, map_size, x1, x2, y1, half)
        _carve_vertical_corridor(ground_truth, map_size, y1, y2, x2, half)
    else:
        _carve_vertical_corridor(ground_truth, map_size, y1, y2, x1, half)
        _carve_horizontal_corridor(ground_truth, map_size, x1, x2, y2, half)


def _carve_horizontal_corridor(ground_truth, map_size, x1, x2, y, half):
    xa, xb = sorted((x1, x2))
    y0 = max(1, y - half)
    y1 = min(map_size - 1, y + half + 1)
    ground_truth[xa:xb + 1, y0:y1] = 0


def _carve_vertical_corridor(ground_truth, map_size, y1, y2, x, half):
    ya, yb = sorted((y1, y2))
    x0 = max(1, x - half)
    x1 = min(map_size - 1, x + half + 1)
    ground_truth[x0:x1, ya:yb + 1] = 0


def _place_room_obstacles(ground_truth, room):
    x, y, w, h = room["bbox"]
    area = w * h
    n_obs = max(3, int(round(area * EASY_BIG_ROOM_TARGET_OBSTACLE_DENSITY)))
    ix1 = x + 1 + OBSTACLE_WALL_MARGIN
    iy1 = y + 1 + OBSTACLE_WALL_MARGIN
    ix2 = x + w - OBSTACLE_WALL_MARGIN
    iy2 = y + h - OBSTACLE_WALL_MARGIN
    if ix2 <= ix1 or iy2 <= iy1:
        return

    placed = 0
    for _ in range(n_obs):
        if _place_table_like_obstacle(ground_truth, ix1, iy1, ix2, iy2):
            placed += 1
            continue
        if _place_small_block(ground_truth, ix1, iy1, ix2, iy2):
            placed += 1


def _place_table_like_obstacle(ground_truth, ix1, iy1, ix2, iy2):
    for _ in range(OBSTACLE_PLACE_ATTEMPTS):
        if np.random.rand() < 0.5:
            ow = np.random.randint(max(2, OBS_MIN_SIZE), min(8, OBS_MAX_SIZE + 4) + 1)
            oh = np.random.randint(2, 4)
        else:
            ow = np.random.randint(2, 4)
            oh = np.random.randint(max(2, OBS_MIN_SIZE), min(8, OBS_MAX_SIZE + 4) + 1)

        if ix2 - ow + 1 < ix1 or iy2 - oh + 1 < iy1:
            continue

        rx = np.random.randint(ix1, ix2 - ow + 2)
        ry = np.random.randint(iy1, iy2 - oh + 2)
        if np.all(ground_truth[rx:rx + ow, ry:ry + oh] == 0):
            ground_truth[rx:rx + ow, ry:ry + oh] = 1
            return True
    return False


def _place_small_block(ground_truth, ix1, iy1, ix2, iy2):
    for _ in range(OBSTACLE_PLACE_ATTEMPTS):
        ow = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
        oh = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
        if ix2 - ow + 1 < ix1 or iy2 - oh + 1 < iy1:
            continue
        rx = np.random.randint(ix1, ix2 - ow + 2)
        ry = np.random.randint(iy1, iy2 - oh + 2)
        if np.all(ground_truth[rx:rx + ow, ry:ry + oh] == 0):
            ground_truth[rx:rx + ow, ry:ry + oh] = 1
            return True
    return False


def _corridor_complexity_score(room_graph):
    degrees = []
    for a, b in room_graph:
        degrees.append((a, b))
    return max(0, len(room_graph) - 2)


def _room_size_variance_score(rooms):
    sizes = [room["bbox"][2] * room["bbox"][3] for room in rooms]
    if not sizes:
        return 0.0
    return float(np.std(sizes) / max(1.0, np.mean(sizes)))


def _obstacle_density_score(ground_truth, _rooms):
    free = np.sum(ground_truth == 0)
    obstacle = np.sum(ground_truth == 1)
    total = free + obstacle
    if total <= 0:
        return 1.0
    density = obstacle / total
    return abs(density - EASY_BIG_ROOM_TARGET_OBSTACLE_DENSITY) * 10.0
