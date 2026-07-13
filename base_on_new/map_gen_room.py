import numpy as np

from config import CORRIDOR_WIDTH, SPAWN_MARGIN


ROOM_COUNT_MIN = 3
ROOM_COUNT_MAX = 4
ROOM_SIZE_AT_200_MIN = 30
ROOM_SIZE_AT_200_MAX = 70
ROOM_SIZE_REFERENCE_MAP = 200
ROOM_PADDING = 4
ROOM_PLACE_ATTEMPTS = 300


def build(map_size):
    """生成 3-4 个矩形房间，并用无分叉走廊连接。"""
    ground_truth = np.ones((map_size, map_size), dtype=np.int8)

    rooms = _generate_rooms(map_size)
    for room in rooms:
        _carve_room(ground_truth, room)

    room_graph = _connect_rooms(ground_truth, rooms, map_size)

    for room in rooms:
        room.setdefault("degree", 0)

    free_area = int(np.sum(ground_truth == 0))
    return ground_truth, {
        "rooms": rooms,
        "room_graph": room_graph,
        "free_area": free_area,
    }


def sample_spawn(ground_truth, meta):
    rooms = meta.get("rooms", [])
    if not rooms:
        raise RuntimeError("房间地图生成失败：未放置任何房间")

    for _ in range(800):
        room = rooms[np.random.randint(len(rooms))]
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

    x, y, w, h = rooms[0]["bbox"]
    return x + 1 + w // 2, y + 1 + h // 2


def _generate_rooms(map_size):
    count = np.random.randint(ROOM_COUNT_MIN, ROOM_COUNT_MAX + 1)
    rooms = []
    min_size, max_size = _room_size_range(map_size)

    for _ in range(count):
        placed = False
        for _ in range(ROOM_PLACE_ATTEMPTS):
            w = np.random.randint(min_size, max_size + 1)
            h = np.random.randint(min_size, max_size + 1)
            if w + 4 >= map_size or h + 4 >= map_size:
                continue

            x = np.random.randint(2, map_size - w - 2)
            y = np.random.randint(2, map_size - h - 2)
            room = {"bbox": (x, y, w, h), "degree": 0, "neighbors": set()}
            if not _overlaps_any(room, rooms):
                rooms.append(room)
                placed = True
                break
        if not placed:
            break

    if len(rooms) < ROOM_COUNT_MIN:
        raise RuntimeError("房间地图生成失败：房间数量不足")

    return rooms


def _room_size_range(map_size):
    scale = map_size / ROOM_SIZE_REFERENCE_MAP
    min_size = max(8, int(round(ROOM_SIZE_AT_200_MIN * scale)))
    max_size = max(min_size + 2, int(round(ROOM_SIZE_AT_200_MAX * scale)))
    max_size = min(max_size, map_size - 6)
    min_size = min(min_size, max_size - 1)
    return min_size, max_size


def _room_outer_bbox(room):
    x, y, w, h = room["bbox"]
    return x - 1, y - 1, x + w + 1, y + h + 1


def _overlaps_any(room, rooms):
    ax1, ay1, ax2, ay2 = _room_outer_bbox(room)
    for other in rooms:
        bx1, by1, bx2, by2 = _room_outer_bbox(other)
        if not (ax2 + ROOM_PADDING <= bx1 or bx2 + ROOM_PADDING <= ax1 or ay2 + ROOM_PADDING <= by1 or by2 + ROOM_PADDING <= ay1):
            return True
    return False


def _carve_room(ground_truth, room):
    x, y, w, h = room["bbox"]
    ground_truth[x:x + w, y:y + h] = 0


def _room_center(room):
    x, y, w, h = room["bbox"]
    return x + w // 2, y + h // 2


def _connect_rooms(ground_truth, rooms, map_size):
    centers = [_room_center(room) for room in rooms]
    edges = []

    order = list(range(len(rooms)))
    np.random.shuffle(order)

    # 用随机生成的树结构连接所有房间，保证任意两房间之间只有唯一通路。
    for idx in range(1, len(order)):
        i = order[idx]
        j = order[np.random.randint(0, idx)]
        _carve_corridor(ground_truth, centers[i], centers[j], map_size)
        rooms[i]["degree"] += 1
        rooms[j]["degree"] += 1
        rooms[i]["neighbors"].add(j)
        rooms[j]["neighbors"].add(i)
        edges.append((i, j))

    return edges


def _carve_corridor(ground_truth, start, end, map_size):
    x1, y1 = start
    x2, y2 = end

    # 走廊允许弯曲，但不允许分叉：始终只走单条折线路径。
    # 通过拉长中间“拐点”位置，让房间之间的往返步数更高。
    if np.random.rand() < 0.55:
        bend_y = _sample_bend_coord(y1, y2, map_size, prefer_longer=True)
        _carve_vertical_corridor(ground_truth, y1, bend_y, x1, map_size)
        _carve_horizontal_corridor(ground_truth, x1, x2, bend_y, map_size)
        _carve_vertical_corridor(ground_truth, bend_y, y2, x2, map_size)
    else:
        bend_x = _sample_bend_coord(x1, x2, map_size, prefer_longer=True)
        _carve_horizontal_corridor(ground_truth, x1, bend_x, y1, map_size)
        _carve_vertical_corridor(ground_truth, y1, y2, bend_x, map_size)
        _carve_horizontal_corridor(ground_truth, bend_x, x2, y2, map_size)


def _carve_horizontal_corridor(ground_truth, x1, x2, y, map_size):
    xa, xb = sorted((x1, x2))
    half = CORRIDOR_WIDTH // 2
    y0 = max(0, y - half)
    y1 = min(map_size, y0 + CORRIDOR_WIDTH)
    ground_truth[xa:xb + 1, y0:y1] = 0


def _carve_vertical_corridor(ground_truth, y1, y2, x, map_size):
    ya, yb = sorted((y1, y2))
    half = CORRIDOR_WIDTH // 2
    x0 = max(0, x - half)
    x1 = min(map_size, x0 + CORRIDOR_WIDTH)
    ground_truth[x0:x1, ya:yb + 1] = 0


def _sample_bend_coord(a, b, _map_size, prefer_longer=True):
    lo, hi = sorted((a, b))
    if hi - lo <= 2:
        return lo

    span = hi - lo
    margin = max(2, span // 5)
    if prefer_longer:
        if np.random.rand() < 0.5:
            left = np.random.randint(lo, min(lo + margin + 1, hi))
            return left
        right = np.random.randint(max(lo, hi - margin), hi + 1)
        return right

    return np.random.randint(lo, hi + 1)
