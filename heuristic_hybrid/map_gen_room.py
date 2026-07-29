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
    ROOM_OBSERVABLE_AREA_RATIO,
    ROOM_GENERATION_ATTEMPTS,
    ROOM_AREA_ADJUST_ATTEMPTS,
    rooms_for,
)


def build(map_size):
    """生成房间-走廊地图，返回 ground_truth 与 rooms 元数据。"""
    target_observable_area = _target_observable_area(map_size)
    best_candidate = None
    best_diff = None

    for _ in range(ROOM_GENERATION_ATTEMPTS):
        ground_truth = np.ones((map_size, map_size), dtype=np.int8)

        rooms = _generate_rooms(ground_truth, map_size)
        for room in rooms:
            _carve_room(ground_truth, room)

        room_graph = _connect_rooms(ground_truth, map_size, rooms)
        for room in rooms:
            _place_room_obstacles(ground_truth, room)

        _adjust_observable_area_to_target(ground_truth, rooms, target_observable_area)
        observable_area = _observable_area(ground_truth)
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

    return ground_truth, {
        "rooms": rooms,
        "room_graph": room_graph,
        "target_observable_area": target_observable_area,
        "observable_area": observable_area,
        "free_area": free_area,
    }


def _target_observable_area(map_size):
    return int(round(map_size * map_size * ROOM_OBSERVABLE_AREA_RATIO))


def _observable_area(ground_truth):
    free = ground_truth == 0
    observable = free.copy()
    observable[1:, :] |= free[:-1, :]
    observable[:-1, :] |= free[1:, :]
    observable[:, 1:] |= free[:, :-1]
    observable[:, :-1] |= free[:, 1:]
    return int(np.sum(observable))


def _adjust_observable_area_to_target(ground_truth, rooms, target_observable_area):
    changed = 0
    for _ in range(ROOM_AREA_ADJUST_ATTEMPTS):
        observable_area = _observable_area(ground_truth)
        diff = target_observable_area - observable_area
        if diff == 0:
            break
        if diff > 0:
            changed += _add_free_area(ground_truth, rooms, min(diff, max(1, abs(diff) // 2)))
        else:
            changed += _remove_free_area(ground_truth, rooms, min(-diff, max(1, abs(diff) // 2)))
    return changed


def _add_free_area(ground_truth, rooms, amount):
    candidates = _wall_candidates_adjacent_to_free(ground_truth)
    if not candidates:
        return 0
    np.random.shuffle(candidates)

    changed = 0
    for x, y in candidates:
        if changed >= amount:
            break
        ground_truth[x, y] = 0
        changed += 1
    return changed


def _remove_free_area(ground_truth, rooms, amount):
    protected = _protected_free_mask(ground_truth, rooms)
    xs, ys = np.where((ground_truth == 0) & (~protected))
    candidates = list(zip(xs.tolist(), ys.tolist()))
    np.random.shuffle(candidates)

    changed = 0
    for x, y in candidates:
        if changed >= amount:
            break
        ground_truth[x, y] = 1
        changed += 1
    return changed


def _wall_candidates_adjacent_to_free(ground_truth):
    free = ground_truth == 0
    adjacent = np.zeros_like(free, dtype=bool)
    adjacent[1:, :] |= free[:-1, :]
    adjacent[:-1, :] |= free[1:, :]
    adjacent[:, 1:] |= free[:, :-1]
    adjacent[:, :-1] |= free[:, 1:]
    adjacent[[0, -1], :] = False
    adjacent[:, [0, -1]] = False
    xs, ys = np.where((ground_truth == 1) & adjacent)
    return list(zip(xs.tolist(), ys.tolist()))


def _protected_free_mask(ground_truth, rooms):
    protected = np.zeros_like(ground_truth, dtype=bool)
    for room in rooms:
        cx, cy = _room_center(room)
        protected[max(0, cx - 2):cx + 3, max(0, cy - 2):cy + 3] = True
        for nx in room.get("neighbors", set()):
            if nx >= len(rooms):
                continue
            ncx, ncy = _room_center(rooms[nx])
            _mark_line_protected(protected, (cx, cy), (ncx, ncy), CORRIDOR_WIDTH)
    return protected & (ground_truth == 0)


def _mark_line_protected(mask, start, end, width):
    x1, y1 = start
    x2, y2 = end
    half = width // 2
    for x, y in _bresenham_line(x1, y1, x2, y2):
        x0 = max(0, x - half)
        x1_lim = min(mask.shape[0], x + half + 1)
        y0 = max(0, y - half)
        y1_lim = min(mask.shape[1], y + half + 1)
        mask[x0:x1_lim, y0:y1_lim] = True


def _bresenham_line(x0, y0, x1, y1):
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


def sample_spawn(ground_truth, meta):
    rooms = meta.get("rooms", [])
    if not rooms:
        raise RuntimeError("房间地图生成失败：未放置任何房间")

    # 优先在双联通/多联通房间出生，避免默认落在只有单一通路的起始房间。
    preferred_rooms = [room for room in rooms if room.get("degree", 0) >= 2]
    candidate_rooms = preferred_rooms if preferred_rooms else rooms

    for _ in range(800):
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
            room = {"bbox": (x, y, w, h), "degree": 0, "neighbors": set()}
            if not _room_overlaps(room, rooms):
                rooms.append(room)
                break
    return rooms


def _room_bbox(room):
    x, y, w, h = room["bbox"]
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
    x, y, w, h = room["bbox"]
    ground_truth[x + 1:x + 1 + w, y + 1:y + 1 + h] = 0


def _room_center(room):
    x, y, w, h = room["bbox"]
    return x + 1 + w // 2, y + 1 + h // 2


def _connect_rooms(ground_truth, map_size, rooms):
    """先保证整体连通，再额外增加回路，让一个走廊可以连接多个房间。"""
    if len(rooms) < 2:
        return []

    centers = [_room_center(room) for room in rooms]
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
        _carve_corridor(ground_truth, map_size, centers[i], centers[j])
        rooms[i]["degree"] += 1
        rooms[j]["degree"] += 1
        rooms[i]["neighbors"].add(j)
        rooms[j]["neighbors"].add(i)
        edges.add(tuple(sorted((i, j))))
        in_tree.add(j)

    # 额外添加一些边，形成双联通/多联通房间和走廊回路。
    extra_edges = max(1, len(rooms) // 3)
    candidates = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if (i, j) not in edges:
                dist = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
                candidates.append((dist, i, j))
    candidates.sort(key=lambda x: x[0])

    added = 0
    for _, i, j in candidates:
        if added >= extra_edges:
            break
        # 限制额外连接的跨度，避免走廊过长破坏房间形态
        if abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1]) > map_size // 2:
            continue
        _carve_corridor(ground_truth, map_size, centers[i], centers[j])
        rooms[i]["degree"] += 1
        rooms[j]["degree"] += 1
        rooms[i]["neighbors"].add(j)
        rooms[j]["neighbors"].add(i)
        edges.add((i, j))
        added += 1

    return list(edges)


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
    x, y, w, h = room["bbox"]
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


