import numpy as np

from config import (
    FISHBONE_CORRIDOR_WIDTH,
    FISHBONE_DOOR_MAX,
    FISHBONE_DOOR_MIN,
    FISHBONE_MAX_GENERATION_ATTEMPTS,
    FISHBONE_MIN_ROOM_WIDTH,
    FISHBONE_OBSTACLE_DENSITY,
    FISHBONE_OBSTACLE_MARGIN,
    FISHBONE_ROOM_COUNT_MIN,
    FISHBONE_ROOM_MARGIN,
    OBS_MAX_SIZE,
    OBS_MIN_SIZE,
    OBSTACLE_PLACE_ATTEMPTS,
    SPAWN_MARGIN,
)


def build(map_size):
    """生成带尺寸自适应 BSP 走廊树、长宽比约束及多向开门的鱼骨地图。"""
    if map_size < 2 * FISHBONE_ROOM_MARGIN + 2 * FISHBONE_MIN_ROOM_WIDTH:
        return _fallback_map(map_size)

    for _ in range(FISHBONE_MAX_GENERATION_ATTEMPTS):
        ground_truth = np.ones((map_size, map_size), dtype=np.int8)
        rooms = []
        room_graph = []
        blocks = []
        margin = FISHBONE_ROOM_MARGIN
        corridor = FISHBONE_CORRIDOR_WIDTH

        def bsp_split(x1, y1, x2, y2, depth, max_depth):
            w = x2 - x1
            h = y2 - y1
            if w < 2 * FISHBONE_MIN_ROOM_WIDTH or h < 2 * FISHBONE_MIN_ROOM_WIDTH:
                blocks.append((x1, y1, x2, y2))
                return

            force_x = w > h * 2.5
            force_y = h > w * 2.5
            can_split_x = w >= 2 * FISHBONE_MIN_ROOM_WIDTH + corridor + 4
            can_split_y = h >= 2 * FISHBONE_MIN_ROOM_WIDTH + corridor + 4
            should_split = depth < max_depth or force_x or force_y

            if should_split and (can_split_x or can_split_y):
                if force_x and can_split_x:
                    axis = "x"
                elif force_y and can_split_y:
                    axis = "y"
                else:
                    axis = "x" if ((w >= h and can_split_x) or not can_split_y) else "y"

                if axis == "x":
                    split_min = x1 + max(FISHBONE_MIN_ROOM_WIDTH, w // 3)
                    split_max = x2 - max(FISHBONE_MIN_ROOM_WIDTH, w // 3)
                    if split_min >= split_max:
                        blocks.append((x1, y1, x2, y2))
                        return
                    split_x = np.random.randint(split_min, split_max)
                    left_end = max(x1 + 1, split_x - corridor // 2 - 1)
                    right_start = min(x2 - 1, split_x + corridor // 2 + 1)
                    ground_truth[max(x1, split_x - corridor // 2):min(x2, split_x + corridor // 2), max(0, y1 - 1):min(map_size, y2 + 1)] = 0
                    bsp_split(x1, y1, left_end, y2, depth + 1, max_depth)
                    bsp_split(right_start, y1, x2, y2, depth + 1, max_depth)
                else:
                    split_min = y1 + max(FISHBONE_MIN_ROOM_WIDTH, h // 3)
                    split_max = y2 - max(FISHBONE_MIN_ROOM_WIDTH, h // 3)
                    if split_min >= split_max:
                        blocks.append((x1, y1, x2, y2))
                        return
                    split_y = np.random.randint(split_min, split_max)
                    top_end = max(y1 + 1, split_y - corridor // 2 - 1)
                    bottom_start = min(y2 - 1, split_y + corridor // 2 + 1)
                    ground_truth[max(0, x1 - 1):min(map_size, x2 + 1), max(y1, split_y - corridor // 2):min(y2, split_y + corridor // 2)] = 0
                    bsp_split(x1, y1, x2, top_end, depth + 1, max_depth)
                    bsp_split(x1, bottom_start, x2, y2, depth + 1, max_depth)
            else:
                blocks.append((x1, y1, x2, y2))

        target_depth = max(1, map_size // 150)
        bsp_split(margin, margin, map_size - margin, map_size - margin, 0, target_depth)
        corridor_mask = (ground_truth == 0).copy()

        for bx1, by1, bx2, by2 in blocks:
            w = bx2 - bx1
            h = by2 - by1
            if w < FISHBONE_MIN_ROOM_WIDTH or h < FISHBONE_MIN_ROOM_WIDTH:
                continue

            corridor_sides = []
            if bx1 > margin and np.any(corridor_mask[bx1 - 2, by1:by2]):
                corridor_sides.append("left")
            if bx2 < map_size - margin and np.any(corridor_mask[bx2 + 1, by1:by2]):
                corridor_sides.append("right")
            if by1 > margin and np.any(corridor_mask[bx1:bx2, by1 - 2]):
                corridor_sides.append("top")
            if by2 < map_size - margin and np.any(corridor_mask[bx1:bx2, by2 + 1]):
                corridor_sides.append("bottom")

            if not corridor_sides:
                continue

            if w >= h:
                preferred = [side for side in corridor_sides if side in ("top", "bottom")]
                side = np.random.choice(preferred) if preferred else np.random.choice(corridor_sides)
            else:
                preferred = [side for side in corridor_sides if side in ("left", "right")]
                side = np.random.choice(preferred) if preferred else np.random.choice(corridor_sides)

            if side in ("top", "bottom"):
                corridor_depth = h
                span = w
                min_span = max(FISHBONE_MIN_ROOM_WIDTH, int(np.ceil(corridor_depth / 3)))
                max_span = max(min_span, min(span, int(corridor_depth * 3)))
                min_rooms = max(1, int(np.ceil(span / max_span)))
                max_rooms = max(min_rooms, int(np.floor(span / min_span)))
                n_rooms = np.random.randint(min_rooms, max_rooms + 1)
                splits = _random_split_space(span, n_rooms, min_span)
                for i in range(n_rooms):
                    rx1 = bx1 + splits[i]
                    rx2 = bx1 + splits[i + 1] - 1
                    if rx2 <= rx1 + 3:
                        continue
                    shrink = np.random.randint(0, max(1, corridor_depth // 4))
                    if side == "top":
                        ry1 = by1
                        ry2 = max(by1 + 4, by2 - shrink)
                    else:
                        ry1 = min(by2 - 4, by1 + shrink)
                        ry2 = by2
                    if ry2 <= ry1 + 3:
                        continue
                    ground_truth[rx1:rx2, ry1:ry2] = 0
                    doors_count = _carve_all_doors(ground_truth, corridor_mask, rx1, rx2, ry1, ry2, map_size)
                    room = {"id": len(rooms), "bbox": (rx1, ry1, rx2 - rx1, ry2 - ry1), "degree": doors_count, "neighbors": set()}
                    _place_room_obstacles(ground_truth, room)
                    rooms.append(room)
                    room_graph.append((room["id"], room["id"]))
            else:
                corridor_depth = w
                span = h
                min_span = max(FISHBONE_MIN_ROOM_WIDTH, int(np.ceil(corridor_depth / 3)))
                max_span = max(min_span, min(span, int(corridor_depth * 3)))
                min_rooms = max(1, int(np.ceil(span / max_span)))
                max_rooms = max(min_rooms, int(np.floor(span / min_span)))
                n_rooms = np.random.randint(min_rooms, max_rooms + 1)
                splits = _random_split_space(span, n_rooms, min_span)
                for i in range(n_rooms):
                    ry1 = by1 + splits[i]
                    ry2 = by1 + splits[i + 1] - 1
                    if ry2 <= ry1 + 3:
                        continue
                    shrink = np.random.randint(0, max(1, corridor_depth // 4))
                    if side == "left":
                        rx1 = bx1
                        rx2 = max(bx1 + 4, bx2 - shrink)
                    else:
                        rx1 = min(bx2 - 4, bx1 + shrink)
                        rx2 = bx2
                    if rx2 <= rx1 + 3:
                        continue
                    ground_truth[rx1:rx2, ry1:ry2] = 0
                    doors_count = _carve_all_doors(ground_truth, corridor_mask, rx1, rx2, ry1, ry2, map_size)
                    room = {"id": len(rooms), "bbox": (rx1, ry1, rx2 - rx1, ry2 - ry1), "degree": doors_count, "neighbors": set()}
                    _place_room_obstacles(ground_truth, room)
                    rooms.append(room)
                    room_graph.append((room["id"], room["id"]))

        if len(rooms) >= FISHBONE_ROOM_COUNT_MIN:
            return ground_truth, {
                "rooms": rooms,
                "room_graph": room_graph,
                "free_area": int(np.sum(ground_truth == 0)),
                "room_count": len(rooms),
                "corridor_width": corridor,
                "generator": "dynamic_bsp_fish_bone",
            }

    return _fallback_map(map_size)


def _carve_door(ground_truth, axis, boundary, start, end):
    width = np.random.randint(FISHBONE_DOOR_MIN, FISHBONE_DOOR_MAX + 1)
    center = (start + end) // 2
    door_start = max(start, center - width // 2)
    door_end = min(end, door_start + width)
    if axis == "x":
        ground_truth[door_start:door_end, boundary] = 0
    else:
        ground_truth[boundary, door_start:door_end] = 0


def _carve_all_doors(ground_truth, corridor_mask, rx1, rx2, ry1, ry2, map_size):
    """扫描房间四周，凡接触走廊的边都开门。"""
    doors_carved = 0
    if ry1 - 2 >= 0 and np.any(corridor_mask[rx1:rx2, ry1 - 2]):
        _carve_door(ground_truth, "x", ry1 - 1, rx1, rx2)
        doors_carved += 1
    if ry2 + 1 < map_size and np.any(corridor_mask[rx1:rx2, ry2 + 1]):
        _carve_door(ground_truth, "x", ry2, rx1, rx2)
        doors_carved += 1
    if rx1 - 2 >= 0 and np.any(corridor_mask[rx1 - 2, ry1:ry2]):
        _carve_door(ground_truth, "y", rx1 - 1, ry1, ry2)
        doors_carved += 1
    if rx2 + 1 < map_size and np.any(corridor_mask[rx2 + 1, ry1:ry2]):
        _carve_door(ground_truth, "y", rx2, ry1, ry2)
        doors_carved += 1
    return doors_carved




def sample_spawn(ground_truth, meta):
    rooms = meta.get("rooms", [])
    if not rooms:
        raise RuntimeError("fish_bone 地图生成失败：未生成任何房间")

    for _ in range(1000):
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


def _random_split_space(total_length, num_rooms, min_len):
    remaining = total_length - num_rooms * min_len
    if remaining <= 0:
        widths = [min_len] * num_rooms
    else:
        weights = np.random.rand(num_rooms)
        weights /= weights.sum()
        extra = np.floor(weights * remaining).astype(int)
        widths = [min_len + int(v) for v in extra]
        widths[-1] += total_length - sum(widths)
    splits = [0]
    for width in widths:
        splits.append(splits[-1] + width)
    return splits


def _place_room_obstacles(ground_truth, room):
    x, y, w, h = room["bbox"]
    area = w * h
    n_obs = max(3, int(round(area * FISHBONE_OBSTACLE_DENSITY)))
    ix1 = x + FISHBONE_OBSTACLE_MARGIN
    iy1 = y + FISHBONE_OBSTACLE_MARGIN
    ix2 = x + w - FISHBONE_OBSTACLE_MARGIN
    iy2 = y + h - FISHBONE_OBSTACLE_MARGIN
    if ix2 <= ix1 or iy2 <= iy1:
        return
    for _ in range(n_obs):
        if _place_table_like_obstacle(ground_truth, ix1, iy1, ix2, iy2):
            continue
        _place_small_block(ground_truth, ix1, iy1, ix2, iy2)


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


def _fallback_map(map_size):
    ground_truth = np.ones((map_size, map_size), dtype=np.int8)
    x1 = max(2, FISHBONE_ROOM_MARGIN)
    x2 = max(x1 + 2, map_size - FISHBONE_ROOM_MARGIN)
    y1 = max(2, FISHBONE_ROOM_MARGIN)
    y2 = max(y1 + 2, map_size - FISHBONE_ROOM_MARGIN)
    ground_truth[x1:x2, y1:y2] = 0
    room = {"id": 0, "bbox": (x1, y1, x2 - x1, y2 - y1), "degree": 0, "neighbors": set()}
    return ground_truth, {"rooms": [room], "room_graph": [], "free_area": int(np.sum(ground_truth == 0)), "room_count": 1, "corridor_width": FISHBONE_CORRIDOR_WIDTH, "generator": "fish_bone"}
