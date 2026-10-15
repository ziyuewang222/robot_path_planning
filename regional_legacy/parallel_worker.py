import collections
import numpy as np

from env.grid_env import GridWorldEnv
from config import MAP_GENERATOR, SENSOR_RANGE


class LightweightBFSPlanner:
    def __init__(self):
        self.motions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def find_nearest_frontier(self, explored_map, robot_pos):
        rx, ry = robot_pos[0], robot_pos[1]
        H, W = explored_map.shape
        visited = np.zeros((H, W), dtype=bool)
        queue = collections.deque([(rx, ry)])
        visited[rx, ry] = True

        while queue:
            curr_x, curr_y = queue.popleft()
            if (curr_x != rx or curr_y != ry) and self._is_frontier(explored_map, curr_x, curr_y, H, W):
                return [curr_x, curr_y]
            for dx, dy in self.motions:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if not visited[nx, ny] and explored_map[nx, ny] == 0:
                        visited[nx, ny] = True
                        queue.append((nx, ny))
        return None

    def _is_frontier(self, explored_map, x, y, H, W):
        if explored_map[x, y] != 0:
            return False
        for dx, dy in self.motions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and explored_map[nx, ny] == -1:
                return True
        return False


def extract_perfect_labels(env_or_ground_truth, map_meta=None, generator_name=None):
    if hasattr(env_or_ground_truth, "ground_truth"):
        env = env_or_ground_truth
        ground_truth = env.ground_truth
        map_meta = getattr(env, "map_meta", None) if map_meta is None else map_meta
        generator_name = getattr(env, "map_generator", MAP_GENERATOR) if generator_name is None else generator_name
    else:
        ground_truth = np.asarray(env_or_ground_truth)
        map_meta = {} if map_meta is None else map_meta
        generator_name = MAP_GENERATOR if generator_name is None else generator_name

    if generator_name == "easy_big_room":
        return extract_easy_big_room_labels(ground_truth, map_meta)
    if generator_name == "fish_bone":
        return extract_fish_bone_labels(ground_truth, map_meta)
    raise ValueError(f"不支持的地图生成器: {generator_name}")


def extract_easy_big_room_labels(ground_truth, map_meta):
    sem = np.zeros_like(ground_truth, dtype=np.int32)
    room_mask = np.zeros_like(ground_truth, dtype=bool)
    for room in map_meta.get("rooms", []):
        x, y, w, h = room["bbox"]
        x1 = x + 1
        y1 = y + 1
        x2 = x1 + w
        y2 = y1 + h
        room_mask[x1:x2, y1:y2] = True
    sem[(ground_truth == 0) & room_mask] = 2
    sem[(ground_truth == 0) & ~room_mask] = 1
    return sem


def extract_fish_bone_labels(ground_truth, map_meta):
    sem = np.zeros_like(ground_truth, dtype=np.int32)
    sem[ground_truth == 1] = 0

    corridor_mask = np.asarray(map_meta.get("corridor_mask")) if map_meta.get("corridor_mask") is not None else None
    if corridor_mask is not None:
        sem[(ground_truth == 0) & corridor_mask] = 1
        sem[(ground_truth == 0) & ~corridor_mask] = 2
        return sem

    rooms = map_meta.get("rooms", [])
    room_mask = np.zeros_like(ground_truth, dtype=bool)
    for room in rooms:
        x, y, w, h = room["bbox"]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        room_mask[x1:x2, y1:y2] = True
    sem[(ground_truth == 0) & room_mask] = 2
    sem[(ground_truth == 0) & ~room_mask] = 1
    return sem


def data_collection_worker(worker_id, data_queue, max_episodes, map_size):
    """子进程工作循环：生成地图、跑 BFS、定期发送训练样本和进度消息。"""
    np.random.seed(np.random.randint(0, 100000) + worker_id)

    env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, map_generator=MAP_GENERATOR)
    planner = LightweightBFSPlanner()

    maps_collected = 0
    episodes_finished = 0

    for _ in range(max_episodes):
        explored, pos = env.reset()
        perfect_labels = extract_perfect_labels(env)

        done = False
        step_count = 0
        episode_maps_collected = 0

        while not done:
            target = planner.find_nearest_frontier(explored, pos)
            if target is None:
                break

            explored, pos, _, _, done, _info = env.step_macro(target)
            step_count += 1

            # 每走 3 步截取一次数据
            if step_count % 3 == 0:
                try:
                    data_queue.put((explored.copy(), perfect_labels.copy()), timeout=10)
                    maps_collected += 1
                    episode_maps_collected += 1
                except Exception:
                    pass

        episodes_finished += 1
        try:
            data_queue.put(
                {
                    "type": "progress",
                    "worker_id": worker_id,
                    "episodes_finished": episodes_finished,
                    "maps_collected": maps_collected,
                    "episode_maps_collected": episode_maps_collected,
                    "worker_finished": False,
                },
                timeout=10,
            )
        except Exception:
            pass

    try:
        data_queue.put(
            {
                "type": "progress",
                "worker_id": worker_id,
                "episodes_finished": episodes_finished,
                "maps_collected": maps_collected,
                "worker_finished": True,
            },
            timeout=10,
        )
    except Exception:
        pass

    data_queue.put(None)
