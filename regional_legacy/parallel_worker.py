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


def extract_perfect_labels(env):
    gt = env.ground_truth
    sem = np.zeros_like(gt, dtype=np.int32)
    sem[gt == 0] = 1

    for room in env.rooms:
        x, y, w, h = room['bbox']
        patch_gt = gt[x + 1:x + 1 + w, y + 1:y + 1 + h]
        patch_sem = sem[x + 1:x + 1 + w, y + 1:y + 1 + h]
        patch_sem[patch_gt == 0] = 2
        sem[x + 1:x + 1 + w, y + 1:y + 1 + h] = patch_sem
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
