import time
import collections
import numpy as np
import matplotlib.pyplot as plt

# 直接复用你现有的底层物理环境
from config import TEST_MAP_SIZE, TEST_SEED, SENSOR_RANGE, max_macro_steps_for
from grid_env import GridWorldEnv

class BFSFrontierPlanner:
    def __init__(self):
        """标准广度优先前线寻路器 (四向运动)"""
        self.motions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def find_nearest_frontier(self, explored_map, robot_pos):
        """
        在认知地图上使用 BFS 寻找真实路径距离最近的合法边界点
        """
        rx, ry = robot_pos[0], robot_pos[1]
        H, W = explored_map.shape

        visited = np.zeros((H, W), dtype=bool)
        queue = collections.deque()
        
        queue.append((rx, ry))
        visited[rx, ry] = True

        while queue:
            curr_x, curr_y = queue.popleft()

            # 判定合法前线：当前格本身不是起点，且属于已知空地(0)，且周围有迷雾(-1)
            if (curr_x != rx or curr_y != ry) and self._is_frontier(explored_map, curr_x, curr_y, H, W):
                return [curr_x, curr_y]

            # 沿已知空地向四个方向波纹式扩散
            for dx, dy in self.motions:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    # BFS 只能在已知空地 (0) 上搜索穿行；绝对不能穿墙(1)或走入未知的迷雾(-1)
                    if not visited[nx, ny] and explored_map[nx, ny] == 0:
                        visited[nx, ny] = True
                        queue.append((nx, ny))

        return None # 地图开荒完毕或被完全困死

    def _is_frontier(self, explored_map, x, y, H, W):
        if explored_map[x, y] != 0:
            return False
        for dx, dy in self.motions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if explored_map[nx, ny] == -1:
                    return True
        return False


def run_bfs_evaluation():
    # 极度关键：使用和 CNN 评估脚本一模一样的地图种子 (seed=888)，确保房间布局完全一致！
    map_size = TEST_MAP_SIZE
    env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, seed=TEST_SEED)
    planner = BFSFrontierPlanner()
    
    explored, pos = env.reset()
    
    print("=== 启动纯贪心 BFS 探索对比演练演示 ===")
    total_steps = 0
    total_area = 0.0
    
    for step in range(1, max_macro_steps_for(map_size) * 2 + 1):
        # 1. 调用 BFS 算子寻路最近的边界
        chosen_coord = planner.find_nearest_frontier(explored, pos)
        
        if chosen_coord is None:
            print("未发现有效探索边界，地图探索完成！")
            break
            
        # 2. 实时渲染可视化图表 (和之前一致)
        env.render(step_num=step)
        
        # 3. 执行相同的宏观闭环动作
        next_explored, next_pos, reward, n_steps, done, info = env.step_macro(chosen_coord)
        
        total_steps += n_steps
        total_area += info['delta_area']
        
        # 4. 打印和 CNN 脚本完全一致的日志，方便你直接截图对齐比较
        print(f"执行宏观决策 {step:02d} | 奔赴目标: {chosen_coord} | 物理消耗步数: {n_steps:2d} | 获得面积收益: {info['delta_area']:4.0f}")
        
        explored = next_explored
        pos = next_pos
        
        if done:
            print("\n🎉 成功达成 95% 以上区域开荒！探索完结！")
            break
            
    print(f"评估总结 (BFS 贪心 baseline) -> 累计物理步数: {total_steps} | 累计探明新区域: {total_area:.0f}")
    print("窗口将保留 10 秒供观察...")
    time.sleep(10)

if __name__ == "__main__":
    run_bfs_evaluation()