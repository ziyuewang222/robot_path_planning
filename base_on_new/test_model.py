import time

import matplotlib.pyplot as plt
import torch

from config import MAP_GENERATOR, SENSOR_RANGE, SIGMA, TEST_MAP_SIZE, TRAIN_MAP_SIZE
from grid_env import GridWorldEnv
from models import SpatialActionDecision, SpatialFCN
from state_pipeline import StateTensorPipeline


def test_scale_invariance():
    """验证模型对尺寸变更的动态适应能力"""
    print("\n--- 正在验证 FCN 模型的尺寸不变性 ---")
    model = SpatialFCN(in_channels=3)

    dummy_train = torch.randn(1, 3, TRAIN_MAP_SIZE, TRAIN_MAP_SIZE)
    out_train = model(dummy_train)
    print(f"输入尺寸 {TRAIN_MAP_SIZE}x{TRAIN_MAP_SIZE} -> 输出尺寸: {out_train.shape} | 验证通过: {out_train.shape == (1, 1, TRAIN_MAP_SIZE, TRAIN_MAP_SIZE)}")

    dummy_test = torch.randn(1, 3, TEST_MAP_SIZE, TEST_MAP_SIZE)
    out_test = model(dummy_test)
    print(f"输入尺寸 {TEST_MAP_SIZE}x{TEST_MAP_SIZE} -> 输出尺寸: {out_test.shape} | 验证通过: {out_test.shape == (1, 1, TEST_MAP_SIZE, TEST_MAP_SIZE)}")
    print(f"当前三尺度融合权重: {model.get_fusion_weights().tolist()}")


def visualize_decision(explored_map, q_raw, q_masked, chosen_coord, step):
    """画出原始 Q 值与 Mask 截断后 Q 值的对比图"""
    plt.clf()

    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title(f"Step {step}: Selected Target (Red Star)")
    ax1.imshow(explored_map, cmap='viridis', origin='upper')
    if chosen_coord:
        ax1.plot(chosen_coord[1], chosen_coord[0], 'r*', markersize=12, label='Target action')
        ax1.legend()

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Neural Q_raw (Before Masking)")
    im2 = ax2.imshow(q_raw, cmap='plasma', origin='upper')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Q_masked (Legal Frontiers Only)")
    q_vis = q_masked.copy()
    q_vis[q_vis == -1e9] = float('nan')
    im3 = ax3.imshow(q_vis, cmap='plasma', origin='upper')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    if chosen_coord:
        ax3.plot(chosen_coord[1], chosen_coord[0], 'g*', markersize=10)

    plt.suptitle("Action Masking Execution Process", fontsize=14)
    plt.tight_layout()
    plt.pause(0.5)


if __name__ == "__main__":
    test_scale_invariance()

    env = GridWorldEnv(
        map_size=TRAIN_MAP_SIZE,
        sensor_range=SENSOR_RANGE,
        seed=1024,
        map_generator=MAP_GENERATOR,
    )
    pipeline = StateTensorPipeline(map_size=TRAIN_MAP_SIZE, sigma=SIGMA)

    model = SpatialFCN()
    decision_maker = SpatialActionDecision(model, epsilon=0.2)

    explored, pos = env.reset()

    plt.ion()
    plt.figure(figsize=(16, 5))

    for _ in range(10):
        explored, pos, _ = env.step(3)

    print("\n--- 启动对有效探索状态的动作决策流程演示 ---")
    for step in range(1, 11):
        state_tensor = pipeline.process(explored, pos)
        chosen_coord, q_masked, q_raw = decision_maker.select_action(state_tensor, explore=True)

        print(f"[Step {step}] 选中高层导航全局终点: {chosen_coord}")
        visualize_decision(explored, q_raw, q_masked, chosen_coord, step)

        explored, pos, _ = env.step(step % 4)

    print("测试完毕！请检查右侧图 (Q_masked) 是否仅仅只在边界交界处保留了数值，其余黑色部分被彻底屏蔽。窗口将保留 5 秒...")
    time.sleep(5)
