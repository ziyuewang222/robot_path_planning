import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models import SpatialFCN


class SMDPReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_coord, reward, next_state, next_mask, done, tau_steps):
        self.buffer.append((state, action_coord, reward, next_state, next_mask, done, tau_steps))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class SpatialQTrainer:
    def __init__(self, lr=1e-4, gamma=0.99, buffer_size=2000, device='cpu'):
        self.device = device
        self.gamma = gamma

        self.policy_net = SpatialFCN().to(device)
        self.target_net = SpatialFCN().to(device)
        self.sync_target_net()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.memory = SMDPReplayBuffer(capacity=buffer_size)

    def update_step(self, batch_size=16):
        if len(self.memory) < batch_size:
            return 0.0

        transitions = self.memory.sample(batch_size)
        states, action_coords, rewards, next_states, next_masks, dones, taus = zip(*transitions)

        states = torch.stack(states, dim=0).to(self.device, non_blocking=True)
        next_states = torch.stack(next_states, dim=0).to(self.device, non_blocking=True)
        next_masks = torch.stack(next_masks, dim=0).to(self.device, non_blocking=True)
        action_coords = torch.tensor(action_coords, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        taus = torch.tensor(taus, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()

        q_raw = self.policy_net(states).squeeze(1)
        batch_indices = torch.arange(batch_size, device=self.device)
        pred_q = q_raw[batch_indices, action_coords[:, 0], action_coords[:, 1]]

        with torch.no_grad():
            q_next_raw = self.target_net(next_states).squeeze(1)
            q_next_masked = q_next_raw.masked_fill(next_masks < 0.5, -1e9)
            max_next_q = q_next_masked.flatten(start_dim=1).max(dim=1).values
            gamma_tau = self.gamma ** taus
            target_q = rewards + (~dones).float() * gamma_tau * max_next_q

        loss = self.criterion(pred_q, target_q)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def soft_update_target(self, tau=0.01):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def sync_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def load_checkpoint(self, checkpoint_path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"初始主策略权重不存在: {path}")
        state_dict = torch.load(path, map_location=self.device)
        current_state = self.policy_net.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        current_state.update(compatible_state)
        self.policy_net.load_state_dict(current_state)
        self.sync_target_net()

    def load_fusion_weights(self, fusion_weight_path) -> None:
        path = Path(fusion_weight_path)
        if not path.exists():
            raise FileNotFoundError(f"初始融合权重不存在: {path}")
        self.policy_net.load_fusion_weights(str(path))
        self.sync_target_net()

    def load_initial_policy(self, checkpoint_path=None, fusion_weight_path=None) -> str:
        loaded_parts = []
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            loaded_parts.append("checkpoint")
        if fusion_weight_path:
            self.load_fusion_weights(fusion_weight_path)
            loaded_parts.append("fusion_weights")
        if not loaded_parts:
            return "scratch"
        return "+".join(loaded_parts)

    def save_all_weights(self, checkpoint_path: str, fusion_weight_path: str) -> None:
        torch.save(self.policy_net.state_dict(), checkpoint_path)
        self.policy_net.save_fusion_weights(fusion_weight_path)
