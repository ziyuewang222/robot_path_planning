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
        self.use_amp = torch.device(device).type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def update_step(self, batch_size=16):
        if len(self.memory) < batch_size:
            return 0.0

        transitions = self.memory.sample(batch_size)
        states, action_coords, rewards, next_states, next_masks, dones, taus = zip(*transitions)

        states = torch.stack(states, dim=0).to(self.device, dtype=torch.float32, non_blocking=True)
        next_states = torch.stack(next_states, dim=0).to(self.device, dtype=torch.float32, non_blocking=True)
        next_masks = torch.stack(next_masks, dim=0).to(self.device, dtype=torch.bool, non_blocking=True)
        action_coords = torch.tensor(action_coords, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        taus = torch.tensor(taus, dtype=torch.float32, device=self.device)

        self.policy_net.train()
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            q_raw = self.policy_net(states).squeeze(1)
            actual_batch_size = states.shape[0]
            batch_indices = torch.arange(actual_batch_size, device=self.device)
            pred_q = q_raw[batch_indices, action_coords[:, 0], action_coords[:, 1]]

            with torch.no_grad():
                q_next_raw = self.target_net(next_states).squeeze(1)
                mask_fill_value = torch.finfo(q_next_raw.dtype).min
                q_next_masked = q_next_raw.masked_fill(~next_masks, mask_fill_value)
                max_next_q = q_next_masked.flatten(start_dim=1).max(dim=1).values
                gamma_tau = self.gamma
                target_q = rewards + (~dones).float() * gamma_tau * max_next_q

            loss = self.criterion(pred_q, target_q)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(loss.detach().item())

    def select_actions_batch(self, states, epsilon=0.1):
        if not states:
            return []

        self.policy_net.eval()
        state_batch = torch.stack(states, dim=0).to(self.device, non_blocking=True)
        masks = state_batch[:, 4]
        valid_flat = (masks > 0.5).flatten(start_dim=1)

        with torch.inference_mode():
            q_raw = self.policy_net(state_batch).squeeze(1)
            q_masked = q_raw.masked_fill(masks < 0.5, -1e9)
            greedy_flat = q_masked.flatten(start_dim=1).argmax(dim=1).detach().cpu().tolist()

        _, _, width = masks.shape
        actions = []
        valid_flat_cpu = valid_flat.detach().cpu()
        for env_idx, flat_idx in enumerate(greedy_flat):
            valid_indices = torch.nonzero(valid_flat_cpu[env_idx], as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                actions.append(None)
                continue
            if random.random() < epsilon:
                flat_idx = int(valid_indices[random.randrange(valid_indices.numel())].item())
            actions.append([int(flat_idx // width), int(flat_idx % width)])
        return actions

    def update_steps(self, batch_size=16, num_updates=1):
        total_loss = 0.0
        update_count = 0
        for _ in range(max(0, num_updates)):
            loss = self.update_step(batch_size=batch_size)
            if loss > 0:
                total_loss += loss
                update_count += 1
        return total_loss, update_count

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
