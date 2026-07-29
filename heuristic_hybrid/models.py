import torch
import torch.nn as nn


class HeuristicWeightScheduler(nn.Module):
    def __init__(self, in_channels: int = 5, base_channels: int = 32, hidden_dim: int = 128, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(max(1e-3, temperature))
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.spatial_compression = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp_head = nn.Sequential(
            nn.Linear(c3, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.spatial_compression(feat)
        feat = torch.flatten(feat, 1)
        logits = self.mlp_head(feat) / self.temperature
        return torch.softmax(logits, dim=-1)


class SpatialFCN(HeuristicWeightScheduler):
    pass


class SpatialActionDecision:
    def __init__(self, model, epsilon=0.1, device='cpu'):
        self.model = model
        self.epsilon = epsilon
        self.device = device

    def select_action(self, state_tensor, heuristic_matrix, explore=True):
        if heuristic_matrix is None or len(heuristic_matrix) == 0:
            return None, None, None
        state_batch = state_tensor.unsqueeze(0).to(self.device)
        hm = torch.as_tensor(heuristic_matrix, dtype=torch.float32, device=self.device).clamp(0.0, 1.0)
        with torch.no_grad():
            omega = self.model(state_batch).squeeze(0)
            scores = torch.matmul(hm, omega.unsqueeze(-1)).squeeze(-1)
        if explore and torch.rand(1).item() < self.epsilon:
            idx = torch.randint(0, hm.shape[0], (1,), device=self.device).item()
        else:
            idx = torch.argmax(scores).item()
        return idx, scores.detach().cpu().numpy(), omega.detach().cpu().numpy()
