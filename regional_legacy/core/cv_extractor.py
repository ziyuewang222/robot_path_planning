from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import TEST_CHECKPOINT_PATH
from core.models import SemanticUNet


class SemanticExtractor:
    """加载一次测试 checkpoint，并为连续观测地图生成语义预测。"""

    def __init__(self, model_path: Optional[str | Path] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SemanticUNet(in_channels=1, num_classes=3).to(self.device)
        self.model.load_state_dict(self._load_checkpoint(model_path), strict=False)
        self.model.eval()

    @staticmethod
    def _load_checkpoint(model_path: Optional[str | Path] = None):
        weight_path = Path(model_path) if model_path is not None else Path(TEST_CHECKPOINT_PATH)
        if not weight_path.exists():
            raise FileNotFoundError(f"找不到测试权重文件: {weight_path}")
        try:
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            return state["state_dict"]
        return state

    def predict(self, explored_map: np.ndarray) -> np.ndarray:
        if explored_map.ndim != 2:
            raise ValueError("explored_map 必须是二维地图")

        x = torch.from_numpy(explored_map.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)


def build_semantic_map(explored_map: np.ndarray, model_path: Optional[str | Path] = None) -> np.ndarray:
    """兼容单次调用接口；连续推理应使用 `SemanticExtractor`。"""
    return SemanticExtractor(model_path).predict(explored_map)
