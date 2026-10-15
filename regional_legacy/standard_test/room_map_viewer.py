from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class RoomMapViewer:
    """静态查看 Explore-Bench 房间地图的轻量工具。"""

    def __init__(self, cmap: str = "gray_r") -> None:
        self.cmap = cmap

    def show(self, grid: np.ndarray, title: Optional[str] = None) -> None:
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap=self.cmap, origin="upper")
        plt.xticks([])
        plt.yticks([])
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()

    def show_path(self, path: str | Path, title: Optional[str] = None) -> None:
        data = np.load(str(path))
        self.show(data, title=title or Path(path).name)
