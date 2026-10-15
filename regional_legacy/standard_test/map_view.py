from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from standard_test.config import STANDARD_DATASET_DIR, STANDARD_MAP_FILE, STANDARD_MAP_FILES, STANDARD_MAP_NAME


def show_map(map_path: str | Path) -> None:
    path = Path(map_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到地图文件: {path}")

    grid = Image.open(path).convert("L")

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r", origin="upper")
    plt.xticks([])
    plt.yticks([])
    plt.title(path.stem)
    plt.tight_layout()
    plt.show()


def list_available_maps() -> None:
    print("可用标准地图如下：")
    for filename in STANDARD_MAP_FILES:
        path = STANDARD_DATASET_DIR / filename
        if path.exists():
            print(f"- {path.stem}: {path}")
        else:
            print(f"- {Path(filename).stem}: missing -> {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_available_maps()
    elif len(sys.argv) > 1:
        show_map(sys.argv[1])
    else:
        show_map(STANDARD_MAP_FILE)
