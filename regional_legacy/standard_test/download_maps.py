from __future__ import annotations

import shutil
from pathlib import Path

from config import STANDARD_DATASET_DIR, STANDARD_MAP_FILES


def download_maps() -> None:
    STANDARD_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = Path("/tmp/Explore-Bench/onpolicy/onpolicy/envs/GridEnv/datasets")
    if not source_dir.exists():
        raise FileNotFoundError(f"找不到已下载的 Explore-Bench 数据目录: {source_dir}")

    copied = 0
    for filename in STANDARD_MAP_FILES:
        source = source_dir / filename
        if not source.exists():
            print(f"跳过缺失地图: {filename}")
            continue
        shutil.copy2(source, STANDARD_DATASET_DIR / filename)
        copied += 1

    print(f"已保存 {copied} 张地图到: {STANDARD_DATASET_DIR}")


if __name__ == "__main__":
    download_maps()
