from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RoomMapCase:
    map_path: Path
    metadata_path: Path
    grid: np.ndarray
    resolution: float
    origin: tuple[float, float, float]
    occupied_thresh: float
    free_thresh: float


def _parse_yaml_kv(path: Path) -> dict:
    metadata: dict = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _parse_float_list(text: str) -> tuple[float, float, float]:
    items = [item.strip() for item in text.strip().strip("[]").split(",") if item.strip()]
    return tuple(float(item) for item in items)  # type: ignore[return-value]


def load_room_case(metadata_path: str | Path) -> RoomMapCase:
    metadata_path = Path(metadata_path).resolve()
    metadata = _parse_yaml_kv(metadata_path)
    map_name = str(metadata["image"])
    candidates = [
        metadata_path.parent / map_name,
        metadata_path.parent.parent / "onpolicy" / "onpolicy" / "envs" / "GridEnv" / "datasets" / map_name,
        Path("/tmp/Explore-Bench/onpolicy/onpolicy/envs/GridEnv/datasets") / map_name,
    ]
    map_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    image = np.asarray(Image.open(map_path).convert("L"), dtype=np.uint8)
    return RoomMapCase(
        map_path=map_path,
        metadata_path=metadata_path,
        grid=image,
        resolution=float(metadata["resolution"]),
        origin=_parse_float_list(metadata["origin"]),
        occupied_thresh=float(metadata["occupied_thresh"]),
        free_thresh=float(metadata["free_thresh"]),
    )


def load_room_cases(blueprint_dir: str | Path) -> list[RoomMapCase]:
    return [load_room_case(path) for path in sorted(Path(blueprint_dir).glob("room*.yaml"))]
