from __future__ import annotations

from pathlib import Path

STANDARD_BLUEPRINT_DIR = Path("/tmp/Explore-Bench/exploration_benchmark/blueprints")
STANDARD_DATASET_DIR = Path(__file__).resolve().parent / "maps"
STANDARD_MAP_NAME = "room_with_corner"
STANDARD_MAP_PATH = STANDARD_BLUEPRINT_DIR / f"{STANDARD_MAP_NAME}.yaml"
STANDARD_MAP_FILE = STANDARD_DATASET_DIR / f"{STANDARD_MAP_NAME}.pgm"

# Explore-Bench 仓库中实际提供的离线栅格地图文件。
STANDARD_MAP_FILES = [
    "corner.pgm",
    "corridor.pgm",
    "corridor_asym.pgm",
    "corridor_sym.pgm",
    "loop_with_corridor_sym.pgm",
    "room1_modified.pgm",
    "room_1.pgm",
    "room_2.pgm",
    "room_3.pgm",
    "room_with_corner.pgm",
    "room_with_corridor.pgm",
    "square_loop.pgm",
]

# 用于查看 YAML 蓝图的标准地图名。
STANDARD_MAP_NAMES = [
    "room",
    "corner",
    "corridor",
    "loop",
    "room_with_corner",
    "loop_with_corridor",
]


def resolve_standard_map_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    return STANDARD_MAP_PATH


def resolve_standard_map_paths(explicit_names: list[str] | None = None) -> list[Path]:
    names = explicit_names if explicit_names is not None else STANDARD_MAP_NAMES
    return [STANDARD_BLUEPRINT_DIR / f"{name}.yaml" for name in names]
