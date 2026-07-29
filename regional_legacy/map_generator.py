# 地图生成逻辑统一入口

import numpy as np

from env import map_gen_easy_big_room


GENERATORS = {
    "easy_big_room": {
        "build": map_gen_easy_big_room.build,
        "sample_spawn": map_gen_easy_big_room.sample_spawn,
    },
}


def available_generators():
    return tuple(GENERATORS.keys())


def resolve_generator(name=None):
    generator = name or "easy_big_room"
    if generator not in GENERATORS:
        raise ValueError(f"未知地图生成器 '{generator}'，可选: {', '.join(available_generators())}")
    return generator


def build_map(map_size, generator=None):
    name = resolve_generator(generator)
    ground_truth, meta = GENERATORS[name]["build"](map_size)
    return ground_truth, meta, name


def sample_spawn(ground_truth, meta, map_size=None, generator=None):
    name = resolve_generator(generator)
    spawn_fn = GENERATORS[name]["sample_spawn"]
    return spawn_fn(ground_truth, meta)
