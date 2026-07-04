from explore.config import MAP_GENERATOR
from explore.env.maps import map_gen_random, map_gen_room

GENERATORS = {
    "room": {
        "build": map_gen_room.build,
        "sample_spawn": map_gen_room.sample_spawn,
    },
    "random": {
        "build": map_gen_random.build,
        "sample_spawn": map_gen_random.sample_spawn,
    },
}


def available_generators():
    return tuple(GENERATORS.keys())


def resolve_generator(name=None):
    generator = name or MAP_GENERATOR
    if generator not in GENERATORS:
        raise ValueError(
            f"未知地图生成器 '{generator}'，可选: {', '.join(available_generators())}"
        )
    return generator


def build_map(map_size, generator=None):
    name = resolve_generator(generator)
    ground_truth, meta = GENERATORS[name]["build"](map_size)
    return ground_truth, meta, name


def sample_spawn(ground_truth, meta, map_size, generator=None):
    name = resolve_generator(generator)
    spawn_fn = GENERATORS[name]["sample_spawn"]
    if name == "random":
        return spawn_fn(ground_truth, meta, map_size)
    return spawn_fn(ground_truth, meta)
