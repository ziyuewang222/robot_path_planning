from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cv_extractor import SemanticExtractor
from core.explorer_policy import decide_frontier
from standard_test.benchmark_env_adapter import BenchmarkEnvAdapter
from standard_test.benchmark_loader import load_room_cases
from standard_test.config import resolve_standard_map_paths


def run_room_benchmark(blueprint_dir: str | Path, model_path: str | Path) -> None:
    cases = load_room_cases(blueprint_dir)
    extractor = SemanticExtractor(model_path)
    for case in cases:
        env = BenchmarkEnvAdapter(case)
        explored_map, _ = env.reset()
        total_steps = 0
        total_area = 0.0
        while True:
            semantic_map = extractor.predict(explored_map)
            decision = decide_frontier(explored_map, env.robot_pos, semantic_map)
            if decision.chosen_frontier is None:
                break
            explored_map, _, _reward, _n_steps, done, info = env.step_macro(decision.chosen_frontier)
            total_steps += 1
            total_area += float(info.get("coverage_gain", 0.0))
            if done:
                break
        print(f"{case.metadata_path.name}: steps={total_steps}, area={total_area:.1f}")


if __name__ == "__main__":
    run_room_benchmark(
        blueprint_dir=resolve_standard_map_paths()[0].parent,
        model_path=PROJECT_ROOT / "results/checkpoints/spatial_fcn_rl_latest.pth",
    )
