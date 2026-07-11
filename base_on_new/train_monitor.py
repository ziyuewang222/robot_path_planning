import torch

from config import MAP_GENERATOR, SENSOR_RANGE
from grid_env import GridWorldEnv
from models import SpatialActionDecision
from state_pipeline import StateTensorPipeline

PARAM_REPORT_INTERVAL = 20
PROBE_SEED_START = 2000
PROBE_STATE_COUNT = 16


def snapshot_params(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def param_change_report(model, previous_snapshot, eps=1e-12):
    module_stats = {}
    total_diff_sq = 0.0
    total_base_sq = 0.0

    for name, param in model.named_parameters():
        if not param.requires_grad or name not in previous_snapshot:
            continue

        current = param.detach().cpu()
        previous = previous_snapshot[name]
        diff_sq = torch.sum((current - previous) ** 2).item()
        base_sq = torch.sum(previous ** 2).item()

        total_diff_sq += diff_sq
        total_base_sq += base_sq

        module_name = name.split(".")[0]
        if module_name not in module_stats:
            module_stats[module_name] = {"diff_sq": 0.0, "base_sq": 0.0}
        module_stats[module_name]["diff_sq"] += diff_sq
        module_stats[module_name]["base_sq"] += base_sq

    total_change = (total_diff_sq ** 0.5) / ((total_base_sq ** 0.5) + eps)
    module_changes = {
        module_name: (stats["diff_sq"] ** 0.5) / ((stats["base_sq"] ** 0.5) + eps)
        for module_name, stats in module_stats.items()
    }
    return total_change, module_changes


def build_probe_states(map_size, device, count=PROBE_STATE_COUNT, seed_start=PROBE_SEED_START):
    pipeline = StateTensorPipeline(map_size=map_size, device=device)
    probe_states = []
    for seed in range(seed_start, seed_start + count):
        env = GridWorldEnv(
            map_size=map_size,
            sensor_range=SENSOR_RANGE,
            seed=seed,
            map_generator=MAP_GENERATOR,
        )
        explored, pos = env.reset()
        probe_states.append(pipeline.process(explored, pos).cpu())
    return probe_states


def probe_actions(model, probe_states, device):
    decision = SpatialActionDecision(model, epsilon=0.0, device=device)
    actions = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for state in probe_states:
            action, _, _ = decision.select_action(state.to(device), explore=False, return_maps=False)
            actions.append(tuple(action) if action is not None else None)
    if was_training:
        model.train()
    return actions


def action_change_rate(current_actions, previous_actions):
    if not current_actions or not previous_actions or len(current_actions) != len(previous_actions):
        return 0.0, 0, 0
    changed = sum(curr != prev for curr, prev in zip(current_actions, previous_actions))
    total = len(current_actions)
    return changed / total, changed, total


def format_module_changes(module_changes):
    preferred_order = [
        "scale_fusion_logits",
        "full_res_branch",
        "half_res_branch",
        "quarter_res_branch",
        "fuse_proj_full",
        "fuse_proj_half",
        "fuse_proj_quarter",
        "head",
    ]
    parts = []
    used = set()
    for name in preferred_order:
        if name in module_changes:
            parts.append(f"{name}={module_changes[name]:.3e}")
            used.add(name)
    for name in sorted(module_changes):
        if name not in used:
            parts.append(f"{name}={module_changes[name]:.3e}")
    return " | ".join(parts)


def print_update_monitor(episode, model, param_snapshot, previous_probe_actions, probe_states, device, interval=PARAM_REPORT_INTERVAL):
    total_change, module_changes = param_change_report(model, param_snapshot)
    current_probe_actions = probe_actions(model, probe_states, device)
    action_rate, changed, total = action_change_rate(current_probe_actions, previous_probe_actions)
    print(
        f"[UPDATE MONITOR {interval}ep] episode={episode} | "
        f"param_total={total_change:.3e} | "
        f"action_change={action_rate * 100.0:.2f}% ({changed}/{total})"
    )
    print(f"[PARAM MODULE Δ] {format_module_changes(module_changes)}")
    return snapshot_params(model), current_probe_actions
