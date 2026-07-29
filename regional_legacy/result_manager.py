"""regional_legacy 的训练结果与权重管理工具。

使用方法
--------
1. 训练脚本通过 `create_run_dirs()` 创建当前 run 目录。
2. 训练结束时使用 `save_model_artifacts()` 保存 latest / best 权重。
3. 评估或可视化脚本通过 `latest_checkpoint_path()`、`best_checkpoint_path()` 等函数加载权重。
4. 如果你想手动指定某次 run 的结果，直接传入 `run_dir` 或目标文件路径即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import PROJECT_ROOT, EXPERIMENT_PREFIX

RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
CHECKPOINTS_ROOT = RESULTS_ROOT / "checkpoints"
FUSION_WEIGHTS_ROOT = RESULTS_ROOT / "fusion_weights"
LOGS_ROOT = RESULTS_ROOT / "logs"
METRICS_ROOT = RESULTS_ROOT / "metrics"
CONFIG_SNAPSHOTS_ROOT = RESULTS_ROOT / "configs"


@dataclass(frozen=True)
class RunPaths:
    run_name: str
    run_dir: Path
    checkpoints_dir: Path
    fusion_weights_dir: Path
    logs_dir: Path
    metrics_dir: Path
    configs_dir: Path


def create_run_dirs(run_name: Optional[str] = None) -> RunPaths:
    run_name = run_name or f"{EXPERIMENT_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_ROOT / run_name
    paths = RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        fusion_weights_dir=run_dir / "fusion_weights",
        logs_dir=run_dir / "logs",
        metrics_dir=run_dir / "metrics",
        configs_dir=run_dir / "configs",
    )
    for path in [
        RESULTS_ROOT,
        RUNS_ROOT,
        CHECKPOINTS_ROOT,
        FUSION_WEIGHTS_ROOT,
        LOGS_ROOT,
        METRICS_ROOT,
        CONFIG_SNAPSHOTS_ROOT,
        paths.run_dir,
        paths.checkpoints_dir,
        paths.fusion_weights_dir,
        paths.logs_dir,
        paths.metrics_dir,
        paths.configs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def latest_checkpoint_path() -> Path:
    return CHECKPOINTS_ROOT / f"{EXPERIMENT_PREFIX}_latest.pth"


def best_checkpoint_path() -> Path:
    return CHECKPOINTS_ROOT / f"{EXPERIMENT_PREFIX}_best.pth"


def latest_fusion_weights_path() -> Path:
    return FUSION_WEIGHTS_ROOT / f"{EXPERIMENT_PREFIX}_fusion_latest.pth"


def best_fusion_weights_path() -> Path:
    return FUSION_WEIGHTS_ROOT / f"{EXPERIMENT_PREFIX}_fusion_best.pth"


def save_model_artifacts(trainer, run_paths: RunPaths, is_best: bool = False) -> None:
    """保存当前权重。

    trainer 需要提供 `save_all_weights(checkpoint_path, fusion_path)` 接口。
    """
    latest_ckpt = latest_checkpoint_path()
    latest_fusion = latest_fusion_weights_path()
    run_ckpt = run_paths.checkpoints_dir / "latest_checkpoint.pth"
    run_fusion = run_paths.fusion_weights_dir / "latest_fusion_weights.pth"

    trainer.save_all_weights(str(latest_ckpt), str(latest_fusion))
    trainer.save_all_weights(str(run_ckpt), str(run_fusion))

    if is_best:
        trainer.save_all_weights(str(best_checkpoint_path()), str(best_fusion_weights_path()))
        trainer.save_all_weights(
            str(run_paths.checkpoints_dir / "best_checkpoint.pth"),
            str(run_paths.fusion_weights_dir / "best_fusion_weights.pth"),
        )


def load_checkpoint_path(run_dir: Optional[Path] = None, best: bool = False) -> Path:
    if run_dir is None:
        return best_checkpoint_path() if best else latest_checkpoint_path()
    return run_dir / "checkpoints" / ("best_checkpoint.pth" if best else "latest_checkpoint.pth")


def load_fusion_weights_path(run_dir: Optional[Path] = None, best: bool = False) -> Path:
    if run_dir is None:
        return best_fusion_weights_path() if best else latest_fusion_weights_path()
    return run_dir / "fusion_weights" / ("best_fusion_weights.pth" if best else "latest_fusion_weights.pth")
