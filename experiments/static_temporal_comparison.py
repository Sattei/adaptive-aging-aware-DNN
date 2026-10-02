"""Controlled training orchestration for static versus temporal trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Type

import numpy as np
import torch

from models.current_mechanism_trajectory_gnn import CurrentMechanismTrajectoryGNN
from models.model_utils import count_trainable_parameters
from models.temporal_gnn import TemporalMechanismTrajectoryGNN
from models.training_pipeline import TrainingPipeline


def set_experiment_seed(seed: int) -> None:
    """Seed the random sources used by model construction and data ordering."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def _shared_model_kwargs(config: Any, node_feature_dim: int) -> dict[str, Any]:
    """Return the configuration shared by both comparison arms."""
    model_cfg = _cfg_get(config, "model", {})
    return {
        "node_feature_dim": node_feature_dim,
        "horizon": int(_cfg_get(model_cfg, "prediction_horizon", 10)),
        "hidden_dim": int(_cfg_get(model_cfg, "hidden_dim", 256)),
        "gnn_layers": int(_cfg_get(model_cfg, "gnn_layers", 3)),
        "gat_heads": int(_cfg_get(model_cfg, "gat_heads", 4)),
        "transformer_layers": int(_cfg_get(model_cfg, "transformer_layers", 2)),
        "transformer_heads": int(_cfg_get(model_cfg, "transformer_heads", 4)),
        "dropout": float(_cfg_get(model_cfg, "dropout", 0.1)),
    }


def build_static_model(config: Any, node_feature_dim: int) -> CurrentMechanismTrajectoryGNN:
    """Construct the cutoff-state control from the shared experiment config."""
    return CurrentMechanismTrajectoryGNN(**_shared_model_kwargs(config, node_feature_dim))


def build_temporal_model(config: Any, node_feature_dim: int) -> TemporalMechanismTrajectoryGNN:
    """Construct the history-aware model from the shared experiment config."""
    model_cfg = _cfg_get(config, "model", {})
    return TemporalMechanismTrajectoryGNN(
        temporal_hidden_dim=int(_cfg_get(model_cfg, "temporal_hidden_dim", 128)),
        gru_layers=int(_cfg_get(model_cfg, "temporal_gru_layers", 1)),
        **_shared_model_kwargs(config, node_feature_dim),
    )


def build_comparison_models(config: Any, node_feature_dim: int):
    """Construct both experiment arms with matched spatial configuration."""
    return (
        build_static_model(config, node_feature_dim),
        build_temporal_model(config, node_feature_dim),
    )


def split_membership(pipeline: Any) -> dict[str, list[int]]:
    """Return the graph indices assigned to each pipeline split."""
    memberships = {}
    for name in ("train", "val", "test"):
        dataset = getattr(pipeline, f"{name}_loader").dataset
        memberships[name] = [int(index) for index in getattr(dataset, "indices", ())]
    return memberships


def split_metadata(membership: dict[str, list[int]]) -> dict[str, Any]:
    """Record a compact, reproducible identity for the shared sample split."""
    encoded = json.dumps(membership, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "counts": {name: len(indices) for name, indices in membership.items()},
        "indices_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def training_metadata(pipeline: Any) -> dict[str, Any]:
    """Return optional observability metadata without constraining custom pipelines."""
    summary = getattr(pipeline, "training_summary", {})
    return dict(summary) if summary is not None else {}


def print_training_summary(label: str, summary: dict[str, Any]) -> None:
    """Print the compact end-of-training comparison summary."""
    if not summary:
        return

    peak_bytes = summary.get("peak_cuda_memory_bytes")
    peak_text = "not applicable" if peak_bytes is None else f"{peak_bytes / (1024 ** 2):.1f} MiB"
    print(
        f"{label}:\n"
        f"  best epoch: {summary.get('best_epoch')}\n"
        f"  best validation loss: {summary.get('best_val_loss')}\n"
        f"  training time: {summary.get('training_seconds', 0.0):.2f}s\n"
        f"  test evaluation time: {summary.get('evaluation_seconds', 0.0):.2f}s\n"
        f"  peak GPU memory: {peak_text}"
    )


def checkpoint_paths(
    checkpoint_dir: Path,
    static_model: CurrentMechanismTrajectoryGNN,
    temporal_model: TemporalMechanismTrajectoryGNN,
) -> dict[str, dict[str, Path]]:
    """Derive and validate the disjoint checkpoint paths used by both arms."""
    paths = {
        "static": {
            "best": checkpoint_dir / f"{static_model.checkpoint_prefix}best.pt",
            "last": checkpoint_dir / f"{static_model.checkpoint_prefix}last.pt",
        },
        "temporal": {
            "best": checkpoint_dir / f"{temporal_model.checkpoint_prefix}best.pt",
            "last": checkpoint_dir / f"{temporal_model.checkpoint_prefix}last.pt",
        },
    }
    all_paths = [path for model_paths in paths.values() for path in model_paths.values()]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Static and temporal comparison checkpoints must be distinct")
    return paths


def compute_metric_deltas(static_metrics: dict[str, Any], temporal_metrics: dict[str, Any]) -> dict[str, float | str]:
    """Return neutral temporal-minus-static aggregate metric differences."""
    result: dict[str, float | str] = {"convention": "temporal_minus_static"}
    for metric in ("loss", "mae", "rmse", "r2"):
        result[f"{metric}_delta"] = float(temporal_metrics[metric]) - float(
            static_metrics[metric]
        )
    return result


def run_static_temporal_comparison(
    config: Any,
    dataset: Any,
    checkpoint_dir: Path,
    seed: int | None = None,
    pipeline_cls: Type[TrainingPipeline] = TrainingPipeline,
) -> dict[str, Any]:
    """Train both controls on one dataset and verify their split identity."""
    if len(dataset) == 0:
        raise ValueError("Static/temporal comparison requires a non-empty dataset")

    checkpoint_dir = Path(checkpoint_dir)
    seed = int(_cfg_get(config, "seed", 42) if seed is None else seed)
    sample = dataset[0]
    node_feature_dim = int(sample.x.shape[-1])
    history_length = int(sample.x_history.shape[1])

    set_experiment_seed(seed)
    static_model = build_static_model(config, node_feature_dim)
    static_pipeline = pipeline_cls(
        config, static_model, dataset, checkpoint_dir=checkpoint_dir, split_seed=seed
    )

    set_experiment_seed(seed)
    temporal_model = build_temporal_model(config, node_feature_dim)
    temporal_pipeline = pipeline_cls(
        config, temporal_model, dataset, checkpoint_dir=checkpoint_dir, split_seed=seed
    )

    static_membership = split_membership(static_pipeline)
    temporal_membership = split_membership(temporal_pipeline)
    if static_membership != temporal_membership:
        raise RuntimeError("Static and temporal pipelines received different sample splits")

    paths = checkpoint_paths(checkpoint_dir, static_model, temporal_model)
    set_experiment_seed(seed)
    static_metrics = static_pipeline.train()
    static_training = training_metadata(static_pipeline)
    print_training_summary("Static", static_training)
    set_experiment_seed(seed)
    temporal_metrics = temporal_pipeline.train()
    temporal_training = training_metadata(temporal_pipeline)
    print_training_summary("Temporal", temporal_training)

    model_cfg = _cfg_get(config, "model", {})
    training_cfg = _cfg_get(config, "training", {})
    return {
        "static": {
            "model_name": type(static_model).__name__,
            "parameter_count": count_trainable_parameters(static_model),
            "test_metrics": static_metrics,
            "checkpoints": {name: str(path) for name, path in paths["static"].items()},
            "training": static_training,
        },
        "temporal": {
            "model_name": type(temporal_model).__name__,
            "parameter_count": count_trainable_parameters(temporal_model),
            "test_metrics": temporal_metrics,
            "checkpoints": {name: str(path) for name, path in paths["temporal"].items()},
            "training": temporal_training,
        },
        "comparison": compute_metric_deltas(static_metrics, temporal_metrics),
        "experiment": {
            "seed": seed,
            "split_seed": seed,
            "dataset_size": len(dataset),
            "dataset": {
                "root": str(getattr(dataset, "root", "")),
                "split": str(getattr(dataset, "split", "")),
                "seed": getattr(dataset, "seed", None),
            },
            "horizon": int(_cfg_get(model_cfg, "prediction_horizon", 10)),
            "history_length": history_length,
            "training": {
                "epochs": int(_cfg_get(training_cfg, "epochs", 100)),
                "batch_size": int(_cfg_get(training_cfg, "batch_size", 32)),
                "learning_rate": float(
                    _cfg_get(training_cfg, "learning_rate", _cfg_get(training_cfg, "lr", 1e-3))
                ),
                "weight_decay": float(_cfg_get(training_cfg, "weight_decay", 1e-4)),
                "patience": int(_cfg_get(training_cfg, "patience", 10)),
                "scheduler": _cfg_get(training_cfg, "scheduler", "CosineAnnealingLR"),
            },
            "split": split_metadata(static_membership),
        },
    }


def json_safe(value: Any) -> Any:
    """Convert non-finite numbers to JSON null rather than fake numeric scores."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_comparison_results(results: dict[str, Any], output_path: Path) -> None:
    """Write standards-compliant JSON, retaining undefined metrics as null."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(results), indent=2, allow_nan=False), encoding="utf-8"
    )
