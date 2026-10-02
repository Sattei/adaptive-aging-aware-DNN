"""Controlled temporal-only refinement experiments using the Stage 5 model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Type

import numpy as np

from experiments.static_temporal_comparison import (
    _cfg_get,
    build_static_model,
    build_temporal_model,
    set_experiment_seed,
    split_metadata,
    training_metadata,
)
from models.model_utils import count_trainable_parameters
from models.training_pipeline import TrainingPipeline


def _membership(pipeline: Any) -> dict[str, list[int]]:
    return {
        name: [int(index) for index in getattr(getattr(pipeline, f"{name}_loader").dataset, "indices", ())]
        for name in ("train", "val", "test")
    }


def _model_metadata(config: Any) -> dict[str, Any]:
    model_cfg = _cfg_get(config, "model", {})
    return {
        "hidden_dim": int(_cfg_get(model_cfg, "hidden_dim", 256)),
        "gnn_layers": int(_cfg_get(model_cfg, "gnn_layers", 3)),
        "gat_heads": int(_cfg_get(model_cfg, "gat_heads", 4)),
        "transformer_layers": int(_cfg_get(model_cfg, "transformer_layers", 2)),
        "transformer_heads": int(_cfg_get(model_cfg, "transformer_heads", 4)),
        "temporal_hidden_dim": int(_cfg_get(model_cfg, "temporal_hidden_dim", 128)),
        "temporal_gru_layers": int(_cfg_get(model_cfg, "temporal_gru_layers", 1)),
        "dropout": float(_cfg_get(model_cfg, "dropout", 0.1)),
        "prediction_horizon": int(_cfg_get(model_cfg, "prediction_horizon", 10)),
    }


def run_model_experiment(
    config: Any,
    dataset: Any,
    checkpoint_dir: Path,
    *,
    seed: int,
    model_builder=build_temporal_model,
    split_seed: int | None = None,
    pipeline_cls: Type[TrainingPipeline] = TrainingPipeline,
) -> dict[str, Any]:
    """Train one existing control model and return reproducible metadata."""
    if len(dataset) == 0:
        raise ValueError("Refinement requires a non-empty dataset")
    split_seed = int(seed if split_seed is None else split_seed)
    sample = dataset[0]
    set_experiment_seed(int(seed))
    model = model_builder(config, int(sample.x.shape[-1]))
    pipeline = pipeline_cls(
        config, model, dataset, checkpoint_dir=Path(checkpoint_dir), split_seed=split_seed
    )
    membership = _membership(pipeline)
    set_experiment_seed(int(seed))
    test_metrics = pipeline.train()
    training = training_metadata(pipeline)
    training_cfg = _cfg_get(config, "training", {})
    return {
        "model_name": type(model).__name__,
        "parameter_count": count_trainable_parameters(model),
        "test_metrics": test_metrics,
        "training": training,
        "checkpoints": {
            "best": str(Path(checkpoint_dir) / f"{model.checkpoint_prefix}best.pt"),
            "last": str(Path(checkpoint_dir) / f"{model.checkpoint_prefix}last.pt"),
        },
        "experiment": {
            "seed": int(seed),
            "split_seed": split_seed,
            "dataset_size": len(dataset),
            "history_length": int(sample.x_history.shape[1]),
            "node_feature_dim": int(sample.x.shape[-1]),
            "nodes_per_graph": int(sample.x.shape[0]),
            "model": _model_metadata(config),
            "training": {
                "epochs": int(_cfg_get(training_cfg, "epochs", 100)),
                "batch_size": int(_cfg_get(training_cfg, "batch_size", 32)),
                "learning_rate": float(_cfg_get(training_cfg, "learning_rate", _cfg_get(training_cfg, "lr", 1e-3))),
                "weight_decay": float(_cfg_get(training_cfg, "weight_decay", 1e-4)),
                "patience": int(_cfg_get(training_cfg, "patience", 10)),
                "scheduler": _cfg_get(training_cfg, "scheduler", "CosineAnnealingLR"),
            },
            "split": split_metadata(membership),
        },
    }


def run_temporal_experiment(
    config: Any,
    dataset: Any,
    checkpoint_dir: Path,
    *,
    seed: int,
    split_seed: int | None = None,
    pipeline_cls: Type[TrainingPipeline] = TrainingPipeline,
) -> dict[str, Any]:
    """Train the unchanged temporal model and return reproducible metadata."""
    return run_model_experiment(
        config,
        dataset,
        checkpoint_dir,
        seed=seed,
        model_builder=build_temporal_model,
        split_seed=split_seed,
        pipeline_cls=pipeline_cls,
    )


def run_static_experiment(
    config: Any,
    dataset: Any,
    checkpoint_dir: Path,
    *,
    seed: int,
    split_seed: int | None = None,
    pipeline_cls: Type[TrainingPipeline] = TrainingPipeline,
) -> dict[str, Any]:
    """Train the unchanged cutoff-state control and return matching metadata."""
    return run_model_experiment(
        config,
        dataset,
        checkpoint_dir,
        seed=seed,
        model_builder=build_static_model,
        split_seed=split_seed,
        pipeline_cls=pipeline_cls,
    )


def aggregate_multiseed_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scalar, mechanism-wise, and horizon-wise test diagnostics."""
    if not results:
        raise ValueError("At least one seed result is required")

    def mean_std(values):
        array = np.asarray(values, dtype=np.float64)
        return {"mean": float(np.mean(array)), "std": float(np.std(array)), "count": int(array.size)}

    metric_names = ("loss", "mae", "rmse", "r2")
    aggregate = {
        name: mean_std([result["test_metrics"][name] for result in results.values()])
        for name in metric_names
    }
    trajectories = [result["test_metrics"].get("trajectory_metrics") for result in results.values()]
    if any(item is None for item in trajectories):
        return {"aggregate": aggregate}

    mechanism_names = trajectories[0]["per_mechanism"].keys()
    per_mechanism = {
        mechanism: {
            metric: mean_std([item["per_mechanism"][mechanism][metric] for item in trajectories])
            for metric in ("mae", "rmse", "r2")
        }
        for mechanism in mechanism_names
    }
    horizon_count = len(trajectories[0]["per_horizon"])
    per_horizon = [
        {
            "horizon": horizon + 1,
            **{
                metric: mean_std([item["per_horizon"][horizon][metric] for item in trajectories])
                for metric in ("mae", "rmse")
            },
        }
        for horizon in range(horizon_count)
    ]
    return {"aggregate": aggregate, "per_mechanism": per_mechanism, "per_horizon": per_horizon}


def results_fingerprint(result: dict[str, Any]) -> str:
    """Stable compact identity for a refinement result's experiment metadata."""
    encoded = json.dumps(result["experiment"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
