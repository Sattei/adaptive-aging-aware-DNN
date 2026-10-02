"""Fair, validation-selected static-versus-temporal final experiment helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.temporal_refinement import (
    aggregate_multiseed_results,
    run_static_experiment,
    run_temporal_experiment,
)


def dataset_fingerprint(dataset: Any) -> str:
    """Hash the model-visible graph content and temporal target tensors."""
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        sample = dataset[index]
        for name in ("x", "x_history", "edge_index", "edge_attr", "y_mechanism_trajectory"):
            value = getattr(sample, name)
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def aggregate_lr_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate test metrics and validation-selected checkpoint values by LR."""
    result = aggregate_multiseed_results(runs)
    values = np.asarray([run["training"]["best_val_loss"] for run in runs.values()], dtype=float)
    result["validation_loss"] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(values.size),
    }
    for field in ("training_seconds", "peak_cuda_memory_bytes"):
        raw = [run["training"].get(field) for run in runs.values()]
        if any(value is None for value in raw):
            result[field] = None
        else:
            values = np.asarray(raw, dtype=float)
            result[field] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": int(values.size)}
    return result


def select_best_lr(lr_results: dict[str, dict[str, Any]]) -> str:
    """Choose exclusively by mean validation loss; test metrics are not consulted."""
    if not lr_results:
        raise ValueError("At least one learning-rate result is required")
    return min(
        lr_results,
        key=lambda lr: (lr_results[lr]["aggregate"]["validation_loss"]["mean"], float(lr)),
    )


def comparison_deltas(static_aggregate: dict[str, Any], temporal_aggregate: dict[str, Any]) -> dict[str, float]:
    """Return neutral temporal-minus-static metrics and descriptive MAE change."""
    static = static_aggregate["aggregate"]
    temporal = temporal_aggregate["aggregate"]
    static_mae = static["mae"]["mean"]
    temporal_mae = temporal["mae"]["mean"]
    return {
        "temporal_minus_static_MAE": float(temporal_mae - static_mae),
        "temporal_minus_static_RMSE": float(temporal["rmse"]["mean"] - static["rmse"]["mean"]),
        "temporal_minus_static_R2": float(temporal["r2"]["mean"] - static["r2"]["mean"]),
        "relative_mae_change_percent": float((static_mae - temporal_mae) / static_mae * 100.0),
    }


def run_fair_sweep(
    config: Any,
    dataset: Any,
    output_dir: Path,
    *,
    learning_rates: list[float],
    seeds: list[int],
    split_seed: int,
    family: str,
    run_callable: Callable | None = None,
) -> dict[str, Any]:
    """Run one architecture family over an identical LR/seed grid."""
    if family not in {"static", "temporal"}:
        raise ValueError("family must be 'static' or 'temporal'")
    if run_callable is None:
        runner = run_static_experiment if family == "static" else run_temporal_experiment

        def run_callable(cfg, data, checkpoint_dir, seed, fixed_split_seed):
            return runner(cfg, data, checkpoint_dir, seed=seed, split_seed=fixed_split_seed)

    fingerprint = dataset_fingerprint(dataset)
    expected_split_hash = None
    lr_results = {}
    for learning_rate in learning_rates:
        label = f"{learning_rate:g}"
        seed_runs = {}
        for seed in seeds:
            run_cfg = copy.deepcopy(config)
            run_cfg.training.learning_rate = learning_rate
            run_cfg.training.lr = learning_rate
            result = run_callable(
                run_cfg,
                dataset,
                Path(output_dir) / family / f"lr_{label}" / f"seed_{seed}" / "checkpoints",
                seed,
                split_seed,
            )
            split_hash = result["experiment"]["split"]["indices_sha256"]
            if expected_split_hash is None:
                expected_split_hash = split_hash
            elif split_hash != expected_split_hash:
                raise RuntimeError("A final-validation run received different split indices")
            result["fairness"] = {
                "dataset_fingerprint": fingerprint,
                "split_indices_sha256": split_hash,
                "dataset_object_id": id(dataset),
            }
            seed_runs[str(seed)] = result
        lr_results[label] = {"runs": seed_runs, "aggregate": aggregate_lr_runs(seed_runs)}
    selected_lr = select_best_lr(lr_results)
    return {
        "family": family,
        "dataset_fingerprint": fingerprint,
        "split_indices_sha256": expected_split_hash,
        "learning_rates": learning_rates,
        "initialization_seeds": seeds,
        "lr_results": lr_results,
        "selected_lr": selected_lr,
    }


def build_summary(final_result: dict[str, Any]) -> str:
    """Render the required compact human-readable result summary."""
    static = final_result["static"]
    temporal = final_result["temporal"]
    comparison = final_result["final_comparison"]
    metadata = final_result["dataset"]
    lines = [
        "DATASET",
        json.dumps(metadata, indent=2),
        "STATIC LR SWEEP",
        json.dumps({lr: value["aggregate"]["validation_loss"] for lr, value in static["lr_results"].items()}, indent=2),
        "TEMPORAL LR SWEEP",
        json.dumps({lr: value["aggregate"]["validation_loss"] for lr, value in temporal["lr_results"].items()}, indent=2),
        "SELECTED STATIC CONFIG",
        static["selected_lr"],
        "SELECTED TEMPORAL CONFIG",
        temporal["selected_lr"],
        "MULTI-SEED STATIC RESULTS",
        json.dumps(static["lr_results"][static["selected_lr"]]["aggregate"]["aggregate"], indent=2),
        "MULTI-SEED TEMPORAL RESULTS",
        json.dumps(temporal["lr_results"][temporal["selected_lr"]]["aggregate"]["aggregate"], indent=2),
        "TEMPORAL HISTORY SENSITIVITY",
        json.dumps(final_result.get("history_sensitivity", {}), indent=2),
        "COMPUTE COST COMPARISON",
        json.dumps(final_result["compute_cost"], indent=2),
        "FINAL FAIR COMPARISON",
        json.dumps(comparison, indent=2),
        "KNOWN LIMITATIONS",
        "TDDB R² is reported but not interpreted because its target variance is extremely low. Edge attributes remain unused by the current GAT configuration.",
    ]
    return "\n\n".join(lines) + "\n"
