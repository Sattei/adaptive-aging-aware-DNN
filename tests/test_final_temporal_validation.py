from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from experiments.final_temporal_validation import (
    build_summary,
    comparison_deltas,
    run_fair_sweep,
    select_best_lr,
)


def _dataset():
    result = []
    for index in range(4):
        history = torch.full((2, 4, 8), float(index))
        result.append(
            Data(
                x=history[:, -1, :].clone(),
                x_history=history,
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                edge_attr=torch.ones(2, 2),
                y_mechanism_trajectory=torch.zeros(2, 2, 3),
            )
        )
    return result


def _run_result(seed, lr, val_loss, test_mae):
    return {
        "training": {"best_val_loss": val_loss, "training_seconds": 1.0, "peak_cuda_memory_bytes": 2},
        "test_metrics": {"loss": test_mae ** 2, "mae": test_mae, "rmse": test_mae * 2, "r2": 0.5},
        "experiment": {"split": {"indices_sha256": "same-split"}},
        "checkpoints": {"best": f"seed-{seed}-lr-{lr}.pt"},
    }


def test_fair_sweep_reuses_one_dataset_and_selects_only_by_validation_loss(tmp_path):
    seen_dataset_ids = []

    def runner(config, dataset, checkpoint_dir, seed, split_seed):
        seen_dataset_ids.append(id(dataset))
        lr = float(config.training.learning_rate)
        # Deliberately make test MAE favor 1e-4 while validation favors 1e-3.
        return _run_result(seed, lr, val_loss=0.1 if lr == 1e-3 else 0.2, test_mae=0.9 if lr == 1e-3 else 0.1)

    sweep = run_fair_sweep(
        OmegaConf.create({"training": {"learning_rate": 1e-4, "lr": 1e-4}}),
        _dataset(), Path(tmp_path), learning_rates=[1e-4, 1e-3], seeds=[42, 123], split_seed=42,
        family="static", run_callable=runner,
    )

    assert len(seen_dataset_ids) == 4
    assert len(set(seen_dataset_ids)) == 1
    assert sweep["selected_lr"] == "0.001"
    assert sweep["split_indices_sha256"] == "same-split"
    assert all(run["fairness"]["dataset_fingerprint"] == sweep["dataset_fingerprint"] for item in sweep["lr_results"].values() for run in item["runs"].values())


def test_lr_selection_and_deltas_are_validation_only_and_sign_correct():
    results = {
        "0.0001": {"aggregate": {"validation_loss": {"mean": 0.2}}},
        "0.001": {"aggregate": {"validation_loss": {"mean": 0.1}}},
    }
    assert select_best_lr(results) == "0.001"
    deltas = comparison_deltas(
        {"aggregate": {"mae": {"mean": 0.4}, "rmse": {"mean": 0.5}, "r2": {"mean": 0.6}}},
        {"aggregate": {"mae": {"mean": 0.3}, "rmse": {"mean": 0.45}, "r2": {"mean": 0.7}}},
    )
    assert deltas["temporal_minus_static_MAE"] == pytest.approx(-0.1)
    assert deltas["relative_mae_change_percent"] == pytest.approx(25.0)


def test_summary_contains_required_human_readable_sections():
    aggregate = {"aggregate": {"mae": {"mean": 0.1}, "rmse": {"mean": 0.2}, "r2": {"mean": 0.3}}, "validation_loss": {"mean": 0.1}}
    result = {
        "dataset": {"dataset_size": 4},
        "static": {"selected_lr": "0.001", "lr_results": {"0.001": {"aggregate": aggregate}}},
        "temporal": {"selected_lr": "0.001", "lr_results": {"0.001": {"aggregate": aggregate}}},
        "history_sensitivity": {"correct": {"mae": 0.1}},
        "compute_cost": {},
        "final_comparison": {},
    }
    summary = build_summary(result)
    for section in ("DATASET", "STATIC LR SWEEP", "TEMPORAL LR SWEEP", "FINAL FAIR COMPARISON", "KNOWN LIMITATIONS"):
        assert section in summary
