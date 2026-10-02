import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from experiments.static_temporal_comparison import (
    compute_metric_deltas,
    run_static_temporal_comparison,
    save_comparison_results,
)


def _graph(index):
    x_history = torch.full((2, 4, 8), float(index))
    return Data(
        x=x_history[:, -1, :].clone(),
        x_history=x_history,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.ones(2, 2),
        y=torch.zeros(2, 1),
        y_trajectory=torch.zeros(2, 3),
        y_mechanism_trajectory=torch.zeros(2, 3, 3),
    )


def _config():
    return OmegaConf.create(
        {
            "seed": 17,
            "runtime": {"device": "cpu"},
            "model": {
                "hidden_dim": 8,
                "gat_heads": 4,
                "transformer_layers": 1,
                "transformer_heads": 4,
                "temporal_hidden_dim": 4,
                "temporal_gru_layers": 1,
                "prediction_horizon": 3,
            },
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "patience": 1,
                "scheduler": "CosineAnnealingLR",
            },
        }
    )


def _metrics(mae):
    return {
        "loss": mae * mae,
        "mae": mae,
        "rmse": mae * 1.5,
        "r2": 0.5,
        "trajectory_metrics": {
            "aggregate": {"mae": mae, "rmse": mae * 1.5, "r2": 0.5},
            "per_mechanism": {},
            "per_horizon": [{"horizon": 1, "mae": mae, "rmse": mae * 1.5}],
            "mae_by_horizon_mechanism": [[mae, mae, mae]],
            "rmse_by_horizon_mechanism": [[mae * 1.5, mae * 1.5, mae * 1.5]],
        },
    }


class _FakePipeline:
    calls = []

    def __init__(self, config, model, dataset, checkpoint_dir=None, split_seed=42):
        self.model = model
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.split_seed = split_seed
        self.train_loader = SimpleNamespace(dataset=SimpleNamespace(indices=[0, 2]))
        self.val_loader = SimpleNamespace(dataset=SimpleNamespace(indices=[1]))
        self.test_loader = SimpleNamespace(dataset=SimpleNamespace(indices=[3]))
        type(self).calls.append(self)

    def train(self):
        mae = 0.08 if self.model.uses_x_history else 0.10
        self.training_summary = {
            "epochs_requested": 2,
            "epochs_completed": 2,
            "best_epoch": 1,
            "best_val_loss": mae * mae,
            "stopped_early": False,
            "training_seconds": 0.0,
            "evaluation_seconds": 0.0,
            "peak_cuda_memory_bytes": None,
            "history": [{"epoch": 1, "val_loss": mae * mae}],
        }
        return _metrics(mae)


def test_comparison_uses_one_dataset_and_identical_split_membership(tmp_path):
    _FakePipeline.calls = []
    dataset = [_graph(index) for index in range(4)]

    results = run_static_temporal_comparison(
        _config(), dataset, tmp_path / "checkpoints", pipeline_cls=_FakePipeline
    )

    assert len(_FakePipeline.calls) == 2
    assert all(call.dataset is dataset for call in _FakePipeline.calls)
    assert [call.split_seed for call in _FakePipeline.calls] == [17, 17]
    assert _FakePipeline.calls[0].model.uses_x_history is False
    assert _FakePipeline.calls[1].model.uses_x_history is True
    assert all(call.model.target_name == "y_mechanism_trajectory" for call in _FakePipeline.calls)
    assert results["experiment"]["split"]["counts"] == {"train": 2, "val": 1, "test": 1}


def test_comparison_result_schema_reports_metrics_parameters_and_distinct_checkpoints(tmp_path):
    _FakePipeline.calls = []
    results = run_static_temporal_comparison(
        _config(), [_graph(index) for index in range(4)], tmp_path / "checkpoints", pipeline_cls=_FakePipeline
    )

    assert set(results) == {"static", "temporal", "comparison", "experiment"}
    for arm in ("static", "temporal"):
        assert results[arm]["parameter_count"] > 0
        assert "trajectory_metrics" in results[arm]["test_metrics"]
        assert results[arm]["test_metrics"]["trajectory_metrics"]["per_horizon"]
    assert results["static"]["checkpoints"] != results["temporal"]["checkpoints"]
    assert results["comparison"]["convention"] == "temporal_minus_static"
    assert results["static"]["training"]["history"] != results["temporal"]["training"]["history"]
    assert results["static"]["training"]["training_seconds"] >= 0.0
    assert results["temporal"]["training"]["evaluation_seconds"] >= 0.0


def test_metric_delta_uses_temporal_minus_static_sign_convention():
    deltas = compute_metric_deltas(_metrics(0.10), _metrics(0.08))

    assert deltas["mae_delta"] == pytest.approx(-0.02)
    assert deltas["rmse_delta"] == pytest.approx(-0.03)
    assert deltas["convention"] == "temporal_minus_static"


def test_nan_metrics_serialize_as_json_null_without_fake_numeric_score(tmp_path):
    result_path = Path(tmp_path) / "comparison.json"
    save_comparison_results({"r2": float("nan"), "nested": [math.inf]}, result_path)

    serialized = json.loads(result_path.read_text(encoding="utf-8"))
    assert serialized == {"r2": None, "nested": [None]}
