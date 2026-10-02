from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from models.training_pipeline import TrainingPipeline


def _config(epochs: int, patience: int):
    return OmegaConf.create(
        {
            "runtime": {"device": "cpu"},
            "training": {
                "epochs": epochs,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "patience": patience,
            },
        }
    )


def _graph(index: int):
    x = torch.full((2, 3), float(index + 1))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.ones(2, 2),
        y=torch.zeros(2, 1),
        y_mechanism_trajectory=torch.zeros(2, 1, 3),
    )


class _ScalarRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.scale * x[:, :1]


class _TrajectoryRegressor(torch.nn.Module):
    target_name = "y_mechanism_trajectory"
    checkpoint_prefix = "trajectory_observability_"

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.trajectory_loss = torch.nn.MSELoss()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.scale * x[:, :3].unsqueeze(1)


def _metrics(loss: float, r2: float):
    return {"loss": loss, "mae": loss + 0.1, "rmse": loss + 0.2, "r2": r2}


def _set_evaluation_sequence(pipeline, validation_metrics, test_metrics):
    values = iter(validation_metrics)

    def evaluate(split="test"):
        return next(values) if split == "val" else test_metrics

    pipeline.evaluate = evaluate


def test_history_and_best_epoch_follow_existing_loss_monitor_for_non_trajectory_model(tmp_path):
    pipeline = TrainingPipeline(
        _config(epochs=3, patience=3),
        _ScalarRegressor(),
        [_graph(index) for index in range(4)],
        checkpoint_dir=tmp_path,
    )
    validation = [_metrics(0.6, 0.1), _metrics(0.2, 0.2), _metrics(0.4, 0.3)]
    _set_evaluation_sequence(pipeline, validation, _metrics(0.3, 0.0))

    pipeline.train()

    summary = pipeline.training_summary
    assert [record["epoch"] for record in summary["history"]] == [1, 2, 3]
    assert [record["train_loss"] for record in summary["history"]]
    assert [record["val_loss"] for record in summary["history"]] == [0.6, 0.2, 0.4]
    assert all("val_mae" in record and "val_rmse" in record and "val_r2" in record for record in summary["history"])
    assert all("learning_rate" in record for record in summary["history"])
    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == pytest.approx(0.2)
    assert summary["best_monitor_name"] == "loss"
    assert summary["best_monitor_value"] == pytest.approx(0.2)
    assert summary["epochs_requested"] == 3
    assert summary["epochs_completed"] == 3
    assert summary["stopped_early"] is False
    assert summary["training_seconds"] >= 0.0
    assert summary["evaluation_seconds"] >= 0.0
    assert summary["peak_cuda_memory_bytes"] is None


def test_early_stopping_metadata_matches_completed_history(tmp_path):
    pipeline = TrainingPipeline(
        _config(epochs=5, patience=1),
        _ScalarRegressor(),
        [_graph(index) for index in range(4)],
        checkpoint_dir=tmp_path,
    )
    _set_evaluation_sequence(
        pipeline,
        [_metrics(0.1, 0.0), _metrics(0.2, 0.0)],
        _metrics(0.1, 0.0),
    )

    pipeline.train()

    summary = pipeline.training_summary
    assert summary["epochs_requested"] == 5
    assert summary["epochs_completed"] == len(summary["history"]) == 2
    assert summary["best_epoch"] == 1
    assert summary["stopped_early"] is True


def test_trajectory_metadata_preserves_r2_checkpoint_selection(tmp_path):
    pipeline = TrainingPipeline(
        _config(epochs=3, patience=3),
        _TrajectoryRegressor(),
        [_graph(index) for index in range(4)],
        checkpoint_dir=tmp_path,
    )
    _set_evaluation_sequence(
        pipeline,
        [_metrics(0.1, 0.2), _metrics(0.5, 0.8), _metrics(0.2, 0.4)],
        _metrics(0.3, 0.0),
    )

    pipeline.train()

    summary = pipeline.training_summary
    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == pytest.approx(0.5)
    assert summary["best_monitor_name"] == "r2"
    assert summary["best_monitor_value"] == pytest.approx(0.8)
