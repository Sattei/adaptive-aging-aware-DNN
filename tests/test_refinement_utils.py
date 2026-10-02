import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from evaluation.target_distribution import (
    analyze_mechanism_trajectories,
    compare_target_splits,
)
from experiments.refinement_utils import HistoryWindowDataset, split_indices, split_lengths
from experiments.temporal_refinement import aggregate_multiseed_results, run_temporal_experiment
from models.training_pipeline import TrainingPipeline
from scripts.run_refinement_diagnostics import HistoryVariantDataset


def _graph(index: int):
    history = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8) + index
    return Data(
        x=history[:, -1, :].clone(),
        x_history=history,
        temperature_history_k=torch.arange(4, dtype=torch.float32).reshape(1, 4),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_attr=torch.ones(1, 2),
        y=torch.zeros(1, 1),
        y_mechanism_trajectory=torch.zeros(1, 3, 3),
    )


def test_target_audit_preserves_mechanism_and_horizon_order_and_detects_saturation():
    targets = torch.tensor(
        [
            [[0.0, 0.2, 1.0], [0.1, 0.2, 1.0], [0.3, 0.2, 1.0]],
            [[0.0, 0.4, 1.0], [0.2, 0.4, 1.0], [0.4, 0.4, 1.0]],
        ]
    )
    audit = analyze_mechanism_trajectories(targets)

    assert audit["mechanism_order"] == ["nbti", "hci", "tddb"]
    assert [item["horizon"] for item in audit["mechanisms"]["nbti"]["by_horizon"]] == [1, 2, 3]
    assert audit["mechanisms"]["tddb"]["overall"]["near_constant"] is True
    assert audit["mechanisms"]["tddb"]["overall"]["near_one_saturation"] is True
    assert audit["mechanisms"]["nbti"]["overall"]["near_zero_range"] is False
    assert audit["mechanisms"]["hci"]["overall"]["low_variance"] is False
    nbti_increment = audit["mechanisms"]["nbti"]["increments"]["aggregate"]
    assert nbti_increment["positive_fraction"] == pytest.approx(1.0)
    assert nbti_increment["negative_fraction"] == pytest.approx(0.0)


def test_target_audit_reports_negative_and_near_zero_increments_without_correcting_them():
    targets = torch.zeros(1, 3, 3)
    targets[0, :, 0] = torch.tensor([0.2, 0.2, 0.1])
    audit = analyze_mechanism_trajectories(targets)

    increments = audit["mechanisms"]["nbti"]["increments"]
    assert increments["increments"][0]["near_zero_fraction"] == pytest.approx(1.0)
    assert increments["increments"][1]["negative_fraction"] == pytest.approx(1.0)


def test_split_comparison_exposes_separate_statistics():
    low = torch.zeros(2, 2, 3)
    high = torch.ones(2, 2, 3)
    result = compare_target_splits({"train": low, "test": high})

    assert set(result["splits"]) == {"train", "test"}
    assert result["comparison"]["nbti"]["mean_range"] == pytest.approx(1.0)


def test_history_window_uses_latest_frames_without_mutating_source():
    dataset = [_graph(0), _graph(1), _graph(2), _graph(3)]
    window = HistoryWindowDataset(dataset, history_length=2)

    sample = window[0]
    assert sample.x_history.shape == (1, 2, 8)
    assert torch.equal(sample.x_history, dataset[0].x_history[:, -2:, :])
    assert torch.equal(sample.temperature_history_k, dataset[0].temperature_history_k[:, -2:])
    assert torch.equal(sample.x, dataset[0].x)
    assert dataset[0].x_history.shape == (1, 4, 8)


def test_history_window_validates_available_history():
    with pytest.raises(ValueError, match="history_length"):
        HistoryWindowDataset([_graph(0)], history_length=5)


def test_history_diagnostic_variants_are_deterministic_and_do_not_mutate_source():
    dataset = [_graph(0)]
    source = dataset[0].x_history.clone()
    reversed_history = HistoryVariantDataset(dataset, "reversed")[0].x_history
    repeated_history = HistoryVariantDataset(dataset, "repeat_latest")[0].x_history
    shuffled_history = HistoryVariantDataset(dataset, "shuffle_earlier")[0].x_history

    assert torch.equal(reversed_history, torch.flip(source, dims=(1,)))
    assert torch.equal(repeated_history, source[:, -1:, :].expand_as(source))
    assert torch.equal(shuffled_history[:, -1, :], source[:, -1, :])
    assert torch.equal(dataset[0].x_history, source)


def test_refinement_split_membership_matches_training_pipeline_policy(tmp_path):
    dataset = [_graph(index) for index in range(12)]
    config = {"runtime": {"device": "cpu"}, "training": {"epochs": 1, "batch_size": 2}}
    pipeline = TrainingPipeline(config, torch.nn.Linear(8, 1), dataset, checkpoint_dir=tmp_path, split_seed=42)
    expected = split_indices(dataset, seed=42)

    assert split_lengths(len(dataset)) == (9, 1, 2)
    assert [int(index) for index in pipeline.train_loader.dataset.indices] == expected["train"]
    assert [int(index) for index in pipeline.val_loader.dataset.indices] == expected["val"]
    assert [int(index) for index in pipeline.test_loader.dataset.indices] == expected["test"]


class _FakeTemporalPipeline:
    def __init__(self, config, model, dataset, checkpoint_dir=None, split_seed=42):
        self.model = model
        self.train_loader = type("Loader", (), {"dataset": type("Subset", (), {"indices": [0, 1]})()})()
        self.val_loader = type("Loader", (), {"dataset": type("Subset", (), {"indices": [2]})()})()
        self.test_loader = type("Loader", (), {"dataset": type("Subset", (), {"indices": [3]})()})()
        self.training_summary = {
            "epochs_requested": 2,
            "epochs_completed": 2,
            "best_epoch": 2,
            "best_val_loss": 0.1,
            "history": [{"epoch": 1}, {"epoch": 2}],
        }

    def train(self):
        return {"loss": 0.1, "mae": 0.2, "rmse": 0.3, "r2": 0.4}


def test_temporal_refinement_result_records_reproducible_history_metadata(tmp_path):
    config = OmegaConf.create(
        {
            "model": {"hidden_dim": 8, "gat_heads": 4, "transformer_layers": 1, "transformer_heads": 4, "temporal_hidden_dim": 4, "temporal_gru_layers": 1, "prediction_horizon": 3},
            "training": {"epochs": 2, "batch_size": 2, "learning_rate": 1e-4, "weight_decay": 0.0, "patience": 2},
        }
    )
    result = run_temporal_experiment(
        config, [_graph(index) for index in range(4)], tmp_path, seed=123, split_seed=42, pipeline_cls=_FakeTemporalPipeline
    )

    assert result["experiment"]["seed"] == 123
    assert result["experiment"]["split_seed"] == 42
    assert result["experiment"]["history_length"] == 4
    assert result["experiment"]["split"]["counts"] == {"train": 2, "val": 1, "test": 1}
    assert result["training"]["history"][-1]["epoch"] == 2


def test_multiseed_aggregation_retains_mechanism_and_horizon_axes():
    trajectory = {
        "per_mechanism": {name: {"mae": 0.1, "rmse": 0.2, "r2": 0.3} for name in ("nbti", "hci", "tddb")},
        "per_horizon": [{"horizon": 1, "mae": 0.1, "rmse": 0.2}, {"horizon": 2, "mae": 0.2, "rmse": 0.3}],
    }
    results = {
        "42": {"test_metrics": {"loss": 0.1, "mae": 0.2, "rmse": 0.3, "r2": 0.4, "trajectory_metrics": trajectory}},
        "123": {"test_metrics": {"loss": 0.3, "mae": 0.4, "rmse": 0.5, "r2": 0.6, "trajectory_metrics": trajectory}},
    }
    aggregate = aggregate_multiseed_results(results)

    assert aggregate["aggregate"]["mae"]["mean"] == pytest.approx(0.3)
    assert aggregate["aggregate"]["mae"]["std"] == pytest.approx(0.1)
    assert set(aggregate["per_mechanism"]) == {"nbti", "hci", "tddb"}
    assert [item["horizon"] for item in aggregate["per_horizon"]] == [1, 2]
