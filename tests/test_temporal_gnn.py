import pytest
import torch
import torch.nn.functional as F
from aging_models.aging_label_generator import AgingLabelGenerator
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.temporal_gnn import TemporalGNN, TemporalMechanismTrajectoryGNN
from models.training_pipeline import TrainingPipeline


def _edge_index(num_nodes):
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    source = torch.arange(num_nodes, dtype=torch.long)
    target = torch.roll(source, shifts=-1)
    return torch.stack([source, target])


def _graph(num_nodes, history_length=4, horizon=3):
    edge_index = _edge_index(num_nodes)
    return Data(
        x_history=torch.rand(num_nodes, history_length, 8),
        x=torch.rand(num_nodes, 8),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), 2),
        y=torch.rand(num_nodes, 1),
        y_trajectory=torch.rand(num_nodes, horizon),
        y_mechanism_trajectory=torch.rand(num_nodes, horizon, 3),
    )


def _model(hidden_dim=32, temporal_hidden_dim=16):
    return TemporalGNN(
        node_feature_dim=8,
        hidden_dim=hidden_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        gnn_layers=1,
        gat_heads=4,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )


def _trajectory_model(horizon=3, hidden_dim=32, temporal_hidden_dim=16):
    return TemporalMechanismTrajectoryGNN(
        node_feature_dim=8,
        horizon=horizon,
        hidden_dim=hidden_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        gnn_layers=1,
        gat_heads=4,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )


def test_temporal_gnn_forward_shape():
    model = _model().eval()
    graph = _graph(5)

    output = model(graph.x_history, graph.edge_index, graph.edge_attr)

    assert output.shape == (5, 1)


def test_temporal_gnn_supports_batched_variable_size_graphs():
    model = _model().eval()
    first = _graph(3)
    second = _graph(5)
    batch = next(iter(DataLoader([first, second], batch_size=2, shuffle=False)))

    batched_output = model(
        batch.x_history, batch.edge_index, batch.edge_attr, batch.batch
    )
    first_output = model(
        first.x_history,
        first.edge_index,
        first.edge_attr,
        torch.zeros(first.num_nodes, dtype=torch.long),
    )
    second_output = model(
        second.x_history,
        second.edge_index,
        second.edge_attr,
        torch.zeros(second.num_nodes, dtype=torch.long),
    )

    assert batched_output.shape == (8, 1)
    assert torch.allclose(batched_output[:3], first_output, atol=1e-6)
    assert torch.allclose(batched_output[3:], second_output, atol=1e-6)


def test_temporal_gnn_processes_oldest_to_newest_and_uses_earlier_history(monkeypatch):
    model = _model(hidden_dim=4, temporal_hidden_dim=1).eval()
    observed = []

    def fake_spatial_encoder(snapshot, edge_index, edge_attr=None, batch=None):
        observed.append(snapshot.detach().clone())
        return snapshot[:, :1].repeat(1, model.hidden_dim)

    monkeypatch.setattr(model, "encode_spatial_snapshot", fake_spatial_encoder)

    with torch.no_grad():
        for parameter in model.temporal_gru.parameters():
            parameter.zero_()
        model.temporal_gru.weight_ih_l0[2, 0] = 1.0
        for layer in (model.regression_head[0], model.regression_head[3], model.regression_head[6]):
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight[0, 0] = 1.0

    history = torch.zeros(2, 4, 8)
    history[:, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    edge_index = _edge_index(2)
    model(history, edge_index)

    assert len(observed) == 4
    assert [frame[0, 0].item() for frame in observed] == [1.0, 2.0, 3.0, 4.0]

    same_cutoff_a = history.clone()
    same_cutoff_b = history.clone()
    same_cutoff_b[:, 0, 0] = 0.0
    output_a = model(same_cutoff_a, edge_index)
    output_b = model(same_cutoff_b, edge_index)

    assert torch.equal(same_cutoff_a[:, -1, :], same_cutoff_b[:, -1, :])
    assert not torch.allclose(output_a, output_b)


def test_temporal_gnn_validates_history_shape():
    model = _model()
    with pytest.raises(ValueError, match="x_history must have shape"):
        model(torch.rand(3, 8), _edge_index(3))


def test_mechanism_trajectory_forward_shape_range_and_order():
    model = _trajectory_model(horizon=3).eval()
    graph = _graph(5, horizon=3)

    output = model(graph.x_history, graph.edge_index, graph.edge_attr)

    assert output.shape == (5, 3, 3)
    assert torch.all((output >= 0.0) & (output <= 1.0))
    assert model.mechanism_order == AgingLabelGenerator.MECHANISM_ORDER
    assert model.mechanism_order == ("nbti", "hci", "tddb")


def test_mechanism_trajectory_supports_dynamic_history_and_horizon():
    model = _trajectory_model(horizon=3).eval()
    for history_length in (2, 4, 6):
        graph = _graph(5, history_length=history_length, horizon=3)
        output = model(graph.x_history, graph.edge_index, graph.edge_attr)
        assert output.shape == (5, 3, 3)

    longer_horizon_model = _trajectory_model(horizon=7).eval()
    graph = _graph(5, horizon=7)
    assert (
        longer_horizon_model(graph.x_history, graph.edge_index, graph.edge_attr).shape
        == (5, 7, 3)
    )


def test_mechanism_trajectory_requires_positive_horizon():
    with pytest.raises(ValueError, match="horizon must be at least one"):
        _trajectory_model(horizon=0)


def test_mechanism_trajectory_batched_variable_size_graphs_are_isolated():
    model = _trajectory_model(horizon=3).eval()
    first = _graph(3, horizon=3)
    second = _graph(5, horizon=3)
    batch = next(iter(DataLoader([first, second], batch_size=2, shuffle=False)))

    batched_output = model(
        batch.x_history, batch.edge_index, batch.edge_attr, batch.batch
    )
    first_output = model(
        first.x_history,
        first.edge_index,
        first.edge_attr,
        torch.zeros(first.num_nodes, dtype=torch.long),
    )
    second_output = model(
        second.x_history,
        second.edge_index,
        second.edge_attr,
        torch.zeros(second.num_nodes, dtype=torch.long),
    )

    assert batched_output.shape == (8, 3, 3)
    assert torch.allclose(batched_output[:3], first_output, atol=1e-6)
    assert torch.allclose(batched_output[3:], second_output, atol=1e-6)


def test_mechanism_trajectory_uses_earlier_history(monkeypatch):
    model = _trajectory_model(horizon=2, hidden_dim=4, temporal_hidden_dim=1).eval()

    def fake_spatial_encoder(snapshot, edge_index, edge_attr=None, batch=None):
        return snapshot[:, :1].repeat(1, model.hidden_dim)

    monkeypatch.setattr(model, "encode_spatial_snapshot", fake_spatial_encoder)

    with torch.no_grad():
        for parameter in model.temporal_gru.parameters():
            parameter.zero_()
        model.temporal_gru.weight_ih_l0[2, 0] = 1.0
        for layer in (model.trajectory_head[0], model.trajectory_head[3]):
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight[:, 0] = 1.0

    history = torch.zeros(2, 4, 8)
    history[:, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    same_cutoff_different_history = history.clone()
    same_cutoff_different_history[:, 0, 0] = 0.0
    edge_index = _edge_index(2)

    output_a = model(history, edge_index)
    output_b = model(same_cutoff_different_history, edge_index)

    assert torch.equal(history[:, -1, :], same_cutoff_different_history[:, -1, :])
    assert not torch.allclose(output_a, output_b)


def test_mechanism_trajectory_mse_gradients_reach_head_gru_and_spatial_encoder():
    torch.manual_seed(7)
    model = _trajectory_model(horizon=3)
    graph = _graph(4, horizon=3)

    output = model(graph.x_history, graph.edge_index, graph.edge_attr)
    F.mse_loss(output, graph.y_mechanism_trajectory).backward()

    parameters = (
        model.trajectory_head[0].weight,
        model.temporal_gru.weight_ih_l0,
        model.spatial_encoder.input_proj.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


class _HistoryOnlyRegressor(torch.nn.Module):
    uses_x_history = True

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x_history, edge_index, edge_attr=None, batch=None):
        assert x_history.dim() == 3
        return self.scale * x_history[:, -1, :1]


def test_training_pipeline_routes_temporal_models_to_history(tmp_path):
    dataset = [_graph(3), _graph(4), _graph(5)]
    config = OmegaConf.create(
        {
            "runtime": {"device": "cpu"},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "patience": 1,
            },
        }
    )

    metrics = TrainingPipeline(
        config, _HistoryOnlyRegressor(), dataset, checkpoint_dir=tmp_path
    ).train()

    assert set(metrics) == {"loss", "mae", "rmse", "r2"}
    assert (tmp_path / "predictor_best.pt").exists()


def test_training_pipeline_routes_stage5b_history_and_target(tmp_path, monkeypatch):
    horizon = 3
    dataset = [_graph(3, horizon=horizon), _graph(4, horizon=horizon), _graph(5, horizon=horizon)]
    for graph in dataset:
        graph.y.fill_(0.0)
        graph.y_trajectory.fill_(1.0)
        graph.y_mechanism_trajectory.fill_(0.25)

    config = OmegaConf.create(
        {
            "runtime": {"device": "cpu"},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "patience": 1,
            },
        }
    )
    model = _trajectory_model(horizon=horizon, hidden_dim=4, temporal_hidden_dim=2)
    pipeline = TrainingPipeline(config, model, dataset, checkpoint_dir=tmp_path)
    observed_targets = []
    observed_history_shapes = []
    original_compute_loss = pipeline._compute_loss
    original_forward = model.forward

    def record_target(prediction, target):
        observed_targets.append(target.detach().cpu())
        return original_compute_loss(prediction, target)

    def record_history(x_history, edge_index, edge_attr=None, batch=None):
        observed_history_shapes.append(tuple(x_history.shape))
        return original_forward(x_history, edge_index, edge_attr, batch)

    pipeline._compute_loss = record_target
    monkeypatch.setattr(model, "forward", record_history)
    metrics = pipeline.train()

    assert observed_history_shapes
    assert all(len(shape) == 3 and shape[1:] == (4, 8) for shape in observed_history_shapes)
    assert observed_targets
    assert all(target.shape[-2:] == (horizon, 3) for target in observed_targets)
    assert all(torch.all(target == 0.25) for target in observed_targets)
    assert set(metrics) == {"loss", "mae", "rmse", "r2", "trajectory_metrics"}
    detailed_metrics = metrics["trajectory_metrics"]
    assert metrics["mae"] == pytest.approx(detailed_metrics["aggregate"]["mae"])
    assert metrics["rmse"] == pytest.approx(detailed_metrics["aggregate"]["rmse"])
    if torch.isnan(torch.tensor(metrics["r2"])):
        assert torch.isnan(torch.tensor(detailed_metrics["aggregate"]["r2"]))
    else:
        assert metrics["r2"] == pytest.approx(detailed_metrics["aggregate"]["r2"])
    assert len(detailed_metrics["per_horizon"]) == horizon
    assert torch.tensor(detailed_metrics["mae_by_horizon_mechanism"]).shape == (horizon, 3)
    assert (tmp_path / "mechanism_trajectory_best.pt").exists()
    assert (tmp_path / "mechanism_trajectory_last.pt").exists()


def test_training_pipeline_rejects_mechanism_trajectory_shape_mismatch(tmp_path):
    config = OmegaConf.create(
        {
            "runtime": {"device": "cpu"},
            "training": {"epochs": 1, "batch_size": 1},
        }
    )
    pipeline = TrainingPipeline(
        config,
        _trajectory_model(horizon=3, hidden_dim=4, temporal_hidden_dim=2),
        [_graph(2, horizon=3)],
        checkpoint_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="Prediction and target shapes must match exactly"):
        pipeline._compute_loss(torch.zeros(2, 3, 3), torch.zeros(2, 1))
