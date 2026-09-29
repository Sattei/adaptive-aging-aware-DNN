import pytest
import torch
import torch.nn.functional as F
from aging_models.aging_label_generator import AgingLabelGenerator
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.current_mechanism_trajectory_gnn import CurrentMechanismTrajectoryGNN
from models.model_utils import count_trainable_parameters
from models.temporal_gnn import TemporalMechanismTrajectoryGNN
from models.training_pipeline import TrainingPipeline


def _edge_index(num_nodes):
    source = torch.arange(num_nodes, dtype=torch.long)
    return torch.stack([source, torch.roll(source, shifts=-1)])


def _graph(num_nodes, horizon=3, history_length=4):
    edge_index = _edge_index(num_nodes)
    x_history = torch.rand(num_nodes, history_length, 8)
    return Data(
        x=x_history[:, -1, :].clone(),
        x_history=x_history,
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), 2),
        y=torch.rand(num_nodes, 1),
        y_trajectory=torch.rand(num_nodes, horizon),
        y_mechanism_trajectory=torch.rand(num_nodes, horizon, 3),
    )


def _model(horizon=3, hidden_dim=32):
    return CurrentMechanismTrajectoryGNN(
        node_feature_dim=8,
        horizon=horizon,
        hidden_dim=hidden_dim,
        gnn_layers=1,
        gat_heads=4,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )


def test_static_mechanism_trajectory_shape_range_and_order():
    model = _model(horizon=3).eval()
    graph = _graph(5, horizon=3)

    output = model(graph.x, graph.edge_index, graph.edge_attr)

    assert output.shape == (5, 3, 3)
    assert torch.all((output >= 0.0) & (output <= 1.0))
    assert model.mechanism_order == AgingLabelGenerator.MECHANISM_ORDER
    assert model.mechanism_order == ("nbti", "hci", "tddb")


@pytest.mark.parametrize("horizon", [1, 3, 7])
def test_static_mechanism_trajectory_supports_dynamic_horizon(horizon):
    graph = _graph(4, horizon=horizon)
    output = _model(horizon=horizon).eval()(graph.x, graph.edge_index, graph.edge_attr)

    assert output.shape == (4, horizon, 3)


def test_static_mechanism_trajectory_requires_positive_horizon():
    with pytest.raises(ValueError, match="horizon must be at least one"):
        _model(horizon=0)


def test_static_mechanism_trajectory_mixed_size_batches_are_isolated():
    model = _model().eval()
    first = _graph(3)
    second = _graph(5)
    batch = next(iter(DataLoader([first, second], batch_size=2, shuffle=False)))

    batched_output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    first_output = model(
        first.x,
        first.edge_index,
        first.edge_attr,
        torch.zeros(first.num_nodes, dtype=torch.long),
    )
    second_output = model(
        second.x,
        second.edge_index,
        second.edge_attr,
        torch.zeros(second.num_nodes, dtype=torch.long),
    )

    assert batched_output.shape == (8, 3, 3)
    assert torch.allclose(batched_output[:3], first_output, atol=1e-6)
    assert torch.allclose(batched_output[3:], second_output, atol=1e-6)


def test_static_mechanism_trajectory_gradients_reach_head_and_spatial_encoder():
    torch.manual_seed(7)
    model = _model()
    graph = _graph(4)

    output = model(graph.x, graph.edge_index, graph.edge_attr)
    F.mse_loss(output, graph.y_mechanism_trajectory).backward()

    for parameter in (
        model.trajectory_head[0].weight,
        model.spatial_encoder.input_proj.weight,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def test_static_mechanism_trajectory_is_independent_of_history():
    model = _model().eval()
    graph = _graph(4)
    graph_with_different_history = graph.clone()
    graph_with_different_history.x_history[:, 0, :] = (
        1.0 - graph_with_different_history.x_history[:, 0, :]
    )

    output_a = model(graph.x, graph.edge_index, graph.edge_attr)
    output_b = model(
        graph_with_different_history.x,
        graph_with_different_history.edge_index,
        graph_with_different_history.edge_attr,
    )

    assert torch.equal(graph.x, graph.x_history[:, -1, :])
    assert torch.equal(graph.x, graph_with_different_history.x)
    assert not torch.equal(graph.x_history, graph_with_different_history.x_history)
    assert torch.allclose(output_a, output_b)


def test_parameter_count_helper_reports_static_and_temporal_models():
    static_model = _model(horizon=3)
    temporal_model = TemporalMechanismTrajectoryGNN(
        node_feature_dim=8,
        horizon=3,
        hidden_dim=32,
        temporal_hidden_dim=16,
        gnn_layers=1,
        gat_heads=4,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )

    static_parameters = count_trainable_parameters(static_model)
    temporal_parameters = count_trainable_parameters(temporal_model)

    assert static_parameters == sum(p.numel() for p in static_model.parameters() if p.requires_grad)
    assert temporal_parameters == sum(p.numel() for p in temporal_model.parameters() if p.requires_grad)
    assert static_parameters > 0
    assert temporal_parameters > 0


def test_pipeline_routes_static_control_to_x_mechanism_target_and_detailed_metrics(
    tmp_path, monkeypatch
):
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
    model = _model(horizon=horizon, hidden_dim=4)
    pipeline = TrainingPipeline(config, model, dataset, checkpoint_dir=tmp_path)
    observed_inputs = []
    observed_targets = []
    original_forward = model.forward
    original_compute_loss = pipeline._compute_loss

    def record_current_state(x, edge_index, edge_attr=None, batch=None):
        observed_inputs.append(tuple(x.shape))
        return original_forward(x, edge_index, edge_attr, batch)

    def record_target(predictions, target):
        observed_targets.append(target.detach().cpu())
        return original_compute_loss(predictions, target)

    monkeypatch.setattr(model, "forward", record_current_state)
    pipeline._compute_loss = record_target
    metrics = pipeline.train()

    assert observed_inputs
    assert all(len(shape) == 2 and shape[1] == 8 for shape in observed_inputs)
    assert observed_targets
    assert all(target.shape[-2:] == (horizon, 3) for target in observed_targets)
    assert all(torch.all(target == 0.25) for target in observed_targets)
    assert set(metrics) == {"loss", "mae", "rmse", "r2", "trajectory_metrics"}
    assert metrics["mae"] == pytest.approx(metrics["trajectory_metrics"]["aggregate"]["mae"])
    assert metrics["rmse"] == pytest.approx(metrics["trajectory_metrics"]["aggregate"]["rmse"])
    assert len(metrics["trajectory_metrics"]["per_horizon"]) == horizon
    assert (tmp_path / "static_mechanism_trajectory_best.pt").exists()
    assert (tmp_path / "static_mechanism_trajectory_last.pt").exists()
