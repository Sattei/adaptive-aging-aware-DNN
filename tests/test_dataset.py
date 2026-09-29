import torch
from pathlib import Path
from omegaconf import OmegaConf

from aging_models.aging_label_generator import AgingLabelGenerator
from graph.graph_dataset import AgingDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def test_aging_dataset(tmp_path):
    root = Path(tmp_path) / "temp_test_data"

    cfg = OmegaConf.create({
        'training': {
            'seq_len': 5
        }
    })

    dataset = AgingDataset(root=str(root), split="test", size=10, cfg=cfg)
    assert len(dataset) == 0
    assert f"labels_v{AgingDataset.LABEL_SCHEMA_VERSION}" in dataset.processed_paths[0]

    for _ in range(3):
        d = Data(
            x=torch.rand(10, 8),
            x_history=torch.rand(10, 4, 8),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.rand(10, 1),
            y_mechanisms=torch.rand(10, 3),
            y_trajectory=torch.rand(10, 5),
            y_mechanism_trajectory=torch.rand(10, 5, 3),
        )
        dataset.add_sample(d)

    dataset.finalize_and_save()

    dataset2 = AgingDataset(root=str(root), split="test", size=10, cfg=cfg)
    assert len(dataset2) == 3

    sample = dataset2[0]
    assert sample.x.shape == (10, 8)
    assert sample.x_history.shape == (10, 4, 8)
    assert sample.y.shape == (10, 1)
    assert sample.y_mechanisms.shape == (10, 3)
    assert sample.y_trajectory.shape == (10, 5)
    assert sample.y_mechanism_trajectory.shape == (10, 5, 3)

    loader = DataLoader(dataset2, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert batch.y.shape == (20, 1)
    assert batch.y_mechanisms.shape == (20, 3)
    assert batch.y_trajectory.shape == (20, 5)
    assert batch.x_history.shape == (20, 4, 8)
    assert batch.y_mechanism_trajectory.shape == (20, 5, 3)


def _temporal_cfg():
    return OmegaConf.create({
        "accelerator": {
            "pe_array": [2, 2],
            "pe_array_rows": 2,
            "pe_array_cols": 2,
            "num_pes": 4,
            "mac_clusters": 4,
            "sram_banks": 2,
            "noc_routers": 2,
            "freq_mhz": 1000.0,
            "voltage_v": 0.8,
        },
        "workloads": [],
        "model": {"prediction_horizon": 3},
    })


def test_generated_dataset_uses_causal_simulator_sequence(tmp_path):
    cfg = _temporal_cfg()
    dataset = AgingDataset(
        root=str(Path(tmp_path) / "persistent_data"),
        split="train",
        size=1,
        cfg=cfg,
        seed=7,
    )
    sample = dataset[0]

    history = AgingDataset.HISTORY_LENGTH
    horizon = 3
    assert sample.x.shape == (8, 8)
    assert sample.x_history.shape == (8, history, 8)
    assert torch.allclose(sample.x, sample.x_history[:, -1, :])
    assert sample.y.shape == (8, 1)
    assert sample.y_mechanisms.shape == (8, 3)
    assert sample.y_trajectory.shape == (8, horizon)
    assert sample.y_mechanism_trajectory.shape == (8, horizon, 3)
    assert torch.all((sample.y_mechanism_trajectory >= 0.0) & (sample.y_mechanism_trajectory <= 1.0))
    assert torch.all(sample.y_mechanism_trajectory[:, 1:] >= sample.y_mechanism_trajectory[:, :-1])
    assert torch.all(sample.y_trajectory[:, 1:] >= sample.y_trajectory[:, :-1])
    assert torch.all(sample.y_trajectory[:, 0] >= sample.y[:, 0])

    generator = AgingLabelGenerator(cfg=dataset.cfg)
    expected_composite = torch.stack([
        torch.tensor(generator.combine_mechanisms(sample.y_mechanism_trajectory[:, h, :].numpy()))
        for h in range(horizon)
    ], dim=1).to(torch.float32)
    assert torch.allclose(sample.y_trajectory, expected_composite, atol=1e-6)

    ambient = float(dataset.cfg.accelerator.ambient_temperature_k)
    max_temp = float(dataset.cfg.accelerator.max_temperature_k)
    expected_temperature_feature = torch.clamp(
        (sample.temperature_history_k - ambient) / (max_temp - ambient),
        0.0,
        1.0,
    )
    assert torch.allclose(sample.x_history[:, :, 4], expected_temperature_feature, atol=1e-6)
    assert torch.all(sample.x_history[:, 1:, 7] >= sample.x_history[:, :-1, 7])


def test_temporal_dataset_is_deterministic_and_batches_node_aligned_time(tmp_path):
    cfg = _temporal_cfg()
    first = AgingDataset(str(Path(tmp_path) / "first"), "train", 2, cfg=cfg, seed=11)
    second = AgingDataset(str(Path(tmp_path) / "second"), "train", 2, cfg=cfg, seed=11)

    for name in ("x", "x_history", "y", "y_mechanisms", "y_trajectory", "y_mechanism_trajectory"):
        assert torch.allclose(getattr(first[0], name), getattr(second[0], name))

    batch = next(iter(DataLoader(first, batch_size=2, shuffle=False)))
    n = first[0].num_nodes
    assert batch.x_history.shape == (2 * n, AgingDataset.HISTORY_LENGTH, 8)
    assert batch.y_mechanism_trajectory.shape == (2 * n, 3, 3)
