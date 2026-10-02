"""Evaluate frozen temporal history perturbations and a train-mean baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.trajectory_metrics import compute_mechanism_trajectory_metrics
from experiments.refinement_utils import split_indices
from experiments.static_temporal_comparison import build_temporal_model, json_safe
from graph.graph_dataset import AgingDataset
from scripts.compare_static_temporal import load_config
from utils.device import dataloader_kwargs, resolve_device


class HistoryVariantDataset(Dataset):
    """Return deterministic history perturbations without modifying source data."""

    def __init__(self, dataset, variant: str):
        self.dataset = dataset
        self.variant = variant
        if variant not in {"correct", "reversed", "repeat_latest", "shuffle_earlier"}:
            raise ValueError(f"Unknown history variant: {variant}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index].clone()
        history = sample.x_history
        if self.variant == "reversed":
            sample.x_history = torch.flip(history, dims=(1,))
        elif self.variant == "repeat_latest":
            sample.x_history = history[:, -1:, :].expand_as(history).clone()
        elif self.variant == "shuffle_earlier" and history.size(1) > 2:
            generator = torch.Generator().manual_seed(int(index))
            order = torch.randperm(history.size(1) - 1, generator=generator)
            sample.x_history = torch.cat((history[:, order, :], history[:, -1:, :]), dim=1)
        return sample


def evaluate(model, dataset, indices, device, batch_size):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, **dataloader_kwargs(device))
    predictions, targets = [], []
    weighted_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            prediction = model(batch.x_history, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(prediction, batch.y_mechanism_trajectory)
            weighted_loss += loss.item() * batch.num_graphs
            predictions.append(prediction.cpu().numpy())
            targets.append(batch.y_mechanism_trajectory.cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    metrics = compute_mechanism_trajectory_metrics(pred, target)
    return {
        "loss": float(weighted_loss / len(loader.dataset)),
        "mae": metrics["aggregate"]["mae"],
        "rmse": metrics["aggregate"]["rmse"],
        "r2": metrics["aggregate"]["r2"],
        "trajectory_metrics": metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default="outputs/refinement/learning_rate/lr_0.001/checkpoints/mechanism_trajectory_best.pt")
    parser.add_argument("--output-dir", default="outputs/refinement/history_sensitivity")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    cfg.seed = args.seed
    cfg.dataset.size = args.dataset_size
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = 1e-3
    cfg.training.lr = 1e-3
    cfg.runtime.device = args.device
    dataset = AgingDataset(str(REPO_ROOT / cfg.dataset.root), str(cfg.dataset.split), int(cfg.dataset.size), cfg=cfg, seed=args.seed)
    memberships = split_indices(dataset, args.seed)
    device = resolve_device(args.device)
    model = build_temporal_model(cfg, int(dataset[0].x.shape[-1])).to(device)
    model.load_state_dict(torch.load(REPO_ROOT / args.checkpoint, map_location=device))

    variants = {
        name: evaluate(model, HistoryVariantDataset(dataset, name), memberships["test"], device, args.batch_size)
        for name in ("correct", "reversed", "repeat_latest", "shuffle_earlier")
    }
    train_targets = torch.cat([dataset[index].y_mechanism_trajectory for index in memberships["train"]], dim=0).numpy()
    test_targets = torch.cat([dataset[index].y_mechanism_trajectory for index in memberships["test"]], dim=0).numpy()
    mean_prediction = np.broadcast_to(np.mean(train_targets, axis=0, keepdims=True), test_targets.shape)
    baseline_metrics = compute_mechanism_trajectory_metrics(mean_prediction, test_targets)
    result = {
        "history_variants": variants,
        "train_mean_baseline": {
            "mae": baseline_metrics["aggregate"]["mae"],
            "rmse": baseline_metrics["aggregate"]["rmse"],
            "r2": baseline_metrics["aggregate"]["r2"],
            "trajectory_metrics": baseline_metrics,
        },
    }
    output = REPO_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    path = output / "diagnostics.json"
    path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved refinement diagnostics to {path}")


if __name__ == "__main__":
    main()
