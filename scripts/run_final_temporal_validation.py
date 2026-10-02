"""Run the final fair, validation-selected static-versus-temporal experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.final_temporal_validation import (
    build_summary,
    comparison_deltas,
    run_fair_sweep,
)
from experiments.refinement_utils import split_indices
from experiments.static_temporal_comparison import build_temporal_model, json_safe
from graph.graph_dataset import AgingDataset
from scripts.compare_static_temporal import load_config
from scripts.run_refinement_diagnostics import HistoryVariantDataset, evaluate
from utils.device import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/final_temporal_validation")
    return parser.parse_args()


def history_sensitivity(config, dataset, temporal_sweep, split_seed, device_request, batch_size):
    selected_lr = temporal_sweep["selected_lr"]
    seed42_run = temporal_sweep["lr_results"][selected_lr]["runs"]["42"]
    device = resolve_device(device_request)
    model = build_temporal_model(config, int(dataset[0].x.shape[-1])).to(device)
    model.load_state_dict(torch.load(seed42_run["checkpoints"]["best"], map_location=device))
    indices = split_indices(dataset, split_seed)["test"]
    return {
        name: evaluate(model, HistoryVariantDataset(dataset, name), indices, device, batch_size)
        for name in ("correct", "reversed", "repeat_latest", "shuffle_earlier")
    }


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.seed = args.dataset_seed
    cfg.dataset.size = args.dataset_size
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.runtime.device = args.device
    dataset = AgingDataset(str(REPO_ROOT / cfg.dataset.root), str(cfg.dataset.split), int(cfg.dataset.size), cfg=cfg, seed=args.dataset_seed)
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    static = run_fair_sweep(cfg, dataset, output_dir, learning_rates=args.learning_rates, seeds=args.seeds, split_seed=args.split_seed, family="static")
    temporal = run_fair_sweep(cfg, dataset, output_dir, learning_rates=args.learning_rates, seeds=args.seeds, split_seed=args.split_seed, family="temporal")
    if static["dataset_fingerprint"] != temporal["dataset_fingerprint"] or static["split_indices_sha256"] != temporal["split_indices_sha256"]:
        raise RuntimeError("Static and temporal sweeps failed final dataset/split fairness checks")

    selected_static = static["lr_results"][static["selected_lr"]]["aggregate"]
    selected_temporal = temporal["lr_results"][temporal["selected_lr"]]["aggregate"]
    result = {
        "dataset": {
            "dataset_size": len(dataset), "dataset_seed": args.dataset_seed, "split_seed": args.split_seed,
            "nodes_per_graph": int(dataset[0].x.shape[0]), "history_length": int(dataset[0].x_history.shape[1]),
            "prediction_horizon": int(dataset[0].y_mechanism_trajectory.shape[1]),
            "dataset_fingerprint": static["dataset_fingerprint"], "split_indices_sha256": static["split_indices_sha256"],
        },
        "static": static,
        "temporal": temporal,
        "history_sensitivity": history_sensitivity(cfg, dataset, temporal, args.split_seed, args.device, args.batch_size),
        "compute_cost": {
            "static": {key: selected_static.get(key) for key in ("training_seconds", "peak_cuda_memory_bytes")},
            "temporal": {key: selected_temporal.get(key) for key in ("training_seconds", "peak_cuda_memory_bytes")},
        },
        "final_comparison": comparison_deltas(selected_static, selected_temporal),
    }
    (output_dir / "static_lr_results.json").write_text(json.dumps(json_safe(static), indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "temporal_lr_results.json").write_text(json.dumps(json_safe(temporal), indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "history_sensitivity.json").write_text(json.dumps(json_safe(result["history_sensitivity"]), indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "final_validation.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "final_validation_summary.txt").write_text(build_summary(json_safe(result)), encoding="utf-8")
    print(f"Saved final validation to {output_dir}")


if __name__ == "__main__":
    main()
