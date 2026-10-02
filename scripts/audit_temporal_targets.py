"""Audit temporal mechanism-target distributions without changing any samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.target_distribution import compare_target_splits
from experiments.refinement_utils import split_indices
from graph.graph_dataset import AgingDataset
from scripts.compare_static_temporal import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs/refinement/target_audit")
    return parser.parse_args()


def collect_targets(dataset, indices):
    return torch.cat([dataset[index].y_mechanism_trajectory for index in indices], dim=0)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.seed = args.seed
    cfg.dataset.size = args.dataset_size
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root

    dataset = AgingDataset(
        root=str(REPO_ROOT / cfg.dataset.root),
        split=str(cfg.dataset.split),
        size=int(cfg.dataset.size),
        cfg=cfg,
        seed=int(cfg.seed),
    )
    memberships = split_indices(dataset, args.seed)
    target_sets = {"whole": collect_targets(dataset, range(len(dataset)))}
    target_sets.update(
        {name: collect_targets(dataset, indices) for name, indices in memberships.items()}
    )
    audit = compare_target_splits(target_sets)
    audit["dataset"] = {
        "size": len(dataset),
        "seed": int(args.seed),
        "history_length": int(dataset[0].x_history.shape[1]),
        "horizon": int(dataset[0].y_mechanism_trajectory.shape[1]),
        "split_counts": {name: len(indices) for name, indices in memberships.items()},
    }

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "target_distribution_audit.json"
    output_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved target audit to {output_path}")


if __name__ == "__main__":
    main()
