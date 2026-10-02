"""Measure temporal-model seed stability with a fixed dataset and split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static_temporal_comparison import json_safe
from experiments.refinement_utils import HistoryWindowDataset
from experiments.temporal_refinement import aggregate_multiseed_results, run_temporal_experiment
from graph.graph_dataset import AgingDataset
from scripts.compare_static_temporal import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", default="outputs/refinement/multiseed")
    parser.add_argument(
        "--reference-seed42",
        default="outputs/refinement/learning_rate/learning_rate_results.json",
        help="Existing matching T=4, 1e-3 seed-42 run to reuse.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.seed = args.dataset_seed
    cfg.dataset.size = args.dataset_size
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.learning_rate
    cfg.training.lr = args.learning_rate
    cfg.runtime.device = args.device
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    dataset = AgingDataset(str(REPO_ROOT / cfg.dataset.root), str(cfg.dataset.split), int(cfg.dataset.size), cfg=cfg, seed=int(args.dataset_seed))
    dataset = HistoryWindowDataset(dataset, args.history_length)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in args.seeds:
        reference_path = REPO_ROOT / args.reference_seed42
        if seed == 42 and reference_path.is_file():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            temporal = dict(reference["runs"]["lr_0.001"])
            metadata = temporal["experiment"]
            if (
                int(metadata["seed"]) != 42
                or int(metadata["split_seed"]) != int(args.split_seed)
                or int(metadata["dataset_size"]) != int(args.dataset_size)
                or int(metadata["history_length"]) != int(args.history_length)
                or float(metadata["training"]["learning_rate"]) != float(args.learning_rate)
            ):
                raise ValueError("The supplied seed-42 result does not match this multi-seed configuration")
            temporal["source"] = {"reused_reference_result": str(reference_path)}
            runs["42"] = temporal
            print(f"Reused matching seed-42 temporal result from {reference_path}")
            continue
        print(f"Running temporal initialization seed {seed} with fixed split seed {args.split_seed}")
        runs[str(seed)] = run_temporal_experiment(
            cfg, dataset, output_dir / f"seed_{seed}" / "checkpoints", seed=seed, split_seed=args.split_seed
        )
    results = {
        "dataset_seed": args.dataset_seed,
        "split_seed": args.split_seed,
        "initialization_seeds": args.seeds,
        "history_length": args.history_length,
        "runs": runs,
        "aggregate": aggregate_multiseed_results(runs),
    }
    path = output_dir / "multiseed_results.json"
    path.write_text(json.dumps(json_safe(results), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved multi-seed results to {path}")


if __name__ == "__main__":
    main()
