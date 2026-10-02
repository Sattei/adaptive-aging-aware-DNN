"""Run causal T=1/T=2/T=4 temporal-history ablations on one cached dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.refinement_utils import HistoryWindowDataset
from experiments.static_temporal_comparison import json_safe
from experiments.temporal_refinement import run_temporal_experiment
from graph.graph_dataset import AgingDataset
from scripts.compare_static_temporal import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-lengths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", default="outputs/refinement/history_ablation")
    parser.add_argument(
        "--reference-t4",
        default="outputs/refinement/reference_512_seed42/comparison_results.json",
        help="Existing matching T=4 comparison result to reuse rather than retrain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.seed = args.seed
    cfg.dataset.size = args.dataset_size
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.learning_rate
    cfg.training.lr = args.learning_rate
    cfg.runtime.device = args.device
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    dataset = AgingDataset(str(REPO_ROOT / cfg.dataset.root), str(cfg.dataset.split), int(cfg.dataset.size), cfg=cfg, seed=int(cfg.seed))

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"experiment": {"dataset_seed": args.seed, "history_lengths": args.history_lengths, "learning_rate": args.learning_rate}, "runs": {}}
    for history_length in args.history_lengths:
        reference_path = REPO_ROOT / args.reference_t4
        if history_length == 4 and reference_path.is_file():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            reference_experiment = reference["experiment"]
            if (
                int(reference_experiment["dataset_size"]) != int(args.dataset_size)
                or int(reference_experiment["seed"]) != int(args.seed)
                or int(reference_experiment["history_length"]) != 4
                or int(reference_experiment["training"]["epochs"]) != int(args.epochs)
                or int(reference_experiment["training"]["batch_size"]) != int(args.batch_size)
                or float(reference_experiment["training"]["learning_rate"]) != float(args.learning_rate)
            ):
                raise ValueError("The supplied T=4 result does not match this ablation configuration")
            temporal = dict(reference["temporal"])
            temporal["experiment"] = dict(reference_experiment)
            temporal["source"] = {"reused_reference_result": str(reference_path)}
            results["runs"]["4"] = temporal
            print(f"Reused matching T=4 temporal result from {reference_path}")
            continue
        window = HistoryWindowDataset(dataset, history_length)
        print(f"Running temporal history length T={history_length}")
        results["runs"][str(history_length)] = run_temporal_experiment(
            cfg, window, output_dir / f"T{history_length}" / "checkpoints", seed=args.seed
        )
    path = output_dir / "history_ablation_results.json"
    path.write_text(json.dumps(json_safe(results), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved history ablation to {path}")


if __name__ == "__main__":
    main()
