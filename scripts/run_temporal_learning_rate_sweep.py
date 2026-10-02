"""Run a small validation-only learning-rate refinement for the temporal model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", default="outputs/refinement/learning_rate")
    parser.add_argument(
        "--reference-baseline",
        default="outputs/refinement/reference_512_seed42/comparison_results.json",
        help="Existing matching 1e-4 result to reuse rather than retrain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.seed = args.seed
    cfg.dataset.size = args.dataset_size
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.runtime.device = args.device
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    dataset = AgingDataset(str(REPO_ROOT / cfg.dataset.root), str(cfg.dataset.split), int(cfg.dataset.size), cfg=cfg, seed=int(cfg.seed))

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"selection_policy": "validation monitor only; test metrics are not used to select learning rate", "runs": {}}
    for learning_rate in args.learning_rates:
        reference_path = REPO_ROOT / args.reference_baseline
        if learning_rate == 1e-4 and reference_path.is_file():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            reference_experiment = reference["experiment"]
            if (
                int(reference_experiment["dataset_size"]) != int(args.dataset_size)
                or int(reference_experiment["seed"]) != int(args.seed)
                or int(reference_experiment["training"]["epochs"]) != int(args.epochs)
                or int(reference_experiment["training"]["batch_size"]) != int(args.batch_size)
                or float(reference_experiment["training"]["learning_rate"]) != learning_rate
            ):
                raise ValueError("The supplied 1e-4 result does not match this sweep configuration")
            temporal = dict(reference["temporal"])
            temporal["experiment"] = dict(reference_experiment)
            temporal["source"] = {"reused_reference_result": str(reference_path)}
            results["runs"]["lr_0.0001"] = temporal
            print(f"Reused matching 1e-4 temporal result from {reference_path}")
            continue
        cfg.training.learning_rate = learning_rate
        cfg.training.lr = learning_rate
        label = f"lr_{learning_rate:g}"
        print(f"Running temporal learning rate {learning_rate:g}")
        results["runs"][label] = run_temporal_experiment(
            cfg, dataset, output_dir / label / "checkpoints", seed=args.seed
        )
    path = output_dir / "learning_rate_results.json"
    path.write_text(json.dumps(json_safe(results), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved learning-rate sweep to {path}")


if __name__ == "__main__":
    main()
