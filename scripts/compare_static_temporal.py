"""Run the controlled static-versus-temporal mechanism trajectory experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from experiments.static_temporal_comparison import (
    run_static_temporal_comparison,
    save_comparison_results,
)
from graph.graph_dataset import AgingDataset


def load_config():
    return OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs/experiments.yaml"),
        OmegaConf.load(REPO_ROOT / "configs/accelerator.yaml"),
        OmegaConf.load(REPO_ROOT / "configs/workloads.yaml"),
        OmegaConf.load(REPO_ROOT / "configs/training.yaml"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.dataset_size is not None:
        cfg.dataset.size = args.dataset_size
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.device is not None:
        cfg.runtime.device = args.device

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / cfg.output_dir / "static_temporal_comparison"
    dataset = AgingDataset(
        root=str(REPO_ROOT / cfg.dataset.root),
        split=str(cfg.dataset.split),
        size=int(cfg.dataset.size),
        cfg=cfg,
        seed=int(cfg.seed),
    )
    results = run_static_temporal_comparison(
        cfg,
        dataset,
        checkpoint_dir=output_dir / "checkpoints",
        seed=int(cfg.seed),
    )
    result_path = output_dir / "comparison_results.json"
    save_comparison_results(results, result_path)

    comparison = results["comparison"]
    print(f"Saved comparison to {result_path}")
    print(f"MAE delta (temporal - static): {comparison['mae_delta']:.6f}")
    print(f"RMSE delta (temporal - static): {comparison['rmse_delta']:.6f}")


if __name__ == "__main__":
    main()
