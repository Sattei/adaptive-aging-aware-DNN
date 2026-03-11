# Publication-Grade Self-Healing DNN Accelerator via Predictive Lifetime Management

A predictive lifetime management framework for DNN accelerators utilizing spatial-temporal learning (Hybrid GNN-Transformer), Multi-Objective Evolutionary Optimization (NSGA-II), and Predictive Reinforcement Learning (PPO).

## Repository Structure
- `configs/`: Hydra configurations for the accelerator, workloads, and training.
- `aging_models/`: Transistor aging physics (NBTI, HCI, TDDB).
- `models/`: ML Models including the Hybrid GNN-Transformer node predictor and future trajectory predictor.
- `rl/`: PPO environment and policy for runtime accelerator control.
- `optimization/`: NSGA-II solver for mapping generation.

## Getting Started
```bash
conda env create -f environment.yaml
conda activate aging-aware-dnn
```

## Running the Pipeline
```bash
python scripts/run_full_pipeline.py --config-name=experiments
```
