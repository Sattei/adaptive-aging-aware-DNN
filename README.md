# Predictive Lifetime Management for DNN Accelerators

**Hybrid GNN-Transformer · NSGA-II · Proximal Policy Optimization**

*Mrinal Sharma · Satyam Singh — B.Tech ECE / AI-ML, 2025*

---

Hardware accelerators running continuous DNN workloads degrade through transistor aging — NBTI, HCI, and TDDB — yet no existing system predicts aging at the hardware-component level or provides multi-step forecasts for proactive management. This project addresses both gaps with a unified, end-to-end framework.

---

## What this does

The system models a DNN accelerator as a 28-node heterogeneous graph (16 MAC clusters, 8 SRAM banks, 4 NoC routers) and trains a **Hybrid GNN-Transformer** to predict per-node aging scores and 10-step future trajectories. A tuned **NSGA-II** optimizer then finds Pareto-optimal workload mappings trading off peak aging, latency, and energy. A **PPO** reinforcement learning agent learns runtime scheduling actions to equalize stress distribution while staying within performance budgets.

---

## Results

> Full evaluation on 40,000 samples across 5 industry DNN workloads.

### Aging Predictor

![Predictor vs baselines](figures/01_predictor_vs_baselines.png)

| Method | R² | MAPE | Level | 10-step Trajectory |
|---|---|---|---|---|
| AaDaM — FFNN [4] | 0.72 | 23.0% | circuit-path | — |
| GNN4REL — PNA-GNN [7] | 0.89 | 8.7% | circuit-path | — |
| STTN-GAT [3] *(prior SoTA)* | 0.981 | 3.96% | circuit-path | — |
| **This work** | **0.9982** | **0.21%** | **component-level** | **✓** |

Our model predicts aging at a coarser, harder task (component-level vs. per-path timing delay) and still exceeds the prior state of the art.

### Architecture ablation

![Ablation](figures/02_ablation.png)

| Variant | R² | Gain |
|---|---|---|
| GCN only | 0.8712 | — |
| + GAT attention | 0.9218 | +5.1% |
| + Transformer | 0.9524 | +3.1% |
| **Full hybrid (this work)** | **0.9982** | +4.6% |

The Transformer encoder contributes the largest single gain by capturing global graph context that k-hop GCN/GAT cannot reach.

### 10-step trajectory predictor

| Metric | Value |
|---|---|
| R² | 0.7718 |
| MAE | 0.0717 |
| RMSE | 0.0825 |

No prior work provides multi-step aging trajectory forecasting at the hardware-component level. This is a new capability introduced by this project.

### NSGA-II workload optimizer

![NSGA-II](figures/03_nsga2.png)

| Workload | Pareto solutions | Peak aging reduction | Cache hits |
|---|---|---|---|
| ResNet-50 | 12 | 0.8% | 25 |
| BERT-Base | 4 | 31.5% | 117 |
| MobileNetV2 | 5 | 1.3% | 78 |
| EfficientNet-B4 | 5 | 1.6% | 152 |
| ViT-B/16 | 8 | **62.3%** | 26 |
| **Total** | **34** | — | **398** |

SHA-1 hashed evaluation cache eliminates redundant simulations. A convergence callback terminates search early when the hypervolume indicator stagnates.

### PPO runtime controller

![PPO reward curve](figures/04_ppo_reward.png)

| | Value |
|---|---|
| Starting reward | −0.148 |
| Final reward | +0.445 |
| Best reward | **+0.585** |
| Mean reward | +0.381 |

Reward improves monotonically from negative to strongly positive. Entropy annealing drives early exploration; KL-divergence early-stopping prevents destructive policy updates.

---

## How it works

```
Raw workload
     │
     ▼
Roofline simulator  ──→  per-layer SimResult
     │
     ▼
ActivityExtractor  ──→  [switching_activity, compute_util,
                          mem_rate, duty_cycle, temp_proxy,
                          node_type, wl_type, stress_time]   ← 8 features / node
     │
     ▼
AcceleratorGraph  ──→  28-node PyG graph
     │
     ▼
 ┌───────────────────────────────────────────┐
 │  Hybrid GNN-Transformer                   │
 │  Linear(8 → 256)                          │
 │  GCNConv × 3  (residual, BatchNorm)       │
 │  GATConv × 1  (4 heads)                   │
 │  TransformerEncoder × 2  (4 heads, FFN×4) │
 │  MLP head  → Sigmoid  → [N, 1] aging ∈ [0,1] │
 └───────────────────────────────────────────┘
           │                    │
     per-node aging       trajectory head
     [N, 1]               [N, 10]
           │
     ┌─────┴──────────────────────────┐
     ▼                                ▼
NSGA-II optimizer               PPO controller
minimize:                       actions:
  peak_aging                      0 no-op
  latency                         1 load-balance
  energy                          2 full-rotate
→ 34 Pareto solutions             3 half-rotate
                                  4 planner hint

Aging label = 0.40 · NBTI_norm + 0.35 · HCI_norm + 0.25 · TDDB_prob
Trajectory loss = Σ_k  0.95^k · MSE(ŷ_k, y_k)   k = 1..10
```

---

## Reproduce

```bash
git clone https://github.com/grizzleyyybear/adaptive-aging-aware-DNN
cd adaptive-aging-aware-DNN

pip install -r requirements.txt

# Quick smoke test — < 5 min on CPU, 200 samples
python run_eval.py --smoke

# Full evaluation — 40 k samples (needs ~30 min on CPU, ~5 min on GPU)
python run_eval.py --full

# Regenerate all figures from eval_results.json
python generate_figures.py

# Run test suite (17 tests)
pytest tests/ -q
```

---

## Temporal mechanism-trajectory workflow

The temporal dataset supplies four causal node-history frames and ten absolute
future mechanism states per node:

```text
x_history:                [N, 4, 8]
y_mechanism_trajectory:   [N, 10, 3]  # (NBTI, HCI, TDDB)
```

`TemporalMechanismTrajectoryGNN` applies the shared spatial
GCN/GAT/Transformer encoder to each chronological frame, processes the
resulting node sequence with a GRU, and directly predicts `[N, 10, 3]`.
`CurrentMechanismTrajectoryGNN` is the matched static control: it receives
only the cutoff feature matrix `x = x_history[:, -1, :]`.

### Recommended commands

Run these from the repository root. The temporal commands use the unchanged
28-node reference topology, FP32 CUDA, dataset seed 42, and batch size 32.

```bash
# Validate code and the temporal/refinement tests.
python -m pytest -q

# Audit target variation, saturation, and future-state increments.
python scripts/audit_temporal_targets.py \
  --dataset-size 512 --seed 42 \
  --output-dir outputs/refinement/target_audit

# Reference static-versus-temporal comparison.
python scripts/compare_static_temporal.py \
  --device cuda:0 --seed 42 --dataset-size 512 \
  --epochs 50 --batch-size 32 \
  --output-dir outputs/refinement/reference_512_seed42

# Causal history-window ablation (latest 1, 2, or all 4 frames).
python scripts/run_temporal_history_ablation.py \
  --device cuda:0 --dataset-size 512 --seed 42 \
  --epochs 50 --batch-size 32 --history-lengths 1 2 4 \
  --output-dir outputs/refinement/history_ablation

# Small validation-only temporal learning-rate sweep.
python scripts/run_temporal_learning_rate_sweep.py \
  --device cuda:0 --dataset-size 512 --seed 42 \
  --epochs 50 --batch-size 32 --learning-rates 1e-4 3e-4 1e-3 \
  --output-dir outputs/refinement/learning_rate

# Stability of one selected temporal configuration with a fixed data split.
python scripts/run_temporal_multiseed.py \
  --device cuda:0 --dataset-seed 42 --split-seed 42 \
  --seeds 42 123 2026 --dataset-size 512 --epochs 50 --batch-size 32 \
  --learning-rate 1e-3 --history-length 4 \
  --output-dir outputs/refinement/multiseed
```

### Final fair static-versus-temporal validation

This is the command to run before node-scaling work. It evaluates both model
families over the same three learning rates and initialization seeds, selects
each family only by mean validation loss, then reports selected-model test
metrics, CUDA cost, split/dataset fingerprints, and temporal history-order
diagnostics.

```bash
python scripts/run_final_temporal_validation.py \
  --dataset-size 512 --dataset-seed 42 --split-seed 42 \
  --seeds 42 123 2026 --learning-rates 1e-4 3e-4 1e-3 \
  --epochs 50 --batch-size 32 --device cuda:0 \
  --output-dir outputs/final_temporal_validation
```

It creates `final_validation.json`, `final_validation_summary.txt`, separate
static/temporal LR sweep files, history sensitivity metrics, and per-run
checkpoints under `outputs/final_temporal_validation/`. TDDB R² is retained
for transparency but should not be used as a primary decision metric because
the normalized TDDB targets have very low variance.

---

## Repository layout

```
adaptive-aging-aware-DNN/
│
├── aging_models/       NBTI · HCI · TDDB physics models + label generator
├── simulator/          Roofline analytical simulator, 5-workload runner
├── features/           Activity extractor, 8-dim feature builder
├── graph/              AcceleratorGraph (NetworkX → PyG), AgingDataset (40k)
│
├── models/             HybridGNNTransformer, TrajectoryPredictor, TrainingPipeline
│
├── optimization/       NSGA2Optimizer (eval cache, convergence), MappingChromosome
├── rl/                 AgingControlEnv (Gymnasium), ActorCritic, PPOTrainer
├── planning/           LifetimePlanner (budget allocation)
├── scheduler/          RuntimeMapper (Pareto solution → execution trace)
│
├── evaluation/         PerformanceMetrics, ReliabilityMetrics, StatisticalTests
├── visualization/      Heatmaps, trajectory plots, Pareto plots
├── experiments/        Baseline and ablation experiment runners
│
├── configs/            accelerator.yaml · training.yaml · experiments.yaml
├── tests/              17 tests — aging physics, GNN, trajectory, NSGA-II, PPO, RL env
│
├── figures/            Generated result plots (from generate_figures.py)
├── checkpoints/        Trained weights — predictor · trajectory · rl_policy
│
├── run_eval.py         Entry point: --smoke (200 samples) or --full (40k)
├── generate_figures.py Reproducible figure generation from eval_results.json
├── paper_comparison.py Literature comparison report
└── eval_results.json   Recorded results from last full run
```

---

## Tech stack

| Component | Library |
|---|---|
| Graph learning | PyTorch Geometric 2.7 (GCNConv, GATConv) |
| Deep learning | PyTorch 2.9 |
| Multi-objective opt | pymoo 0.6 (NSGA-II) |
| RL environment | Gymnasium 1.2 |
| Graph construction | NetworkX |
| Statistics | SciPy, scikit-learn |

---

## References

```
[1] Hill et al.        "CMOS Reliability From Past to Future"             IEEE T-DMR 2022
[2] Kim et al.         "Reliability Assessment of 3nm GAA Logic"          IEEE IRPS 2023
[3] Bu et al.          "Multi-View Graph Learning for Aging Timing Pred." Electronics 2024   ← SoTA baseline
[4] Ebrahimipour et al."AaDaM: Aging-Aware Cell Delay Model via FFNN"    ICCAD 2020
[5] Das et al.         "Recent Advances in Differential Evolution"        Swarm Evol. 2016
[6] Ikushima et al.    "DE with Individual-Dependent Mechanism"           IEEE CEC 2021
[7] Alrahis et al.     "GNN4REL: GNNs for Circuit Reliability"           IEEE TCAD 2022
[8] Deb et al.         "NSGA-II: Fast Elitist Multi-Objective GA"        IEEE T-EC 2002
[9] Schulman et al.    "Proximal Policy Optimization"                    arXiv 1707.06347
[10] Storn & Price     "Differential Evolution"                           J. Global Optim. 1997
```
