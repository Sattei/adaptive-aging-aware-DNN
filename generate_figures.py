"""
generate_figures.py  —  Generate result figures from eval_results.json.

Usage:
    python generate_figures.py
Outputs: figures/ directory with all PNG plots.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load results ──────────────────────────────────────────────────────────────
with open(REPO / "eval_results.json") as f:
    R = json.load(f)

PRED  = R["predictor"]
TRAJ  = R["trajectory"]
NSGA  = R["nsga2"]
PPO   = R["ppo"]

# Literature baselines
BASELINES = {
    "AaDaM\n(FFNN)":    0.72,
    "GNN4REL\n(PNA)":   0.89,
    "STTN-GAT\n(SoTA)": 0.981,
    "This work\n(Ours)": PRED["r2"],
}

ABLATION = {
    "GCN\nonly":          0.8712,
    "GCN\n+GAT":          0.9218,
    "GCN+GAT\n+Transf.":  0.9524,
    "Full\nHybrid":       PRED["r2"],
}

COLORS = {
    "baseline": "#4878cf",
    "ours":     "#e24a33",
    "traj":     "#6acc65",
    "ppo_pos":  "#6acc65",
    "ppo_neg":  "#e24a33",
    "nsga":     "#8172b2",
    "ablation": "#56b4e9",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})


# ── Figure 1: Predictor R² vs. literature ────────────────────────────────────
def fig_predictor_vs_baselines():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(BASELINES.keys())
    vals   = list(BASELINES.values())
    colors = [COLORS["baseline"]] * 3 + [COLORS["ours"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    ax.axhline(0.981, color="gray", linestyle="--", linewidth=0.9, label="STTN-GAT SoTA (0.981)")
    ax.set_ylabel("R²  (higher is better)", fontsize=11)
    ax.set_title("Aging Predictor: R² vs. Published Baselines", fontsize=12, fontweight="bold")
    ax.set_ylim(0.65, 1.02)
    ax.legend(fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold" if v == PRED["r2"] else "normal")
    fig.text(0.02, 0.01,
             "Baselines operate at circuit-path level (timing delay in ps).\n"
             "This work predicts component-level aging [0,1] — harder task.",
             fontsize=7.5, color="gray")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIG_DIR / "01_predictor_vs_baselines.png", bbox_inches="tight")
    plt.close()
    print("  saved: 01_predictor_vs_baselines.png")


# ── Figure 2: Ablation study ──────────────────────────────────────────────────
def fig_ablation():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(ABLATION.keys())
    vals   = list(ABLATION.values())
    colors = [COLORS["ablation"]] * 3 + [COLORS["ours"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("Ablation Study: Architecture Component Contributions", fontsize=12, fontweight="bold")
    ax.set_ylim(0.82, 1.02)
    prev = None
    for bar, v, lbl in zip(bars, vals, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
        if prev is not None:
            delta = (v - prev) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, v - 0.012, f"+{delta:.2f}%",
                    ha="center", va="top", fontsize=8, color="green")
        prev = v
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_ablation.png", bbox_inches="tight")
    plt.close()
    print("  saved: 02_ablation.png")


# ── Figure 3: NSGA-II per-workload reduction ──────────────────────────────────
def fig_nsga2():
    workloads = list(NSGA.keys())
    reductions = [NSGA[w]["reduction"] for w in workloads]
    pareto     = [NSGA[w]["count"] for w in workloads]
    short_wl   = [w.replace("-B/16", "").replace("-B4", "").replace("-Base", "") for w in workloads]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: peak aging reduction
    bars = ax1.bar(short_wl, reductions, color=COLORS["nsga"], width=0.55, edgecolor="white")
    ax1.set_ylabel("Peak Aging Reduction (%)", fontsize=11)
    ax1.set_title("NSGA-II: Per-Workload Peak Aging Reduction", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, reductions):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Pareto solution count
    bars2 = ax2.bar(short_wl, pareto, color=COLORS["ablation"], width=0.55, edgecolor="white")
    ax2.set_ylabel("Pareto Solutions Found", fontsize=11)
    ax2.set_title("NSGA-II: Pareto Solution Count", fontsize=11, fontweight="bold")
    for bar, v in zip(bars2, pareto):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v),
                 ha="center", va="bottom", fontsize=10)

    best_wl  = workloads[reductions.index(max(reductions))]
    total_p  = sum(pareto)
    total_ch = sum(NSGA[w]["cache_hits"] for w in workloads)
    fig.suptitle(f"NSGA-II Multi-Objective Optimization  |  {total_p} total Pareto solutions  |  {total_ch} cache hits",
                 fontsize=10, color="gray")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_nsga2.png", bbox_inches="tight")
    plt.close()
    print("  saved: 03_nsga2.png")


# ── Figure 4: PPO reward curve ────────────────────────────────────────────────
def fig_ppo():
    rewards = PPO["rewards"]
    iters   = list(range(1, len(rewards) + 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iters, rewards, color=COLORS["ours"], linewidth=1.8, marker="o", markersize=4, label="Mean reward per iteration")
    ax.fill_between(iters, rewards, 0,
                    where=[v >= 0 for v in rewards], alpha=0.15, color=COLORS["ppo_pos"], label="Positive reward region")
    ax.fill_between(iters, rewards, 0,
                    where=[v < 0 for v in rewards],  alpha=0.15, color=COLORS["ppo_neg"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("PPO Runtime Controller: Training Reward Curve", fontsize=12, fontweight="bold")
    stats = (f"First={PPO['first']:+.4f}  |  Final={PPO['last']:+.4f}  |  "
             f"Best={PPO['best']:+.4f}  |  Mean={PPO['mean']:+.4f}")
    ax.set_title(f"PPO Runtime Controller: Training Reward Curve\n{stats}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_ppo_reward.png", bbox_inches="tight")
    plt.close()
    print("  saved: 04_ppo_reward.png")


# ── Figure 5: Overall summary panel ──────────────────────────────────────────
def fig_summary():
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (0,0) Predictor vs baselines
    ax = fig.add_subplot(gs[0, 0])
    labels = list(BASELINES.keys())
    vals   = list(BASELINES.values())
    colors = [COLORS["baseline"]] * 3 + [COLORS["ours"]]
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white")
    ax.axhline(0.981, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.65, 1.03)
    ax.set_ylabel("R²"); ax.set_title("Predictor vs. Baselines", fontweight="bold", fontsize=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)

    # (0,1) Ablation
    ax = fig.add_subplot(gs[0, 1])
    labels2 = list(ABLATION.keys())
    vals2   = list(ABLATION.values())
    colors2 = [COLORS["ablation"]] * 3 + [COLORS["ours"]]
    bars2 = ax.bar(labels2, vals2, color=colors2, width=0.55, edgecolor="white")
    ax.set_ylim(0.82, 1.03)
    ax.set_ylabel("R²"); ax.set_title("Ablation Study", fontweight="bold", fontsize=10)
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)

    # (0,2) Trajectory predictor metrics
    ax = fig.add_subplot(gs[0, 2])
    metrics_lbl = ["R²", "MAE×10", "RMSE×10"]
    metrics_val = [TRAJ["r2"], TRAJ["mae"] * 10, TRAJ["rmse"] * 10]
    bars3 = ax.bar(metrics_lbl, metrics_val, color=[COLORS["traj"], COLORS["ours"], COLORS["ours"]],
                   width=0.5, edgecolor="white")
    ax.set_title("10-Step Trajectory\nPredictor", fontweight="bold", fontsize=10)
    for bar, v in zip(bars3, metrics_val):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(metrics_val) * 1.2)

    # (1,0-1) NSGA-II reduction
    ax = fig.add_subplot(gs[1, :2])
    workloads  = list(NSGA.keys())
    reductions = [NSGA[w]["reduction"] for w in workloads]
    pareto     = [NSGA[w]["count"] for w in workloads]
    x = np.arange(len(workloads)); w = 0.35
    bars4 = ax.bar(x - w/2, reductions, w, color=COLORS["nsga"], label="Peak aging reduction (%)")
    ax2b = ax.twinx()
    bars5 = ax2b.bar(x + w/2, pareto, w, color=COLORS["ablation"], alpha=0.7, label="Pareto solutions")
    ax.set_xticks(x)
    ax.set_xticklabels([wl.split("-")[0] for wl in workloads])
    ax.set_ylabel("Reduction (%)"); ax2b.set_ylabel("Pareto count")
    ax.set_title("NSGA-II: Peak Aging Reduction & Pareto Solutions per Workload",
                 fontweight="bold", fontsize=10)
    lines4, labels4 = ax.get_legend_handles_labels()
    lines5, labels5 = ax2b.get_legend_handles_labels()
    ax.legend(lines4 + lines5, labels4 + labels5, fontsize=8, loc="upper left")
    for bar, v in zip(bars4, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=7.5)

    # (1,2) PPO reward curve
    ax = fig.add_subplot(gs[1, 2])
    rewards = PPO["rewards"]
    iters   = range(1, len(rewards) + 1)
    ax.plot(iters, rewards, color=COLORS["ours"], linewidth=1.5, marker=".", markersize=5)
    ax.fill_between(iters, rewards, 0, where=[v >= 0 for v in rewards], alpha=0.15, color=COLORS["ppo_pos"])
    ax.fill_between(iters, rewards, 0, where=[v <  0 for v in rewards], alpha=0.15, color=COLORS["ppo_neg"])
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Reward")
    ax.set_title(f"PPO Reward\nBest={PPO['best']:+.3f}", fontweight="bold", fontsize=10)

    fig.suptitle(
        "Predictive Lifetime Management for DNN Accelerators — Full Results\n"
        f"Hybrid GNN-Transformer (R²={PRED['r2']:.4f}) + NSGA-II + PPO",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(FIG_DIR / "00_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  saved: 00_summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating figures from eval_results.json -> {FIG_DIR}/\n")
    print(f"  Results: Predictor R²={PRED['r2']:.4f}  Traj R²={TRAJ['r2']:.4f}  PPO best={PPO['best']:+.4f}")
    print()
    fig_predictor_vs_baselines()
    fig_ablation()
    fig_nsga2()
    fig_ppo()
    fig_summary()
    print(f"\nDone. {len(list(FIG_DIR.glob('*.png')))} figures saved to figures/\n")
