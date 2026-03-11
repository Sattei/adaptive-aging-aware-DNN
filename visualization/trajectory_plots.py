import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

def plot_aging_trajectories(
    trajectories: Dict[str, np.ndarray],
    component_ids: List[int],
    time_axis: np.ndarray,
    failure_threshold: float,
    save_path: Path,
) -> None:
    """
    Line plot mapping historical aging degradation trends against Time to Failure constraints.
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    c_idx = 0
    
    for system_name, traj_data in trajectories.items():
        # Extrapolate mean and std
        # traj_data is shape [T, N]
        mean_aging = np.mean(traj_data, axis=1)
        std_aging = np.std(traj_data, axis=1)
        
        c = colors[c_idx % len(colors)]
        plt.plot(time_axis, mean_aging, label=f"{system_name} (Mean)", color=c, linewidth=2)
        plt.fill_between(time_axis, mean_aging - std_aging, mean_aging + std_aging, color=c, alpha=0.2)
        
        # Check intersection with threshold
        intersect_idx = np.where(mean_aging >= failure_threshold)[0]
        if len(intersect_idx) > 0:
            first = intersect_idx[0]
            plt.plot(time_axis[first], mean_aging[first], 'x', color='black', markersize=10)
            plt.annotate(
                f"TTF: {time_axis[first]:.1f}Y",
                (time_axis[first], mean_aging[first]),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center'
            )
            
        c_idx += 1
        
    plt.axhline(y=failure_threshold, color='r', linestyle='--', linewidth=2, label='Failure Threshold')
    
    plt.xlabel('Time (Years)', fontsize=12)
    plt.ylabel('Normalized Aging Score', fontsize=12)
    plt.title('Aging Trajectories Over Component Lifetime', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best')
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_lifetime_comparison_bar(results: Dict[str, float], save_path: Path) -> None:
    """Bar chart mapping extrapolated lifetimes of different systems."""
    plt.figure(figsize=(10, 6))
    
    systems = list(results.keys())
    lifetimes = list(results.values())
    
    # Highlight highest
    colors = ['lightcoral' if l < max(lifetimes) else 'mediumseagreen' for l in lifetimes]
    
    bars = plt.bar(systems, lifetimes, color=colors, edgecolor='black')
    
    # Baseline line proxy (assuming first is baseline)
    if len(lifetimes) > 0:
        plt.axhline(y=lifetimes[0], color='gray', linestyle='--', label='Baseline Lifetime')
        
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} Yrs', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.ylabel('Time to Failure (Years)', fontsize=12)
    plt.title('Accelerator Lifetime Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
