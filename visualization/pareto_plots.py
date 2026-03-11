import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from optimization.nsga2_optimizer import ParetoSolution

def plot_pareto_2d(pareto_solutions: List[ParetoSolution], obj_x: str, obj_y: str, save_path: Path) -> None:
    """
    Plots a 2D scatter of the pareto frontier for two chosen objectives.
    Valid choices: 'peak_aging', 'latency', 'energy'
    """
    x_vals = [getattr(s, obj_x) for s in pareto_solutions]
    y_vals = [getattr(s, obj_y) for s in pareto_solutions]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, c='blue', alpha=0.7, edgecolors='k')
    plt.xlabel(obj_x.replace('_', ' ').title())
    plt.ylabel(obj_y.replace('_', ' ').title())
    plt.title(f'Pareto Frontier: {obj_x} vs {obj_y}')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_3d(pareto_solutions: List[ParetoSolution], save_path: Path) -> None:
    """
    3D scatter plot: aging × latency × energy. 
    Color = aging score for intensity representation.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = [s.latency for s in pareto_solutions]
    y = [s.energy for s in pareto_solutions]
    z = [s.peak_aging for s in pareto_solutions]
    
    sc = ax.scatter(x, y, z, c=z, cmap='hot_r', marker='o', s=50, edgecolors='k')
    
    ax.set_xlabel('Latency (Cycles)')
    ax.set_ylabel('Energy (pJ)')
    ax.set_zlabel('Peak Aging')
    plt.title('3D Pareto Frontier Mappings')
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Peak Aging Score')
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
