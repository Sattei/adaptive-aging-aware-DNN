"""Visualization functions for results and analysis."""
from .aging_heatmap import plot_aging_heatmap
from .trajectory_plots import plot_aging_trajectories, plot_lifetime_comparison_bar
from .pareto_plots import plot_pareto_2d, plot_pareto_3d
from .architecture_diagrams import render_architecture_block_diagram

__all__ = [
    'plot_aging_heatmap',
    'plot_aging_trajectories',
    'plot_lifetime_comparison_bar',
    'plot_pareto_2d',
    'plot_pareto_3d',
    'render_architecture_block_diagram',
]
