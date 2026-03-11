import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pathlib import Path
from graph.accelerator_graph import AcceleratorGraph

def plot_aging_heatmap(
    accelerator_graph: AcceleratorGraph,
    aging_vector: np.ndarray,
    title: str,
    save_path: Path,
    cmap: str = 'hot_r',
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    2D spatial heatmap of PE array colored by aging score.
    """
    nx_graph = accelerator_graph.graph
    
    # Simple grid layout proxy for visualization since hardware is logically tiled
    pos = nx.spring_layout(nx_graph, seed=42)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    nodes = nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax, 
        node_color=aging_vector, 
        cmap=plt.get_cmap(cmap),
        node_size=300,
        vmin=vmin, vmax=vmax,
        edgecolors='black',
        linewidths=0.5
    )
    
    nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.3, arrows=False)
    
    # Custom labels based on node type
    labels = {}
    for i in range(accelerator_graph.get_num_nodes()):
        info = accelerator_graph.get_node_info(i)
        if info['type'] == 'mac':
            labels[i] = f"M{info['local_idx']}"
        elif info['type'] == 'sram':
            labels[i] = f"S{info['local_idx']}"
        else:
            labels[i] = f"R{info['local_idx']}"
            
    nx.draw_networkx_labels(nx_graph, pos, labels, ax=ax, font_size=8, font_color='white')
    
    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Aging Score', rotation=270, labelpad=20, fontsize=12)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
