import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def render_architecture_block_diagram(output_path: str):
    """
    Renders a block diagram of the proposed paradigm.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simple blocks representation (to be customized)
    ax.add_patch(patches.Rectangle((0.1, 0.4), 0.2, 0.2, fill=True, color='lightblue', lw=2, ec='black'))
    ax.text(0.2, 0.5, "Simulator\n& Workload\nMonitor", ha='center', va='center')
    
    ax.add_patch(patches.Rectangle((0.4, 0.4), 0.2, 0.2, fill=True, color='lightgreen', lw=2, ec='black'))
    ax.text(0.5, 0.5, "GNN-Trans.\nTrajectory\nPredictor", ha='center', va='center')
    
    ax.add_patch(patches.Rectangle((0.7, 0.4), 0.2, 0.2, fill=True, color='salmon', lw=2, ec='black'))
    ax.text(0.8, 0.5, "RL Controller\n& NSGA-II\nOptimizer", ha='center', va='center')
    
    # Arrows
    ax.annotate('', xy=(0.4, 0.5), xytext=(0.3, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(0.7, 0.5), xytext=(0.6, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Feedback
    ax.annotate('', xy=(0.2, 0.4), xytext=(0.8, 0.4), arrowprops=dict(facecolor='black', shrink=0.05, connectionstyle="arc3,rad=0.3"))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title("System Architecture: Predictive Lifetime Management", fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if output_path.endswith('.pdf'):
        plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
        
    plt.close()
