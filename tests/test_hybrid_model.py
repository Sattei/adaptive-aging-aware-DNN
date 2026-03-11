import pytest
import torch
from omegaconf import OmegaConf

from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.trajectory_predictor import TrajectoryPredictor
from graph.graph_dataset import AgingDataset
from torch_geometric.data import Data

def test_hybrid_gnn_transformer():
    model = HybridGNNTransformer(
        node_feature_dim=21, 
        hidden_dim=64, 
        seq_len=1 # Test single instantaneous mode
    )
    
    # Create dummy graph with 16 nodes and random connectivity
    x = torch.rand((16, 21))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long) # ring path
    batch = torch.zeros(16, dtype=torch.long)
    
    out = model(x, edge_index, batch=batch)
    
    assert out.shape == (16, 1), "Should output a single continuous label per node"

def test_trajectory_predictor():
    base = HybridGNNTransformer(node_feature_dim=10, hidden_dim=32, seq_len=3)
    model = TrajectoryPredictor(gnn_encoder=base, horizon=5)
    
    # Dummy batched sequenced graph data 
    # seq=3, N=8 nodes
    total_nodes = 3 * 8
    x = torch.rand((total_nodes, 10))
    # Full mesh intra-graph dummy (won't be physically accurate but works for shapes)
    edge_index = torch.randint(0, total_nodes, (2, 40))
    
    out = model(x, edge_index)
    
    assert out.shape == (24, 5), "Should output flattened trajectories for all nodes"
    
    # Test loss function
    target = torch.rand((24, 5))
    loss = model.trajectory_loss(out, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
