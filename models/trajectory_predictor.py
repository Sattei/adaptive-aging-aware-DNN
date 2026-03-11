import torch
import torch.nn as nn
from torch import Tensor
import math
from models.hybrid_gnn_transformer import HybridGNNTransformer

class TrajectoryPredictor(nn.Module):
    """
    Extends the base hybrid predictor to forecast a temporal horizon.
    """
    def __init__(
        self,
        gnn_encoder: HybridGNNTransformer,
        hidden_dim: int = 256,
        horizon: int = 10,
        gamma: float = 0.95,
    ):
        super().__init__()
        self.encoder = gnn_encoder
        self.horizon = horizon
        self.gamma = gamma
        
        # Replaces the final 1D regression head with a k-dimensional sequence head
        # We reuse the GNN weights but build a new head for trajectories.
        self.traj_head = nn.Sequential(
            nn.Linear(getattr(self.encoder, 'hidden_dim', hidden_dim), 128),
            nn.ReLU(),
            nn.Linear(128, horizon)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.encoder.encode_graph(x, edge_index, edge_attr)
        trajectory = self.traj_head(h)  # [N, horizon]
        return trajectory

    def trajectory_loss(self, pred, target):
        """
        Discounted MSE: L = sum_j gamma^(j+1) * MSE(pred[:,j], target[:,j])
        pred, target: [N, horizon]
        """
        k = pred.shape[1]
        weights = torch.tensor(
            [self.gamma ** (j + 1) for j in range(k)],
            dtype=pred.dtype, device=pred.device
        )
        per_step = ((pred - target) ** 2).mean(dim=0)  # [k]
        return (weights * per_step).sum()
