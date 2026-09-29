"""Current-state control model for mechanism trajectory forecasting."""

from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from aging_models.aging_label_generator import AgingLabelGenerator
from models.hybrid_gnn_transformer import HybridGNNTransformer


class CurrentMechanismTrajectoryGNN(nn.Module):
    """Predict future mechanism trajectories from the cutoff graph only.

    This is the non-temporal control for ``TemporalMechanismTrajectoryGNN``:
    it uses the same spatial encoder family and direct output construction, but
    accepts ``x`` rather than ``x_history`` and has no recurrent state.
    """

    uses_x_history = False
    target_name = "y_mechanism_trajectory"
    checkpoint_prefix = "static_mechanism_trajectory_"
    mechanism_order = AgingLabelGenerator.MECHANISM_ORDER

    def __init__(
        self,
        node_feature_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        gnn_layers: int = 3,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        components: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be at least one")

        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.spatial_encoder = HybridGNNTransformer(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            gnn_layers=gnn_layers,
            gat_heads=gat_heads,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            components=components,
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon * len(self.mechanism_order)),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        spatial_state = self.spatial_encoder.encode_graph(x, edge_index, edge_attr, batch)
        trajectory = self.trajectory_head(spatial_state)
        return trajectory.reshape(-1, self.horizon, len(self.mechanism_order))
