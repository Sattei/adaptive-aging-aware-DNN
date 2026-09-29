"""Temporal cutoff-aging predictor built on the existing spatial GNN encoder."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from aging_models.aging_label_generator import AgingLabelGenerator
from models.hybrid_gnn_transformer import HybridGNNTransformer


class _TemporalHistoryEncoder(nn.Module):
    """Shared chronological graph-history encoder for temporal aging models.

    The enclosed ``HybridGNNTransformer`` is used only through ``encode_graph``.
    It is called once for each history position with shared parameters, so its
    GCN, GAT, and node-attention layers remain spatial rather than temporal.
    In training mode its BatchNorm layers update once per historical frame.
    """

    uses_x_history = True

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        temporal_hidden_dim: int = 128,
        gru_layers: int = 1,
        gnn_layers: int = 3,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        components: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        if gru_layers < 1:
            raise ValueError("gru_layers must be at least one")

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.gru_layers = gru_layers

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
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

    def encode_spatial_snapshot(self, x, edge_index, edge_attr=None, batch=None):
        """Return shared spatial embeddings for one ``[M, F]`` history frame."""
        return self.spatial_encoder.encode_graph(x, edge_index, edge_attr, batch)

    def encode_history(self, x_history, edge_index, edge_attr=None, batch=None):
        """Encode history oldest-to-newest and return the final GRU state."""
        self._validate_history(x_history)
        spatial_states = [
            self.encode_spatial_snapshot(x_history[:, step, :], edge_index, edge_attr, batch)
            for step in range(x_history.size(1))
        ]
        spatial_sequence = torch.stack(spatial_states, dim=1)
        _, hidden = self.temporal_gru(spatial_sequence)
        return hidden[-1]

    def _validate_history(self, x_history) -> None:
        if x_history.dim() != 3:
            raise ValueError(
                "x_history must have shape [num_nodes, history_length, feature_dim], "
                f"got {tuple(x_history.shape)}"
            )
        if x_history.size(1) == 0:
            raise ValueError("x_history must contain at least one historical frame")
        if x_history.size(2) != self.node_feature_dim:
            raise ValueError(
                f"x_history feature dimension must be {self.node_feature_dim}, "
                f"got {x_history.size(2)}"
            )


class TemporalGNN(_TemporalHistoryEncoder):
    """Encode chronological graph snapshots and predict cutoff aging per node."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        temporal_hidden_dim: int = 128,
        gru_layers: int = 1,
        gnn_layers: int = 3,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        components: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            gru_layers=gru_layers,
            gnn_layers=gnn_layers,
            gat_heads=gat_heads,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            components=components,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.temporal_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_history, edge_index, edge_attr=None, batch=None):
        temporal_state = self.encode_history(x_history, edge_index, edge_attr, batch)
        return self.regression_head(temporal_state)


class TemporalMechanismTrajectoryGNN(_TemporalHistoryEncoder):
    """Direct future NBTI/HCI/TDDB trajectory predictor from graph history."""

    target_name = "y_mechanism_trajectory"
    checkpoint_prefix = "mechanism_trajectory_"
    mechanism_order = AgingLabelGenerator.MECHANISM_ORDER

    def __init__(
        self,
        node_feature_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        temporal_hidden_dim: int = 128,
        gru_layers: int = 1,
        gnn_layers: int = 3,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        components: Optional[Iterable[str]] = None,
    ):
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        super().__init__(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            gru_layers=gru_layers,
            gnn_layers=gnn_layers,
            gat_heads=gat_heads,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            components=components,
        )
        self.horizon = horizon
        self.trajectory_head = nn.Sequential(
            nn.Linear(self.temporal_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon * len(self.mechanism_order)),
            nn.Sigmoid(),
        )

    def forward(self, x_history, edge_index, edge_attr=None, batch=None):
        temporal_state = self.encode_history(x_history, edge_index, edge_attr, batch)
        trajectory = self.trajectory_head(temporal_state)
        return trajectory.reshape(-1, self.horizon, len(self.mechanism_order))
