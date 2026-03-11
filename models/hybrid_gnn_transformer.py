import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import math
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class HybridGNNTransformer(nn.Module):
    """
    Hybrid Architecture capturing spatial hardware topology (GNN)
    and temporal workload sequences (Transformer).
    """
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 3,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        seq_len: int = 10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(gnn_layers - 1)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(gnn_layers - 1)
        ])
        
        self.gat = GATConv(hidden_dim, hidden_dim // gat_heads, heads=gat_heads)
        self.gat_bn = nn.BatchNorm1d(hidden_dim)
        
        # Temporal Transformer (Unused in single-step forward, but kept for init compatibility)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=transformer_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Stage 3: Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def encode_graph(self, x, edge_index, edge_attr=None):
        import torch.nn.functional as F
        h = F.relu(self.input_proj(x))

        for gcn, bn in zip(self.gcn_layers, self.bn_layers):
            h_new = F.relu(bn(gcn(h, edge_index)))
            h = h + h_new  # residual

        # GAT: pass edge_attr only if it exists and model expects it
        try:
            if edge_attr is not None and edge_attr.shape[0] > 0:
                h_gat = self.gat(h, edge_index, edge_attr=edge_attr)
            else:
                h_gat = self.gat(h, edge_index)
        except TypeError:
            # Fallback: some GAT variants don't accept edge_attr
            h_gat = self.gat(h, edge_index)
        h = self.gat_bn(h_gat) + h
        return h

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass. Returns [N, 1] aging predictions.
        Compatible with single graphs and PyG Batch objects.
        """
        h = self.encode_graph(x, edge_index, edge_attr)
        out = self.head(h)
        return out
