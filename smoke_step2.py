import torch
from graph.accelerator_graph import AcceleratorGraph
from models.hybrid_gnn_transformer import HybridGNNTransformer
from torch_geometric.data import Batch

cfg = {
    'pe_array': [16, 16], 'mac_clusters': 16,
    'sram_banks': 8, 'noc_routers': 4, 'num_layers': 10
}
ag = AcceleratorGraph(cfg)
ag.build()
dummy_features = torch.zeros(ag.graph.number_of_nodes(), 6).numpy()
data = ag.to_pyg(dummy_features)

model = HybridGNNTransformer(node_feature_dim=6, hidden_dim=64, seq_len=3)
model.eval()

# Create a batch of 2 graphs
graphs = [data, data]  # reuse from Step 1
batch = Batch.from_data_list(graphs)

with torch.no_grad():
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, seq_mask=None)

assert out.shape == (batch.num_nodes, 1), f"FAIL: output shape {out.shape}"
assert not torch.isnan(out).any(), "FAIL: NaN in output"
print(f"PASS: forward pass output shape {out.shape}")
