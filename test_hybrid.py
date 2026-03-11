import torch
import numpy as np
from omegaconf import OmegaConf
from graph.accelerator_graph import AcceleratorGraph
from models.hybrid_gnn_transformer import HybridGNNTransformer

cfg_accel = OmegaConf.create({
    "num_mac_clusters": 4,
    "num_sram_banks": 4,
    "num_noc_routers": 2,
    "pe_array": [4, 4],
})

ag = AcceleratorGraph(cfg_accel)
ag.build()
dummy_feats = np.random.rand(10, 7).astype(np.float32)
data = ag.to_pyg(dummy_feats)

model = HybridGNNTransformer(node_feature_dim=7, hidden_dim=32,
                               gnn_layers=2, gat_heads=2,
                               transformer_layers=1, transformer_heads=2,
                               seq_len=3)
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index, data.edge_attr)

print(f"Output shape: {out.shape}")
assert out.shape == (ag.get_num_nodes(), 1), f"Expected [{ag.get_num_nodes()}, 1], got {out.shape}"
assert not torch.isnan(out).any(), "NaN in output"
# output range check removed because no activation on last layer
# test backward
model.train()
out = model(data.x, data.edge_index, data.edge_attr)
loss = out.mean()
loss.backward()
print("HYBRID GNN-TRANSFORMER OK (forward + backward)")
