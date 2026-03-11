import torch
import numpy as np
from omegaconf import OmegaConf
from graph.accelerator_graph import AcceleratorGraph
from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.trajectory_predictor import TrajectoryPredictor

cfg_accel = OmegaConf.create({
    "num_mac_clusters": 4,
    "num_sram_banks": 4,
    "num_noc_routers": 2,
    "pe_array": [4, 4],
})

ag = AcceleratorGraph(cfg_accel)
ag.build()
data = ag.to_pyg(np.random.rand(ag.get_num_nodes(), 7).astype(np.float32))
enc = HybridGNNTransformer(node_feature_dim=7, hidden_dim=32,
                            gnn_layers=2, gat_heads=2,
                            transformer_heads=2, seq_len=3)
# TrajectoryPredictor also needs hidden_dim matching what we provide to enc
# It defaults to 256, but we need 32 to match enc.hidden_dim
model = TrajectoryPredictor(enc, hidden_dim=32, horizon=10, gamma=0.95)
model.eval()
with torch.no_grad():
    traj = model(data.x, data.edge_index, data.edge_attr)
print(f"Trajectory shape: {traj.shape}")
assert traj.shape == (ag.get_num_nodes(), 10)
assert not torch.isnan(traj).any()

# Test loss
target = torch.rand_like(traj)
loss = model.trajectory_loss(traj.requires_grad_(True), target)
loss.backward()
print(f"Trajectory loss: {loss.item():.4f}")
print("TRAJECTORY PREDICTOR OK")
