import torch
from graph.accelerator_graph import AcceleratorGraph
from omegaconf import OmegaConf

# Match run_full_pipeline config shape
cfg = {
    'pe_array': [16, 16],
    'mac_clusters': 16,
    'sram_banks': 8,
    'noc_routers': 4,
    'num_layers': 10
}
ag = AcceleratorGraph(cfg)
ag.build()

dummy_features = torch.zeros(ag.graph.number_of_nodes(), 6).numpy()
data = ag.to_pyg(dummy_features)

assert data.edge_index.shape[0] == 2, "FAIL: edge_index shape"
assert data.edge_index.dtype == torch.long, "FAIL: edge_index dtype"
assert data.x.shape[1] == 6, "FAIL: node feature dim"
print("PASS: edge_index is correct")
print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
