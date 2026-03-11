import sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
import torch

cfg = OmegaConf.create({
    "accelerator": {
        "pe_rows": 4, "pe_cols": 4,
        "num_mac_clusters": 4, "num_sram_banks": 4, "num_noc_routers": 2,
        "clock_ghz": 1.0, "supply_voltage": 0.8,
        "mac_energy_pj_per_op": 0.1, "sram_read_energy_pj": 2.0, "noc_hop_energy_pj": 0.5,
        "ops_per_cycle": 2, "max_macs_per_cluster": 256, "max_memory_per_bank": 4096,
        "noc_bandwidth_gbps": 64.0,
    },
    "workloads": {
        "burst_length": 5, "total_steps": 10,
    },
    "aging": {
        "nbti_A": 0.005, "nbti_n": 0.25,
        "hci_B": 0.0001, "hci_m": 0.5,
        "tddb_k": 2.5, "tddb_beta": 10.0,
    },
})

from graph.graph_dataset import AgingDataset
import shutil, os
# Clean any stale cache
if os.path.exists("data/processed"):
    shutil.rmtree("data/processed")

ds = AgingDataset(root="data", split="train", size=20, cfg=cfg, seed=42)
print(f"Dataset size: {len(ds)}")
s = ds[0]
print(f"x: {s.x.shape}, dtype: {s.x.dtype}")
print(f"edge_index: {s.edge_index.shape}, dtype: {s.edge_index.dtype}")
print(f"y: {s.y.shape}")
print(f"y_trajectory: {s.y_trajectory.shape}")

# Critical checks
assert s.edge_index.shape[0] == 2, "FAIL edge_index shape"
assert s.edge_index.dtype == torch.long, "FAIL edge_index dtype"
assert s.x.shape[1] == 7, "FAIL feature dim"
assert not torch.isnan(s.x).any(), "FAIL NaN in features"
assert not torch.isnan(s.y).any(), "FAIL NaN in labels"
assert s.y.min() >= 0 and s.y.max() <= 1, "FAIL aging not in [0,1]"
print("ALL DATASET CHECKS PASSED")
