from omegaconf import OmegaConf
from simulator.timeloop_runner import TimeloopRunner
import numpy as np

# Mock config
cfg = OmegaConf.create({
    "accelerator": {
        "pe_rows": 4, "pe_cols": 4,
        "num_mac_clusters": 4, "num_sram_banks": 4, "num_noc_routers": 2,
        "clock_ghz": 1.0, "supply_voltage": 0.8,
        "mac_energy_pj_per_op": 0.1, "sram_read_energy_pj": 2.0, "noc_hop_energy_pj": 0.5,
        "ops_per_cycle": 2, "max_macs_per_cluster": 256, "max_memory_per_bank": 4096,
        "noc_bandwidth_gbps": 64.0,
    }
})

sim = TimeloopRunner(cfg.accelerator)
layer = {"type": "conv2d", "K": 64, "C": 3, "R": 3, "S": 3, "P": 56, "Q": 56}
mapping = np.array([0], dtype=np.int32)
r = sim.run_layer(layer, mapping)
print(f"LayerResult: latency={r.latency_cycles}, energy={r.energy_pj:.1f}pJ")
print(f"switching_activity shape: {r.switching_activity.shape}")
assert r.latency_cycles > 0
assert r.energy_pj > 0
assert r.switching_activity.min() >= 0
assert r.switching_activity.max() <= 1.0
print("SIMULATOR OK")
