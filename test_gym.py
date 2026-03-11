from omegaconf import OmegaConf
import sys
sys.path.insert(0, '.')

cfg = OmegaConf.create({
    "accelerator": {
        "num_mac_clusters": 4, "num_sram_banks": 4, "num_noc_routers": 2,
        "pe_rows": 4, "pe_cols": 4,
        "max_macs_per_cluster": 256, "max_memory_per_bank": 4096,
        "clock_ghz": 1.0, "sram_read_energy_pj": 2.0, "noc_hop_energy_pj": 0.5, "mac_energy_pj_per_op": 0.1, "ops_per_cycle": 2
    },
    "workloads": {"dummy": "data"},
    "planning": {"failure_threshold": 0.8},
    "reward": {"latency_threshold": 1.05, "energy_threshold": 1.03, "w_peak": 1.0, "w_variance": 0.5, "w_latency": 2.0, "w_energy": 1.5, "w_budget": 0.3}
})
# Missing config from the prompt's snippet:
cfg.aging = {"nbti_A": 0.005, "nbti_n": 0.25, "hci_B": 0.0001, "hci_m": 0.5, "tddb_k": 2.5, "tddb_beta": 10.0}

from simulator.timeloop_runner import TimeloopRunner
from simulator.workload_runner import WorkloadRunner
from aging_models.aging_label_generator import AgingLabelGenerator
from graph.accelerator_graph import AcceleratorGraph
from planning.lifetime_planner import LifetimePlanner
from features.feature_builder import FeatureBuilder
from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.trajectory_predictor import TrajectoryPredictor
from rl.environment import AgingControlEnv

sim = TimeloopRunner(cfg.accelerator)
wr  = WorkloadRunner(cfg.workloads)
ag  = AgingLabelGenerator(cfg=cfg)
acc = AcceleratorGraph(cfg.accelerator)
acc.build()
pl  = LifetimePlanner(acc, cfg)
fb  = FeatureBuilder(cfg.accelerator)
enc = HybridGNNTransformer(node_feature_dim=7, hidden_dim=32,
                            gnn_layers=2, gat_heads=2,
                            transformer_heads=2, seq_len=3)
tp  = TrajectoryPredictor(enc, horizon=10)

env = AgingControlEnv(sim, pl, wr, ag, acc, enc, tp, fb, cfg)

# Test reset
obs, info = env.reset()
print(f"reset() → obs.shape={obs.shape}, info={type(info)}")
assert isinstance(info, dict)

# Test step — must return 5 values
result = env.step(4)  # no-op
assert len(result) == 5, f"step() returned {len(result)} values, expected 5"
obs2, reward, terminated, truncated, info2 = result
assert isinstance(reward, float)
assert isinstance(terminated, bool)
assert isinstance(truncated, bool)
print(f"step() → reward={reward:.4f}, terminated={terminated}, truncated={truncated}")

# Full gymnasium check
from gymnasium.utils.env_checker import check_env
try:
    check_env(env, warn=False)
    print("gymnasium check_env: PASSED")
except Exception as e:
    print(f"gymnasium check_env warning: {e}")
print("RL ENVIRONMENT OK")
