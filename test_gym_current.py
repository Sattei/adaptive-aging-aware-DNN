import gymnasium as gym
from omegaconf import OmegaConf
from simulator.timeloop_runner import TimeloopRunner
from planning.lifetime_planner import LifetimePlanner
from rl.environment import AgingControlEnv

cfg = OmegaConf.create({
    "mac_clusters": 4, "sram_banks": 4, "noc_routers": 2, "pe_array": [4, 4]
})
sim = TimeloopRunner(cfg)
class MockPlanner:
    def allocate_budgets(self, **kwargs): return {}
    def check_budget_violations(self, *args, **kwargs): return []
pl = MockPlanner()

env_cfg = OmegaConf.create({'horizon_length': 5, 'max_episode_steps': 200, 'workload_feature_dim': 16, 'max_layers': 50})
env = AgingControlEnv(env_cfg, sim, pl)

from gymnasium.utils.env_checker import check_env
try:
    check_env(env, warn=False)
    print("gymnasium check_env: PASSED")
except Exception as e:
    print(f"gymnasium check_env warning/error: {e}")
