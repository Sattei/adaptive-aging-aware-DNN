import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AgingControlEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, sim, planner, workload_runner, aging_gen, acc_graph, encoder, trajectory_pred, feature_builder, cfg):
        super().__init__()
        self.sim = sim
        self.planner = planner
        self.wr = workload_runner
        self.ag = aging_gen
        self.acc = acc_graph
        self.enc = encoder
        self.tp = trajectory_pred
        self.fb = feature_builder
        self.cfg = cfg
        
        self.num_macs = max(self.sim.num_mac_clusters, 1)
        self.N = self.acc.get_num_nodes()
        self.max_steps = 100
        
        self.k_horizon = self.cfg.get('horizon_length', 5) if isinstance(self.cfg, dict) else getattr(self.cfg, 'horizon_length', 5)
        self.W = self.cfg.get('workload_feature_dim', 16) if isinstance(self.cfg, dict) else getattr(self.cfg, 'workload_feature_dim', 16)
        self.L = self.cfg.get('max_layers', 10) if isinstance(self.cfg, dict) else getattr(self.cfg, 'max_layers', 10)
        
        self.action_space = spaces.Discrete(5)
        
        obs_dim = self.N + (self.N * self.k_horizon) + self.W + self.L + 1 + self.N
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.step_count = 0
        self.aging_vector = np.zeros(self.N, dtype=np.float32)
        self._workload_names = ["ResNet50"]
        self.current_workload = "ResNet50"

    def _get_obs(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _reset_state(self):
        self.step_count = 0
        self.aging_vector = np.zeros(self.N, dtype=np.float32)

    def _apply_action(self, action):
        pass

    def _run_simulation_step(self):
        return {"latency_ratio": 1.0, "energy_ratio": 1.0}
        
    def step(self, action: int):
        """Returns (obs, reward, terminated, truncated, info) — 5-tuple."""
        prev_peak = float(self.aging_vector.max() if len(self.aging_vector) > 0 else 0)
        prev_var  = float(np.var(self.aging_vector))

        self._apply_action(int(action))
        sim_info = self._run_simulation_step()

        curr_peak = float(self.aging_vector.max() if len(self.aging_vector) > 0 else 0)
        curr_var  = float(np.var(self.aging_vector))
        try:
            violations = len(self.planner.check_budget_violations(self.aging_vector))
        except:
            violations = 0

        lat_pen = max(0.0, sim_info["latency_ratio"] - self.cfg.reward.latency_threshold)
        eng_pen = max(0.0, sim_info["energy_ratio"]  - self.cfg.reward.energy_threshold)

        reward = float(
            self.cfg.reward.w_peak    * (prev_peak - curr_peak)
            + self.cfg.reward.w_variance * (prev_var  - curr_var)
            - self.cfg.reward.w_latency  * lat_pen
            - self.cfg.reward.w_energy   * eng_pen
            - self.cfg.reward.w_budget   * violations
        )

        self.step_count += 1
        terminated = bool(curr_peak >= getattr(self.cfg.planning, 'failure_threshold', 1.0))
        truncated  = bool(self.step_count >= self.max_steps)

        info = {
            "peak_aging":      curr_peak,
            "aging_variance":  curr_var,
            "latency_ratio":   sim_info["latency_ratio"],
            "energy_ratio":    sim_info["energy_ratio"],
            "budget_violations": violations,
        }
        return self._get_obs(), reward, terminated, truncated, info  # 5-tuple

    def reset(self, seed=None, options=None):
        """Returns (obs, info) — 2-tuple."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        else:
            self._rng = np.random.default_rng()
        self._reset_state()
        self.current_workload = str(self._rng.choice(self._workload_names))
        self._run_simulation_step()
        return self._get_obs(), {}  # 2-tuple
