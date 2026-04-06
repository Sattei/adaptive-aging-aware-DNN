import numpy as np

class TDDBModel:
    """Time-Dependent Dielectric Breakdown: F(t) = 1 − exp(−exp(k·E − β))"""

    def __init__(self, k: float, beta: float):
        self.k = k
        self.beta = beta

    def failure_probability(self, electric_field_proxy: np.ndarray, stress_time_s: np.ndarray) -> np.ndarray:
        exp_inner = np.clip(np.exp(np.clip(self.k * electric_field_proxy - self.beta, -50, 50)), 0, 1e22)
        return np.clip(1.0 - np.exp(-exp_inner), 0.0, 1.0)

    def time_to_failure(self, electric_field_proxy: np.ndarray, target_failure_prob: float = 0.001) -> np.ndarray:
        failure_rate = np.exp(np.clip(self.k * electric_field_proxy - self.beta, -50, 50)) + 1e-15
        return -np.log(1.0 - np.clip(target_failure_prob, 1e-6, 0.999999)) / (failure_rate + 1e-12)

