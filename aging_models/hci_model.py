import numpy as np

class HCIModel:
    """Hot Carrier Injection: ΔId/Id = B·current_density^m·√t"""

    def __init__(self, B: float, m: float):
        self.B = B
        self.m = m

    def compute_degradation(self, current_density: np.ndarray, stress_time_s: np.ndarray) -> np.ndarray:
        return self.B * np.power(current_density + 1e-12, self.m) * np.sqrt(np.clip(stress_time_s, 0.0, None))

