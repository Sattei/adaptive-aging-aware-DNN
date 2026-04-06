import numpy as np

class NBTIModel:
    """Negative Bias Temperature Instability: ΔVth = A·(activity·t)^n"""

    def __init__(self, A: float, n: float, temperature_K: float = 373.0):
        self.A = A
        self.n = n
        self.temperature_K = temperature_K

    def compute_degradation(self, stress_time_s: np.ndarray, switching_activity: np.ndarray) -> np.ndarray:
        effective_stress = np.clip(switching_activity * stress_time_s, 1e-12, None)
        return self.A * np.power(effective_stress, self.n)

    def accumulate(self, existing_degradation: np.ndarray, new_stress: np.ndarray, delta_t: float) -> np.ndarray:
        safe_stress = np.clip(new_stress, 1e-9, None)
        t_eq = np.power(existing_degradation / self.A, 1.0 / self.n) / safe_stress
        return self.compute_degradation(t_eq + delta_t, new_stress)
