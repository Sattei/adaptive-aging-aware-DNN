import numpy as np

class HCIModel:
    """
    Hot Carrier Injection (HCI) Model.
    Models drain current shift over time.
    """
    def __init__(self, B: float, m: float):
        self.B = B
        self.m = m
        
    def compute_degradation(self, current_density: np.ndarray, stress_time_s: np.ndarray) -> np.ndarray:
        """
        Computes absolute degradation from hot carrier injection.
        ΔId/Id = B * (current_density)^m * sqrt(stress_time)
        
        Args:
            current_density: np.ndarray of shape [N], proxy current density.
            stress_time_s: np.ndarray of shape [N], seconds of stress.
            
        Returns:
            np.ndarray of shape [N] representing ΔId/Id.
        """
        safe_time = np.clip(stress_time_s, 0.0, None)
        return self.B * np.power(current_density + 1e-12, self.m) * np.sqrt(safe_time)

