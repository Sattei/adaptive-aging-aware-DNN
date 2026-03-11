import numpy as np

class NBTIModel:
    """
    Negative Bias Temperature Instability (NBTI) Model.
    Models threshold voltage shift over time.
    """
    def __init__(self, A: float, n: float, temperature_K: float = 373.0):
        self.A = A
        self.n = n
        self.temperature_K = temperature_K
        
    def compute_degradation(self, stress_time_s: np.ndarray, switching_activity: np.ndarray) -> np.ndarray:
        """
        Computes absolute degradation from stress.
        ΔVth = A * (switching_activity * stress_time)^n
        
        Args:
            stress_time_s: np.ndarray of shape [N] containing total seconds of stress per node.
            switching_activity: np.ndarray of shape [N] containing duty cycle proxy (0.0 to 1.0).
            
        Returns:
            np.ndarray of shape [N] representing ΔVth.
        """
        effective_stress = switching_activity * stress_time_s
        # Ensure we don't take power of exactly zero to avoid warnings or NaN in some np versions
        effective_stress = np.clip(effective_stress, 1e-12, None)
        return self.A * np.power(effective_stress, self.n)

    def accumulate(self, existing_degradation: np.ndarray, new_stress: np.ndarray, delta_t: float) -> np.ndarray:
        """
        Incremental degradation accumulation utilizing time superposition.
        
        Args:
            existing_degradation: np.ndarray [N] of current ΔVth
            new_stress: np.ndarray [N] of current switching activity
            delta_t: float, continuous time block length in seconds
            
        Returns:
            np.ndarray [N] of updated ΔVth
        """
        # Inverse function: get equivalent time to reach existing degradation
        # t_eq = (ΔVth / A)^(1/n) / switching_activity
        # Protect against division by zero 
        safe_stress = np.clip(new_stress, 1e-9, None)
        
        t_eq = np.power(existing_degradation / self.A, 1.0 / self.n) / safe_stress
        
        return self.compute_degradation(t_eq + delta_t, new_stress)

