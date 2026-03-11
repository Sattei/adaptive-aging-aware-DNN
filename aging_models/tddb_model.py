import numpy as np

class TDDBModel:
    """
    Time-Dependent Dielectric Breakdown (TDDB) Model.
    Models probability of dielectric failure based on electric field and time.
    """
    def __init__(self, k: float, beta: float):
        self.k = k
        self.beta = beta
        
    def failure_probability(self, electric_field_proxy: np.ndarray, stress_time_s: np.ndarray) -> np.ndarray:
        """
        Computes failure probability.
        failure_prob = 1 - exp(-exp(k * E_field - beta))
        
        Args:
            electric_field_proxy: np.ndarray of shape [N]
            stress_time_s: np.ndarray of shape [N] (included for interface consistency)
            
        Returns:
            np.ndarray of shape [N] with values [0, 1].
        """
        # Canonical spec formula
        exponent_inner = self.k * electric_field_proxy - self.beta
        # Clip to prevent overflow/underflow
        exponent_inner = np.clip(exponent_inner, -50, 50)
        
        # exp(inner) gets clipped at 1e22 before negation
        exp_inner = np.exp(exponent_inner)
        exp_inner = np.clip(exp_inner, 0, 1e22)
        
        prob = 1.0 - np.exp(-exp_inner)
        return np.clip(prob, 0.0, 1.0)
        

    def time_to_failure(self, electric_field_proxy: np.ndarray, target_failure_prob: float = 0.001) -> np.ndarray:
        """
        Estimates expected time to failure given a target cumulative unreliability.
        
        Args:
            electric_field_proxy: np.ndarray of shape [N]
            target_failure_prob: float
            
        Returns:
            np.ndarray of shape [N] in seconds.
        """
        exponent_inner = self.k * electric_field_proxy - self.beta
        exponent_inner = np.clip(exponent_inner, -50, 50)
        failure_rate = np.exp(exponent_inner) + 1e-15
        
        # 1 - P = exp(-F)  => -ln(1-P) = F => F/failure_rate = t (normalized)
        ttf = -np.log(1.0 - np.clip(target_failure_prob, 1e-6, 0.999999)) / (failure_rate + 1e-12)
        return ttf

