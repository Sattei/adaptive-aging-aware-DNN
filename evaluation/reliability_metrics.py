import numpy as np

class ReliabilityMetrics:
    """Calculates formal reliability evaluation constants."""
    def peak_aging(self, aging_vector: np.ndarray) -> float:
        return float(np.max(aging_vector))
        
    def aging_variance(self, aging_vector: np.ndarray) -> float:
        return float(np.var(aging_vector))
        
    def hotspot_count(self, aging_vector: np.ndarray, threshold: float = 0.7) -> int:
        return int(np.sum(aging_vector >= threshold))
        
    def time_to_failure(self, aging_trajectory: np.ndarray, threshold: float = 0.8) -> float:
        if len(aging_trajectory.shape) > 1:
            current_score = np.max(aging_trajectory[-1, :])
        else:
            current_score = np.max(aging_trajectory)
            
        if current_score >= threshold: return 0.0
        if current_score <= 1e-6: return 10.0
        
        current_time_years = 1.0 # default arbitrary simulation block size
        rate = current_score / current_time_years
        return float(np.clip(threshold / rate, 0.0, 10.0))
        
    def lifetime_improvement(self, ttf_baseline: float, ttf_system: float) -> float:
        if ttf_baseline < 1e-5:
            return 0.0
        return float((ttf_system - ttf_baseline) / ttf_baseline * 100.0)
        
    def hotspot_reduction_pct(self, peak_before: float, peak_after: float) -> float:
        if peak_before < 1e-5:
            return 0.0
        return float((peak_before - peak_after) / peak_before * 100.0)


class PerformanceMetrics:
    def latency_overhead_pct(self, lat_baseline: float, lat_system: float) -> float:
        return float((lat_system - lat_baseline) / lat_baseline * 100.0)
        
    def energy_overhead_pct(self, energy_baseline: float, energy_system: float) -> float:
        return float((energy_system - energy_baseline) / energy_baseline * 100.0)
        
    def throughput(self, latency_cycles: int, clock_ghz: float) -> float:
        # Cycles / (Cycles/Sec) = Sec
        if latency_cycles == 0:
            return 0.0
        time_s = latency_cycles / (clock_ghz * 1e9)
        return float(1.0 / time_s)
        
    def energy_efficiency(self, throughput: float, energy_pj: float) -> float: # IPS/W
        if energy_pj == 0.0:
            return 0.0
        power_w = energy_pj * 1e-12 * throughput
        return float(throughput / power_w) if power_w > 0 else 0.0
