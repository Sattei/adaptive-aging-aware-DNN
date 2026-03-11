class PerformanceMetrics:
    def latency_overhead_pct(self, lat_baseline: float, lat_system: float) -> float:
        return float((lat_system - lat_baseline) / lat_baseline * 100.0)
        
    def energy_overhead_pct(self, energy_baseline: float, energy_system: float) -> float:
        return float((energy_system - energy_baseline) / energy_baseline * 100.0)
        
    def throughput(self, latency_cycles: int, clock_ghz: float) -> float:
        if latency_cycles == 0:
            return 0.0
        time_s = latency_cycles / (clock_ghz * 1e9)
        return float(1.0 / time_s)
        
    def energy_efficiency(self, throughput: float, energy_pj: float) -> float:
        if energy_pj == 0.0:
            return 0.0
        power_w = energy_pj * 1e-12 * throughput
        return float(throughput / power_w) if power_w > 0 else 0.0

    def compute_all_metrics(self, pred, target):
        """
        Returns dict: {'mae': float, 'rmse': float, 'r2': float}
        """
        import torch
        import numpy as np
        from sklearn.metrics import r2_score

        # Move to CPU numpy for safe scikit-learn usage
        if isinstance(pred, torch.Tensor):
            p = pred.detach().cpu().numpy()
        else:
            p = np.array(pred)

        if isinstance(target, torch.Tensor):
            t = target.detach().cpu().numpy()
        else:
            t = np.array(target)

        # Flatten if necessary
        p = p.flatten()
        t = t.flatten()

        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))

        # Safe R2
        if len(t) < 2 or np.var(t) == 0:
            r2 = 0.0
        else:
            r2 = r2_score(t, p)

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
