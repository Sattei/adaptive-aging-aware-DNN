import torch
from evaluation.performance_metrics import PerformanceMetrics

pm = PerformanceMetrics()
p = torch.tensor([[0.5, 0.6], [0.1, 0.9]])
t = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
res = pm.compute_all_metrics(p, t)

print(f"Metrics: {res}")
assert 'mae' in res
assert 'rmse' in res
assert 'r2' in res
assert res['mae'] > 0
print("PERFORMANCE METRICS OK")
