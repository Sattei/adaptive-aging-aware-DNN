import pytest
import numpy as np
from evaluation.reliability_metrics import ReliabilityMetrics
from evaluation.performance_metrics import PerformanceMetrics
from evaluation.statistical_tests import StatisticalTests

def test_reliability_metrics():
    metrics = ReliabilityMetrics()
    aging = np.array([0.1, 0.5, 0.8, 0.2])
    
    assert metrics.peak_aging(aging) == 0.8
    assert metrics.hotspot_count(aging, 0.7) == 1
    
    # 0.8 Threshold, max trajectory is 0.8 in 1 year = 0.8 rate. TTF = 0.8 / 0.8 = 1.0 (but it's already failed >= threshold)
    ttf = metrics.time_to_failure(aging, threshold=0.8)
    assert ttf == 0.0
    
    aging_safe = np.array([0.1, 0.4]) # max 0.4 -> rate 0.4. TTF = 0.8 / 0.4 = 2.0
    ttf_safe = metrics.time_to_failure(aging_safe, threshold=0.8)
    assert np.isclose(ttf_safe, 2.0)

def test_performance_metrics():
    metrics = PerformanceMetrics()
    
    # throughput test: 1 million cycles at 1 GHz = 1M / 1B = 0.001 sec -> 1000 IPS
    tp = metrics.throughput(1_000_000, 1.0)
    assert np.isclose(tp, 1000.0)
    
    # efficiency: 1000 IPS, 1e9 pJ = 1 mJ. Power = E * throughput = 1mJ * 1000 = 1 J/s = 1 W
    # eff = 1000 IPS / 1 W = 1000 IPS/W
    eff = metrics.energy_efficiency(tp, 1e9)
    assert np.isclose(eff, 1000.0)

def test_statistical_tests():
    stats = StatisticalTests()
    
    baseline = [5.0, 5.1, 4.9, 5.2, 4.8]
    system = [7.0, 7.2, 6.9, 7.1, 6.8]
    
    res = stats.paired_ttest(baseline, system)
    assert res['significant'] == True
    assert res['p_value'] < 0.05
    assert res['effect_size_cohens_d'] < 0 # baseline - system is negative
    
    ci_low, ci_high = stats.confidence_interval(system)
    assert ci_low < np.mean(system) < ci_high
    
    df = stats.run_full_comparison({"Base1": baseline}, system)
    assert len(df) == 1
    assert "Base1" in df["Baseline"].values
