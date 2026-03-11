import pytest
import numpy as np
from omegaconf import OmegaConf

from planning.lifetime_planner import LifetimePlanner
from optimization.chromosome_representation import MappingChromosome
from optimization.nsga2_optimizer import NSGA2Optimizer
from graph.accelerator_graph import AcceleratorGraph
from simulator.timeloop_runner import TimeloopRunner

def test_lifetime_planner():
    cfg = OmegaConf.create({
        'pe_array': [16, 16],
        'mac_clusters': 16,
        'sram_banks': 8,
        'noc_routers': 4,
        'failure_threshold': 0.8
    })
    graph = AcceleratorGraph(cfg)
    graph.build()
    
    planner = LifetimePlanner(graph, cfg)
    
    # Test equalized budget allocation
    budgets = planner.allocate_budgets(10.0, strategy='equalized')
    assert len(budgets) == 16 + 8 + 4
    assert budgets[0] == 0.8
    
    # Test budget violations
    scores = np.zeros(28)
    scores[5] = 0.9 # Trigger violation
    violations = planner.check_budget_violations(scores, budgets)
    assert len(violations) == 1
    assert violations[0] == 5
    
    # Test TTF calculation
    ttf = planner.compute_ttf(np.array([0.4]), 0.8) # 0.4 / 1 yr = 0.4 rate. 0.8 / 0.4 = 2 years
    assert np.isclose(ttf, 2.0)

def test_chromosome():
    chrom = MappingChromosome(num_layers=10, num_clusters=64)
    
    c1 = chrom.random_init(seed=42)
    assert len(c1) == 10
    assert np.all((c1 >= 0) & (c1 < 64))
    
    # Test validity
    assert chrom.is_valid(c1, {})
    
    invalid = np.array([-1, 0, 65, 3])
    # manually override chrom len for valid test
    chrom.num_layers = 4
    assert not chrom.is_valid(invalid, {})
    
    repaired = chrom.repair(invalid, {})
    assert np.all((repaired >= 0) & (repaired < 64))
    assert chrom.is_valid(repaired, {})
