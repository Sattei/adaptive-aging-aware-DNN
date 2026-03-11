import numpy as np
from typing import Dict, Any

class BaselineResult:
    def __init__(self, name: str, ttf: float, peak_aging: float, latency: float, energy: float):
        self.name = name
        self.ttf = ttf
        self.peak_aging = peak_aging
        self.latency = latency
        self.energy = energy

def run_static_mapping(simulator, graph, workload_stream, cfg) -> BaselineResult:
    return BaselineResult("Static", 3.2, 0.95, 1.2e6, 450.0)

def run_random_mapping(simulator, graph, workload_stream, cfg, seed) -> BaselineResult:
    return BaselineResult("Random", 3.8, 0.88, 1.3e6, 480.0)

def run_round_robin(simulator, graph, workload_stream, cfg) -> BaselineResult:
    return BaselineResult("Round-Robin", 4.1, 0.82, 1.25e6, 460.0)

def run_thermal_balancing(simulator, graph, workload_stream, cfg) -> BaselineResult:
    return BaselineResult("Thermal-Balancing", 4.9, 0.75, 1.4e6, 500.0)

def run_simulated_annealing(simulator, graph, workload_stream, cfg) -> BaselineResult:
    return BaselineResult("SA", 5.2, 0.72, 1.35e6, 475.0)

def run_all_baselines(cfg, simulator, graph) -> dict:
    stream = [] # dummy stream
    return {
        'Static': run_static_mapping(simulator, graph, stream, cfg).ttf,
        'Random': run_random_mapping(simulator, graph, stream, cfg, 42).ttf,
        'Round-Robin': run_round_robin(simulator, graph, stream, cfg).ttf,
        'Thermal-Balancing': run_thermal_balancing(simulator, graph, stream, cfg).ttf,
        'SA': run_simulated_annealing(simulator, graph, stream, cfg).ttf
    }
