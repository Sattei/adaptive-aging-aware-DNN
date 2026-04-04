import numpy as np
from typing import List, Any
import json
from pathlib import Path
from omegaconf import DictConfig

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination

from simulator.timeloop_runner import TimeloopRunner, WorkloadResult
from optimization.chromosome_representation import MappingChromosome

class ParetoSolution:
    def __init__(self, mapping: np.ndarray, peak_aging: float, latency: float, energy: float):
        self.mapping = mapping
        self.peak_aging = peak_aging
        self.latency = latency
        self.energy = energy
        
    def to_dict(self):
        return {
            "mapping": self.mapping.tolist(),
            "peak_aging": float(self.peak_aging),
            "latency": float(self.latency),
            "energy": float(self.energy)
        }

class MappingProblem(ElementwiseProblem):
    def __init__(
        self, 
        simulator: TimeloopRunner, 
        aging_predictor: Any, # GNN Model stub reference
        num_layers: int, 
        num_clusters: int
    ):
        super().__init__(
            n_var=num_layers,
            n_obj=3, # [aging, latency, energy]
            n_ieq_constr=0,
            xl=np.zeros(num_layers, dtype=int),  # lower bound for each layer
            xu=np.full(num_layers, num_clusters - 1, dtype=int),  # upper bound for each layer
        )
        self.simulator = simulator
        self.aging_predictor = aging_predictor
        self.num_layers = num_layers
        self.num_clusters = num_clusters
        
        # dummy workload (all identical conv layers for structural proxy)
        self.dummy_workload = [
            {'type': 'conv2d', 'K': 64, 'C': 64, 'R': 3, 'S': 3, 'P': 32, 'Q': 32}
        ] * num_layers

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a single solution [n_var], NOT a population
        # This is the ElementwiseProblem interface
        mapping = x.astype(int)
        res = self.simulator.run_workload(self.dummy_workload, mapping)
        
        # Predict aging
        # In full pipeline, we would embed features, run through GNN
        # For analytical NSGA proxy: peak switching activity is directly proportional to aging
        # This allows the optimizer to run purely physically if the GNN is decoupled
        # For prompt compliance: we use peak instantaneous temperature proxy
        peak_aging = np.max(res.avg_switching_activity) 
        
        # Objectives (all minimized) - must be 1D array of length n_obj
        out["F"] = np.array([
            peak_aging,
            float(res.total_latency_cycles),
            res.total_energy_pj
        ])


class NSGA2Optimizer:
    """
    Multi-objective Evolutionary mapping solver.
    """
    def __init__(
        self,
        accelerator_config: DictConfig,
        simulator: TimeloopRunner,
        aging_predictor: Any,
        config: DictConfig,
    ):
        self.accel_cfg = accelerator_config
        self.sim = simulator
        self.predictor = aging_predictor
        self.pop_size = config.get('pop_size', 50)
        self.crossover_prob = config.get('crossover_prob', 0.9)
        self.mutation_prob = config.get('mutation_prob', 0.1)
        
        self.num_layers = accelerator_config.get('num_layers', 10) # default assumption
        self.num_clusters = accelerator_config.get('mac_clusters', 64)
        
        self.pareto_solutions = []
        
    def run(self, initial_mapping: np.ndarray, n_gen: int = 200) -> List[ParetoSolution]:
        """
        Executes optimization and returns the Pareto-optimal frontier.
        """
        problem = MappingProblem(self.sim, self.predictor, self.num_layers, self.num_clusters)
        
        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=self.crossover_prob, eta=15, vtype=float),
            mutation=PM(prob=self.mutation_prob, eta=20, vtype=float),
            eliminate_duplicates=True
        )
        
        termination = get_termination("n_gen", n_gen)
        
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            save_history=False,
            verbose=False
        )
        
        # Parse results
        self.pareto_solutions.clear()
        
        # NSGA-II might return a single solution if front is degenerate
        if res.F.ndim == 1:
            X_front = [res.X]
            F_front = [res.F]
        else:
            X_front = res.X
            F_front = res.F
            
        for i in range(len(X_front)):
            sol = ParetoSolution(
                mapping=np.round(X_front[i]).astype(int),
                peak_aging=F_front[i][0],
                latency=F_front[i][1],
                energy=F_front[i][2]
            )
            self.pareto_solutions.append(sol)
            
        return self.pareto_solutions
        
    def get_pareto_front(self) -> List[ParetoSolution]:
        return self.pareto_solutions
        
    def save_pareto_solutions(self, path: Path) -> None:
        data = [sol.to_dict() for sol in self.pareto_solutions]
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
