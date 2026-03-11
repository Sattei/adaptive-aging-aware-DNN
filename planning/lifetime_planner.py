import numpy as np
from typing import List, Dict, Any
from omegaconf import DictConfig

class LifetimePlanner:
    """
    Tracks and allocates aging budgets across the accelerator components.
    """
    def __init__(self, accelerator_graph: Any, config: DictConfig):
        self.graph = accelerator_graph
        self.config = config
        
        self.num_nodes = self.graph.get_num_nodes()
        self.failure_threshold = self.config.get('failure_threshold', 0.8) # Normalize score representing failure
        
    def allocate_budgets(self, target_lifetime_years: float, strategy: str = 'equalized') -> Dict[int, float]:
        """
        Calculates maximum allowable aging per node to meet target lifetime.
        
        Args:
            target_lifetime_years: Global lifetime goal
            strategy: 'equalized', 'capacity_weighted', 'type_weighted'
            
        Returns:
            Dict mapping node_id to max allowable score.
        """
        # For simplicity, treat the max failure threshold as the total budget available over `target_lifetime_years`
        # We assign an instantaneous "budget" conceptually, but mathematically here 
        # budget_i represents the targeted *maximum* score for that node at the end of life.
        
        budgets = {}
        if strategy == 'equalized':
            # Everyone gets same budget up to threshold
            target = self.failure_threshold
            for i in range(self.num_nodes):
                budgets[i] = target
        elif strategy == 'type_weighted':
            # Routers fail faster, allocate them lower thresholds? 
            # OR allocate them more buffer. Let's dictate budget == allowable degradation.
            for i in range(self.num_nodes):
                ntype = self.graph.get_node_info(i)['type']
                if ntype == 'mac':
                    budgets[i] = self.failure_threshold * 1.0
                elif ntype == 'sram':
                    budgets[i] = self.failure_threshold * 0.9
                else:
                    budgets[i] = self.failure_threshold * 0.8
        else: # Capacity weighted
            total_cap = sum(self.graph.get_node_info(i).get('capacity', 1.0) for i in range(self.num_nodes))
            for i in range(self.num_nodes):
                cap_ratio = self.graph.get_node_info(i).get('capacity', 1.0) / total_cap
                # Scaled arbitrarily around threshold
                budgets[i] = min(1.0, self.failure_threshold * (0.5 + cap_ratio * self.num_nodes))
                
        return budgets

    def check_budget_violations(self, current_aging_vector: np.ndarray, budgets: Dict[int, float] = None) -> List[int]:
        """
        Returns list of node indices exceeding their allocated budget threshold.
        """
        if budgets is None:
            budgets = self.allocate_budgets(10.0, 'equalized')
            
        violations = []
        for i, score in enumerate(current_aging_vector):
            if score > budgets.get(i, self.failure_threshold):
                violations.append(i)
        return violations

    def compute_equalization_reward(self, aging_trajectory: np.ndarray) -> float:
        """
        Computes a scalar reward to incentivize uniform aging.
        Higher is better.
        
        Args:
            aging_trajectory: np.ndarray [T, N] or [N] of scores
        """
        # Extract terminal state if sequence
        if len(aging_trajectory.shape) > 1:
            terminal_scores = aging_trajectory[-1, :]
        else:
            terminal_scores = aging_trajectory
            
        variance = np.var(terminal_scores)
        peak = np.max(terminal_scores)
        
        # reward = -Var(A_i(T)) - λ * max_i(A_i(T))
        lambda_val = self.config.get('penalty_lambda', 2.0)
        
        reward = -float(variance) - (lambda_val * float(peak))
        return reward

    def recommend_rebalance(self, predicted_trajectories: np.ndarray, current_mapping: np.ndarray) -> dict:
        """
        Analyzes predicted future bottlenecks and suggests greedy mapping corrections.
        
        Args:
            predicted_trajectories: [N, k] future score horizon
            current_mapping: array of cluster assignments [L]
        """
        # 1. Find hottest predicted node at horizon k
        terminal_preds = predicted_trajectories[:, -1]
        hottest_node = int(np.argmax(terminal_preds))
        coolest_node = int(np.argmin(terminal_preds))
        
        # We only really care about transferring MAC clustering loads for simple greedy strategies
        hottest_info = self.graph.get_node_info(hottest_node)
        coolest_info = self.graph.get_node_info(coolest_node)
        
        reassign_actions = []
        
        if hottest_info['type'] == 'mac' and coolest_info['type'] == 'mac':
            h_idx = hottest_info['local_idx']
            c_idx = coolest_info['local_idx']
            
            # Find a layer mapped to the hot cluster and move it
            candidate_layers = np.where(current_mapping == h_idx)[0]
            if len(candidate_layers) > 0:
                layer_to_move = candidate_layers[0] # arbitrarily pick first
                reassign_actions.append((int(layer_to_move), int(c_idx)))
                
        return {
            'swap': [], # full cluster swaps
            'reassign': reassign_actions # single layer reassignments
        }

    def compute_ttf(self, aging_trajectory: np.ndarray, failure_threshold: float = 0.8) -> float:
        """
        Estimates expected Time To Failure (years) via linear extrapolation if not yet failed.
        """
        if len(aging_trajectory.shape) > 1:
            current_score = np.max(aging_trajectory[-1, :])
        else:
            current_score = np.max(aging_trajectory)
            
        if current_score >= failure_threshold:
            return 0.0
            
        if current_score <= 1e-6:
            return 10.0 # upper bound cap
            
        # Assuming normalized timestep unit = 1 Year for simplicity in the proxy calculation
        # Trajectory rate = current_score / current_time 
        # TTF = Threshold / Rate
        current_time_years = 1.0 # arbitrary reference scaling
        rate = current_score / current_time_years
        
        ttf = failure_threshold / rate
        return float(np.clip(ttf, 0.0, 10.0))
