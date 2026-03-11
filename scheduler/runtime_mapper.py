import numpy as np
from optimization.chromosome_representation import MappingChromosome

class RuntimeMapper:
    """
    Applies optimization results to runtime scheduler structures.
    """
    def __init__(self, config):
        self.config = config
        
    def dispatch(self, workload_layers: list, mapping: np.ndarray) -> dict:
        """
        Creates an actionable execution trace from an optimal layer map.
        Returns: Trace dictionary
        """
        assert len(workload_layers) == len(mapping), "Mismatch in layers to bindings"
        
        trace = []
        for i, layer in enumerate(workload_layers):
            action = {
                'layer_idx': i,
                'target_cluster': int(mapping[i]),
                'layer_config': layer
            }
            trace.append(action)
            
        return {'status': 'success', 'trace': trace}
