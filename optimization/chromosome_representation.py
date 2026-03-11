import numpy as np

class MappingChromosome:
    """
    Encodes mapping decisions (Layer -> Cluster).
    """
    def __init__(self, num_layers: int, num_clusters: int):
        self.num_layers = num_layers
        self.num_clusters = num_clusters
        
    def random_init(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, self.num_clusters, size=self.num_layers)
        
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """Single point crossover."""
        pt = np.random.randint(1, self.num_layers)
        c1 = np.concatenate([parent1[:pt], parent2[pt:]])
        c2 = np.concatenate([parent2[:pt], parent1[pt:]])
        return c1, c2
        
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Random reset mutation."""
        c_new = chromosome.copy()
        mask = np.random.rand(self.num_layers) < mutation_rate
        # Generate new random genes for mutated loci
        new_genes = np.random.randint(0, self.num_clusters, size=np.sum(mask))
        c_new[mask] = new_genes
        return c_new
        
    def is_valid(self, chromosome: np.ndarray, constraints: dict) -> bool:
        """
        Validates if structural dependency limits are met.
        For simple spatial mapping, all bounds [0, C-1] are valid.
        """
        if np.any((chromosome < 0) | (chromosome >= self.num_clusters)):
            return False
            
        return True
        
    def repair(self, chromosome: np.ndarray, constraints: dict) -> np.ndarray:
        """
        Clamps to valid cluster range.
        """
        return np.clip(chromosome, 0, self.num_clusters - 1)
