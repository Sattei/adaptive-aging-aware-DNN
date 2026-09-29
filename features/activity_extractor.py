import numpy as np
from typing import Dict, Any
from simulator.timeloop_runner import WorkloadResult

class ActivityExtractor:
    """
    Extracts and normalizes hardware node activities from simulator outputs.
    """
    def __init__(self, accelerator_config: Any):
        self.config = accelerator_config
        self.num_clusters = accelerator_config.get('mac_clusters', 64)
        self.num_banks = accelerator_config.get('sram_banks', 16)
        self.num_routers = accelerator_config.get('noc_routers', 8)
        
    def extract_activities(self, sim_data: WorkloadResult, workload: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Takes raw simulation output and normalizes it to node features.
        
        Args:
            sim_data: simulator.timeloop_runner.WorkloadResult containing average simulated metrics
            workload: Dict workload profile parameters
            
        Returns:
            Dict of standardized feature traces bounded [0, 1] except temperature.
        """
        # Ensure we always deal with arrays of the right shape
        if isinstance(sim_data.avg_switching_activity, np.ndarray):
            sw_act = sim_data.avg_switching_activity
        else:
            sw_act = np.zeros(self.num_clusters + self.num_banks + self.num_routers)
            
        # Mac clusters
        mac_util = sim_data.avg_mac_utilization if hasattr(sim_data, 'avg_mac_utilization') else np.zeros(self.num_clusters)
        
        # Sram banks
        sram_access = sim_data.avg_sram_access_rate if hasattr(sim_data, 'avg_sram_access_rate') else np.zeros(self.num_banks)
        
        # Routers
        noc_activity = sim_data.avg_noc_traffic if hasattr(sim_data, 'avg_noc_traffic') else np.zeros(self.num_routers)
        
        # Prefer the simulator's Stage-2C thermal state.
        n_nodes = self.num_clusters + self.num_banks + self.num_routers
        sim_temp_k = getattr(sim_data, "temperature_k", None)
        if isinstance(sim_temp_k, np.ndarray) and sim_temp_k.shape == (n_nodes,):
            temperature_k = sim_temp_k.astype(np.float32)
        else:
            # Compatibility fallback for older SimResult objects.
            mac_temp_c = 30.0 + (50.0 * mac_util)
            sram_temp_c = 35.0 + (30.0 * sram_access)
            noc_temp_c = 30.0 + (25.0 * noc_activity)
            temperature_k = (
                np.concatenate([mac_temp_c, sram_temp_c, noc_temp_c]) + 273.15
            ).astype(np.float32)

        mac_temp_k = temperature_k[:self.num_clusters]
        sram_temp_k = temperature_k[
            self.num_clusters:self.num_clusters + self.num_banks
        ]
        noc_temp_k = temperature_k[self.num_clusters + self.num_banks:]
        
        return {
            "mac_switching": sw_act[:self.num_clusters],
            "mac_utilization": mac_util,
            "mac_temperature": mac_temp_k - 273.15,
            "mac_temperature_k": mac_temp_k,
            "sram_access": sram_access,
            "sram_temperature": sram_temp_k - 273.15,
            "sram_temperature_k": sram_temp_k,
            "noc_activity": noc_activity,
            "noc_temperature": noc_temp_k - 273.15,
            "noc_temperature_k": noc_temp_k,
            "global_temperature_k": temperature_k,
            "global_switching": sw_act
        }
