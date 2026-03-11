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
        
        # Temperature Proxy (proxying switching power logic)
        # Simplified: Base + scalar * activity
        mac_temp = 30.0 + (50.0 * mac_util)
        sram_temp = 35.0 + (30.0 * sram_access)
        noc_temp = 30.0 + (25.0 * noc_activity)
        
        return {
            "mac_switching": sw_act[:self.num_clusters],
            "mac_utilization": mac_util,
            "mac_temperature": mac_temp,
            "sram_access": sram_access,
            "sram_temperature": sram_temp,
            "noc_activity": noc_activity,
            "noc_temperature": noc_temp,
            "global_switching": sw_act
        }
