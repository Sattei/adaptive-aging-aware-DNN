import torch
import numpy as np

class FeatureBuilder:
    """
    Builds the [N x 7] node feature matrix for PyTorch Geometric datasets.
    """
    def __init__(self, acc_cfg):
        """
        Args:
            acc_cfg: Accelerator config dict
        """
        self.acc_cfg = acc_cfg
        self.num_macs = acc_cfg.get('num_mac_clusters', 64)
        self.num_srams = acc_cfg.get('num_sram_banks', 16)
        self.num_routers = acc_cfg.get('num_noc_routers', 8)
        self.N = self.num_macs + self.num_srams + self.num_routers
        
    def build_node_features(self, activity_dict: dict, workload_name: str, latency: float, energy: float) -> torch.Tensor:
        """
        Constructs the 7-dimensional feature matrix for all N nodes.
        
        Features:
          0: is_mac (0/1)
          1: is_sram (0/1)
          2: is_router (0/1)
          3: switching_activity
          4: component_utilization
          5: normalized_latency
          6: normalized_energy
          
        Args:
            activity_dict: keys [switching_activity, mac_utilization, sram_access_rate, noc_traffic]
            workload_name: string (currently unused in node feats since graph_dataset adds workload_emb globally)
            latency: scalar
            energy: scalar
            
        Returns:
            torch.Tensor shape [N, 7]
        """
        features = torch.zeros((self.N, 7), dtype=torch.float32)
        idx = 0
        sw_act = activity_dict["switching_activity"]
        mac_util = activity_dict["mac_utilization"]
        sram_util = activity_dict["sram_access_rate"]
        noc_util = activity_dict["noc_traffic"]
        
        # normalize latency and energy
        lat_norm = float(min(latency / 1e8, 1.0))
        eng_norm = float(min(energy / 1e9, 1.0))
        
        # MACs
        for i in range(self.num_macs):
            if idx < len(sw_act) and i < len(mac_util):
                features[idx, 0] = 1.0
                features[idx, 3] = float(sw_act[idx])
                features[idx, 4] = float(mac_util[i])
                features[idx, 5] = lat_norm
                features[idx, 6] = eng_norm
            idx += 1
            
        # SRAMs
        for i in range(self.num_srams):
            if idx < len(sw_act) and i < len(sram_util):
                features[idx, 1] = 1.0
                features[idx, 3] = float(sw_act[idx])
                features[idx, 4] = float(sram_util[i])
                features[idx, 5] = lat_norm
                features[idx, 6] = eng_norm
            idx += 1
            
        # Routers
        for i in range(self.num_routers):
            if idx < len(sw_act) and i < len(noc_util):
                features[idx, 2] = 1.0
                features[idx, 3] = float(sw_act[idx])
                features[idx, 4] = float(noc_util[i])
                features[idx, 5] = lat_norm
                features[idx, 6] = eng_norm
            idx += 1
            
        return features
