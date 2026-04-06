import numpy as np
from typing import List, Dict

from .nbti_model import NBTIModel
from .hci_model import HCIModel
from .tddb_model import TDDBModel

class AgingLabelGenerator:
    """Combines NBTI + HCI + TDDB into a normalized per-node aging score [0, 1]."""

    def __init__(self, nbti=None, hci=None, tddb=None, weights=None, cfg=None):
        if cfg is not None:
            acfg = cfg.get('aging', {})
            self.nbti = NBTIModel(A=acfg.get('nbti_A', 0.005), n=acfg.get('nbti_n', 0.25))
            self.hci  = HCIModel(B=acfg.get('hci_B', 0.0001), m=acfg.get('hci_m', 0.5))
            self.tddb = TDDBModel(k=acfg.get('tddb_k', 2.5), beta=acfg.get('tddb_beta', 10.0))
            self.weights = cfg.get('planning', {})
        else:
            self.nbti = nbti
            self.hci  = hci
            self.tddb = tddb
            self.weights = weights

    def compute_aging_score(self, activity_metrics: dict, stress_time_s: float) -> np.ndarray:
        N = len(activity_metrics['switching_activity'])
        time_arr = np.full(N, stress_time_s)
        sw_act = activity_metrics['switching_activity']

        if all(k in activity_metrics for k in ('mac_utilization', 'sram_access_rate', 'noc_traffic')):
            util = np.concatenate([
                activity_metrics['mac_utilization'],
                activity_metrics['sram_access_rate'],
                activity_metrics['noc_traffic'],
            ])
        else:
            util = activity_metrics.get('mac_utilization', sw_act)
            if len(util) != len(sw_act):
                util = sw_act

        voltage = activity_metrics.get('voltage', np.ones(N) * 0.8)
        current_density = sw_act * util
        e_field = sw_act * voltage

        nbti_norm = np.clip(self.nbti.compute_degradation(time_arr, sw_act) / 0.2, 0, 1)
        hci_norm  = np.clip(self.hci.compute_degradation(current_density, time_arr) / 0.1, 0, 1)
        tddb_norm = self.tddb.failure_probability(e_field, time_arr)

        score = (
            self.weights.get('nbti', 0.4) * nbti_norm +
            self.weights.get('hci',  0.4) * hci_norm  +
            self.weights.get('tddb', 0.2) * tddb_norm
        )
        return np.clip(score, 0.0, 1.0)

    def generate_trajectory_labels(self, activity_sequence: List[dict], timestep_s: float) -> np.ndarray:
        T = len(activity_sequence)
        N = len(activity_sequence[0]['switching_activity'])
        trajectories = np.zeros((T, N))
        cumulative_time = 0.0
        for t in range(T):
            cumulative_time += timestep_s
            trajectories[t] = self.compute_aging_score(activity_sequence[t], cumulative_time)
        return trajectories

