import numpy as np
from typing import List, Dict

from .nbti_model import NBTIModel
from .hci_model import HCIModel
from .tddb_model import TDDBModel

class AgingLabelGenerator:
    """Combines NBTI + HCI + TDDB into a normalized per-node aging score [0, 1]."""

    MECHANISM_ORDER = ("nbti", "hci", "tddb")

    def __init__(self, nbti=None, hci=None, tddb=None, weights=None, cfg=None):
        if cfg is not None:
            acfg = cfg.get('aging', {})
            voltage_ref_v = acfg.get('voltage_ref_v', 0.8)
            temperature_ref_k = acfg.get('temperature_ref_k', 373.0)

            self.nbti = NBTIModel(
                A=acfg.get('nbti_A', 0.005),
                n=acfg.get('nbti_n', 0.25),
                temperature_K=temperature_ref_k,
                voltage_ref_v=voltage_ref_v,
                voltage_exp=acfg.get('nbti_voltage_exp', 4.0),
                activation_energy_ev=acfg.get('nbti_ea_ev', 0.15),
            )

            self.hci = HCIModel(
                B=acfg.get('hci_B', 0.0001),
                m=acfg.get('hci_m', 0.5),
                time_exp=acfg.get('hci_time_exp', 0.5),
                voltage_ref_v=voltage_ref_v,
                voltage_exp=acfg.get('hci_voltage_exp', 3.0),
                temperature_ref_k=temperature_ref_k,
                activation_energy_ev=acfg.get('hci_ea_ev', 0.10),
            )

            self.tddb = TDDBModel(
                k=acfg.get('tddb_field_gamma', acfg.get('tddb_k', 2.5)),
                beta=acfg.get('tddb_weibull_beta', 2.0),
                eta_ref_s=acfg.get('tddb_eta_ref_s', 5_000_000.0),
                field_ref=acfg.get('tddb_field_ref', voltage_ref_v),
                temperature_ref_k=temperature_ref_k,
                activation_energy_ev=acfg.get('tddb_ea_ev', 0.30),
            )
            self.weights = cfg.get('planning', {})
        else:
            self.nbti = nbti
            self.hci  = hci
            self.tddb = tddb
            self.weights = weights

    @staticmethod
    def _node_array(activity_metrics: dict, key: str, n: int) -> np.ndarray:
        if key not in activity_metrics:
            raise KeyError(f"activity_metrics must include '{key}'")

        arr = np.asarray(activity_metrics[key], dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(f"{key} must have shape ({n},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} must contain only finite values")
        return arr

    @staticmethod
    def _switching_activity(activity_metrics: dict) -> np.ndarray:
        sw_act = np.asarray(activity_metrics["switching_activity"], dtype=np.float64)
        if sw_act.ndim != 1:
            raise ValueError(f"switching_activity must be one-dimensional, got {sw_act.shape}")
        if not np.all(np.isfinite(sw_act)):
            raise ValueError("switching_activity must contain only finite values")
        return sw_act

    @staticmethod
    def _utilization(activity_metrics: dict, switching_activity: np.ndarray) -> np.ndarray:
        n = switching_activity.size
        if all(k in activity_metrics for k in ("mac_utilization", "sram_access_rate", "noc_traffic")):
            util = np.concatenate([
                activity_metrics["mac_utilization"],
                activity_metrics["sram_access_rate"],
                activity_metrics["noc_traffic"],
            ])
        else:
            util = activity_metrics.get("mac_utilization", switching_activity)
            if len(util) != n:
                util = switching_activity

        util = np.asarray(util, dtype=np.float64)
        if util.shape != (n,):
            raise ValueError(f"utilization vector must have shape ({n},), got {util.shape}")
        if not np.all(np.isfinite(util)):
            raise ValueError("utilization vector must contain only finite values")
        return util

    def compute_mechanisms(self, activity_metrics: dict, stress_time_s: float) -> np.ndarray:
        sw_act = self._switching_activity(activity_metrics)
        N = len(sw_act)
        time_arr = np.full(N, stress_time_s, dtype=np.float64)
        util = self._utilization(activity_metrics, sw_act)

        voltage = self._node_array(activity_metrics, 'voltage', N)
        temperature_k = self._node_array(activity_metrics, 'temperature_k', N)

        current_density = sw_act * util
        e_field = sw_act * voltage

        nbti_norm = np.clip(
            self.nbti.compute_degradation(
                time_arr,
                sw_act,
                voltage,
                temperature_k,
            ) / 0.2,
            0,
            1,
        )

        hci_norm = np.clip(
            self.hci.compute_degradation(
                current_density,
                time_arr,
                voltage,
                temperature_k,
            ) / 0.1,
            0,
            1,
        )

        tddb_norm = self.tddb.failure_probability(
            e_field,
            time_arr,
            temperature_k,
        )

        return np.stack([nbti_norm, hci_norm, tddb_norm], axis=1)

    def compute_raw_state(self, activity_metrics: dict, stress_time_s: float) -> np.ndarray:
        """Return raw [NBTI, HCI, TDDB hazard] state under constant stress."""
        if not np.isfinite(stress_time_s) or stress_time_s < 0.0:
            raise ValueError("stress_time_s must be finite and nonnegative")

        sw_act = self._switching_activity(activity_metrics)
        n = sw_act.size
        util = self._utilization(activity_metrics, sw_act)
        voltage = self._node_array(activity_metrics, "voltage", n)
        temperature_k = self._node_array(activity_metrics, "temperature_k", n)
        time_arr = np.full(n, stress_time_s, dtype=np.float64)
        current_density = sw_act * util
        e_field = sw_act * voltage
        eta = self.tddb.characteristic_lifetime(e_field, temperature_k)

        raw_state = np.empty((n, 3), dtype=np.float64)
        raw_state[:, 0] = self.nbti.compute_degradation(
            time_arr, sw_act, voltage, temperature_k
        )
        raw_state[:, 1] = self.hci.compute_degradation(
            current_density, time_arr, voltage, temperature_k
        )
        raw_state[:, 2] = np.power(time_arr / eta, self.tddb.beta)
        if not np.all(np.isfinite(raw_state)) or np.any(raw_state < 0.0):
            raise ValueError("raw state must be finite and nonnegative")
        return raw_state

    @staticmethod
    def initialize_state(num_nodes: int) -> np.ndarray:
        if num_nodes < 0:
            raise ValueError("num_nodes must be nonnegative")
        return np.zeros((num_nodes, 3), dtype=np.float64)

    def step_state(
        self,
        previous_state: np.ndarray,
        activity_metrics: dict,
        delta_t_s: float,
    ) -> np.ndarray:
        if not np.isfinite(delta_t_s) or delta_t_s < 0.0:
            raise ValueError("delta_t_s must be finite and nonnegative")
        sw_act = self._switching_activity(activity_metrics)
        N = len(sw_act)

        state = np.asarray(previous_state, dtype=np.float64)
        if state.shape != (N, 3):
            raise ValueError(
                f"previous_state must have shape ({N}, 3), got {state.shape}"
            )
        if not np.all(np.isfinite(state)):
            raise ValueError("previous_state must contain only finite values")
        if np.any(state < 0.0):
            raise ValueError("previous_state must be nonnegative")

        voltage = self._node_array(activity_metrics, "voltage", N)
        temperature_k = self._node_array(activity_metrics, "temperature_k", N)

        util = self._utilization(activity_metrics, sw_act)

        current_density = sw_act * util
        e_field = sw_act * voltage

        new_state = state.copy()
        new_state[:, 0] = self.nbti.accumulate(
            state[:, 0],
            sw_act,
            delta_t_s,
            voltage,
            temperature_k,
        )
        new_state[:, 1] = self.hci.accumulate(
            state[:, 1],
            current_density,
            delta_t_s,
            voltage,
            temperature_k,
        )
        new_state[:, 2] = self.tddb.accumulate_hazard(
            state[:, 2],
            e_field,
            delta_t_s,
            temperature_k,
        )
        if not np.all(np.isfinite(new_state)) or np.any(new_state < 0.0):
            raise ValueError("updated state must be finite and nonnegative")
        return new_state

    def state_to_mechanisms(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 3:
            raise ValueError(f"state must have shape [N, 3], got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("state must contain only finite values")
        if np.any(state < 0.0):
            raise ValueError("state must be nonnegative")

        nbti_norm = np.clip(state[:, 0] / 0.2, 0.0, 1.0)
        hci_norm = np.clip(state[:, 1] / 0.1, 0.0, 1.0)
        tddb_prob = self.tddb.hazard_to_probability(state[:, 2])

        return np.stack([nbti_norm, hci_norm, tddb_prob], axis=1)

    def combine_mechanisms(self, mechanisms: np.ndarray) -> np.ndarray:
        if mechanisms.ndim != 2 or mechanisms.shape[1] != 3:
            raise ValueError(
                f"mechanisms must have shape [N, 3], got {mechanisms.shape}"
            )

        score = (
            self.weights.get('nbti', 0.4) * mechanisms[:, 0] +
            self.weights.get('hci', 0.4) * mechanisms[:, 1] +
            self.weights.get('tddb', 0.2) * mechanisms[:, 2]
        )
        return np.clip(score, 0.0, 1.0)

    def compute_aging_score(self, activity_metrics: dict, stress_time_s: float) -> np.ndarray:
        mechanisms = self.compute_mechanisms(activity_metrics, stress_time_s)
        return self.combine_mechanisms(mechanisms)

    def generate_trajectory_labels(
        self,
        activity_sequence: List[dict],
        timestep_s: float,
        initial_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sequentially advance raw state and return composite labels [T, N]."""
        if not activity_sequence:
            raise ValueError("activity_sequence must not be empty")
        T = len(activity_sequence)
        N = len(activity_sequence[0]['switching_activity'])
        state = self.initialize_state(N) if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
        if state.shape != (N, 3):
            raise ValueError(f"initial_state must have shape ({N}, 3), got {state.shape}")
        trajectories = np.zeros((T, N), dtype=np.float64)
        for t in range(T):
            state = self.step_state(state, activity_sequence[t], timestep_s)
            trajectories[t] = self.combine_mechanisms(self.state_to_mechanisms(state))
        return trajectories
