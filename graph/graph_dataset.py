# graph/graph_dataset.py — COMPLETE IMPLEMENTATION
"""
PyTorch Geometric InMemoryDataset for aging graph samples.
Compatible with torch_geometric==2.7.0
"""

from __future__ import annotations
import logging
import random
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

logger = logging.getLogger(__name__)


class AgingDataset(InMemoryDataset):
    """
    Dataset of accelerator hardware graphs labeled with aging scores.

    Each Data object contains:
      x              : node features [N, 8]   float32
      edge_index     : graph edges   [2, E]   long
      edge_attr      : edge features [E, 2]   float32
      y              : aging score   [N, 1]   float32
      y_mechanisms   : [NBTI,HCI,TDDB] [N, 3] float32
      x_history      : past node features [N, T, 8] float32
      y_trajectory   : future aging  [N, k]   float32
      y_mechanism_trajectory : future mechanisms [N, k, 3] float32
      workload_emb   : one-hot       [5]      float32
      latency        : scalar        [1]      float32
      energy         : scalar        [1]      float32
    """

    WORKLOAD_LIST = [
        "ResNet-50", "MobileNetV2", "EfficientNet-B4", "BERT-Base", "ViT-B/16"
    ]
    FEATURE_DIM = 8
    FEATURE_SCHEMA_VERSION = 3
    LABEL_SCHEMA_VERSION = 8
    # Four real simulator intervals provide a compact causal context while
    # preserving the existing ten-step prediction horizon by default.
    HISTORY_LENGTH = 4
    HORIZON = 10
    SECONDS_PER_STEP = 3600.0
    DEFAULT_CFG = {
        "accelerator": {
            "pe_array": [4, 4],
            "pe_array_rows": 4,
            "pe_array_cols": 4,
            "mac_clusters": 16,
            "sram_banks": 8,
            "noc_routers": 4,
            "num_layers": 10,
            "voltage_v": 0.8,
            "supply_voltage": 0.8,
            "temperature_k": 373.0,
            "ambient_temperature_k": 318.15,
            "thermal_package_rth_k_per_w": 1.5,
            "thermal_local_rth_k_per_w": 60.0,
            "max_temperature_k": 398.15,
        },
        "workloads": [],
        "aging": {
            "voltage_ref_v": 0.8,
            "temperature_ref_k": 373.0,
            "nbti_A": 0.005,
            "nbti_n": 0.25,
            "nbti_voltage_exp": 4.0,
            "nbti_ea_ev": 0.15,
            "hci_B": 0.0001,
            "hci_m": 0.5,
            "hci_time_exp": 0.5,
            "hci_voltage_exp": 3.0,
            "hci_ea_ev": 0.10,
            "tddb_field_gamma": 2.5,
            "tddb_weibull_beta": 2.0,
            "tddb_eta_ref_s": 5_000_000.0,
            "tddb_field_ref": 0.8,
            "tddb_ea_ev": 0.30,
        },
        "planning": {
            "failure_threshold": 0.8,
            "nbti": 0.40,
            "hci": 0.35,
            "tddb": 0.25,
        },
        "model": {
            "prediction_horizon": 10,
        },
    }

    def __init__(
        self,
        root: str,
        split: str,          # 'train' | 'val' | 'test'
        size: int,
        cfg: Optional[DictConfig] = None,
        config: Optional[DictConfig] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        user_cfg = cfg if cfg is not None else config
        user_cfg = user_cfg if user_cfg is not None else {}
        self._auto_generate = self._has_required_sections(user_cfg)
        self.split = split
        self.size = size
        self.cfg = OmegaConf.merge(OmegaConf.create(self.DEFAULT_CFG), OmegaConf.create(user_cfg))
        self.seed = seed
        self._acc_cfg = self.cfg.accelerator
        self.horizon = int(self.cfg.model.get("prediction_horizon", self.HORIZON))
        super().__init__(root=root, transform=transform)
        # PyG 2.7.0: load() replaces the old from_data_list pattern
        self.load(self.processed_paths[0])
        self._dynamic_samples: List[Data] = [self.get(i) for i in range(len(self))]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        n_mac = int(self._acc_cfg.get("mac_clusters", self._acc_cfg.get("num_mac_clusters", 64)))
        return [
            f"aging_{self.split}_{self.size}_mac{n_mac}_"
            f"feat{self.FEATURE_DIM}_fv{self.FEATURE_SCHEMA_VERSION}_"
            f"labels_v{self.LABEL_SCHEMA_VERSION}.pt"
        ]

    def download(self) -> None:
        pass  # Synthetic dataset — no download needed

    def process(self) -> None:
        """
        Generate all graph samples and save as PyG dataset.
        Runs once; cached on disk afterward.
        """
        if not self._auto_generate or self.size <= 0:
            self._save_empty_dataset()
            logger.info("Initialized empty dataset cache at %s", self.processed_paths[0])
            return

        # Lazy imports to avoid circular deps
        from simulator.timeloop_runner import TimeloopRunner
        from simulator.workload_runner import WorkloadRunner
        from features.feature_builder import FeatureBuilder
        from aging_models.aging_label_generator import AgingLabelGenerator
        from graph.accelerator_graph import AcceleratorGraph

        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        simulator = TimeloopRunner(self._acc_cfg)
        workload_runner = WorkloadRunner(self.cfg.workloads)
        feature_builder = FeatureBuilder(self._acc_cfg)
        aging_gen = AgingLabelGenerator(cfg=self.cfg)
        acc_graph = AcceleratorGraph(self._acc_cfg)
        acc_graph.build()
        num_nodes = acc_graph.get_num_nodes()

        data_list: List[Data] = []
        logger.info(f"Generating {self.size} samples [{self.split}]...")

        num_macs = int(self._acc_cfg.get("mac_clusters", self._acc_cfg.get("num_mac_clusters", 64)))
        voltage_v = float(self._acc_cfg.get("voltage_v", self._acc_cfg.get("supply_voltage", 0.8)))

        for idx in tqdm(range(self.size), desc=f"AgingDataset[{self.split}]"):
            # Initial age is applied once as raw preconditioning, never as a
            # delta time.  Subsequent state updates use SECONDS_PER_STEP only.
            initial_age_s = float(rng.uniform(3600, 1_800_000))
            workload_start = int(rng.integers(0, len(self.WORKLOAD_LIST)))
            state = None
            history_features: list[torch.Tensor] = []
            history_temperatures: list[np.ndarray] = []
            future_mechanisms: list[np.ndarray] = []
            future_composites: list[np.ndarray] = []
            cutoff_result = cutoff_mapping = cutoff_mechanisms = None
            cutoff_workload = ""

            # A seeded cycle through the existing workload pool guarantees
            # temporal variation without inventing a separate scheduler.
            for step in range(self.HISTORY_LENGTH + self.horizon):
                wl_idx = (workload_start + step) % len(self.WORKLOAD_LIST)
                wl_name = self.WORKLOAD_LIST[wl_idx]
                layers = workload_runner.get_workload_layers(wl_name)
                mapping = rng.integers(0, max(num_macs, 1), size=len(layers)).astype(np.int32)
                result = simulator.run_workload(layers, mapping)

                temperature_k = np.asarray(result.temperature_k, dtype=np.float32)
                if temperature_k.shape != (num_nodes,):
                    raise ValueError(
                        f"simulator temperature_k must have shape ({num_nodes},), "
                        f"got {temperature_k.shape}"
                    )
                activity = {
                    "switching_activity": result.switching_activity,
                    "mac_utilization": result.mac_utilization,
                    "sram_access_rate": result.sram_access_rate,
                    "noc_traffic": result.noc_traffic,
                    "voltage": np.full(num_nodes, voltage_v, dtype=np.float32),
                    "temperature_k": temperature_k,
                }

                if state is None:
                    state = aging_gen.compute_raw_state(activity, initial_age_s)

                # The feature age represents the device age at this interval's
                # endpoint, matching the persistent state after this update.
                elapsed_age_s = initial_age_s + (step + 1) * self.SECONDS_PER_STEP
                node_features = feature_builder.build_node_features(
                    activity_dict=activity,
                    workload_name=wl_name,
                    latency=result.total_latency_cycles,
                    energy=result.total_energy_pj,
                    stress_time_s=elapsed_age_s,
                )
                state = aging_gen.step_state(state, activity, self.SECONDS_PER_STEP)
                mechanisms = aging_gen.state_to_mechanisms(state)

                if step < self.HISTORY_LENGTH:
                    history_features.append(node_features)
                    history_temperatures.append(temperature_k)
                    if step == self.HISTORY_LENGTH - 1:
                        cutoff_result = result
                        cutoff_mapping = mapping
                        cutoff_mechanisms = mechanisms
                        cutoff_workload = wl_name
                else:
                    future_mechanisms.append(mechanisms)
                    future_composites.append(aging_gen.combine_mechanisms(mechanisms))

            assert cutoff_result is not None and cutoff_mapping is not None and cutoff_mechanisms is not None
            x_history = torch.stack(history_features, dim=1)  # [N, T, 8]
            node_features = x_history[:, -1, :]
            temperature_history_k = torch.tensor(
                np.stack(history_temperatures, axis=1), dtype=torch.float32
            )
            mechanisms = cutoff_mechanisms
            aging_score = aging_gen.combine_mechanisms(mechanisms)
            y_mechanism_trajectory = torch.tensor(
                np.stack(future_mechanisms, axis=1), dtype=torch.float32
            )  # [N, H, 3]
            y_trajectory = torch.tensor(
                np.stack(future_composites, axis=1), dtype=torch.float32
            )  # [N, H]
            result = cutoff_result
            mapping = cutoff_mapping
            wl_name = cutoff_workload
            wl_idx = self.WORKLOAD_LIST.index(wl_name)
            stress_time = initial_age_s + self.HISTORY_LENGTH * self.SECONDS_PER_STEP

            # --- Build PyG graph (edge_index must be [2, E] long) ---
            pyg_data = acc_graph.to_pyg(node_features)
            # Defensive assertion
            assert pyg_data.edge_index.shape[0] == 2, \
                f"edge_index shape error: {pyg_data.edge_index.shape}"
            assert pyg_data.edge_index.dtype == torch.long, \
                f"edge_index dtype error: {pyg_data.edge_index.dtype}"

            # --- Workload one-hot [5] ---
            wl_emb = torch.zeros(len(self.WORKLOAD_LIST), dtype=torch.float32)
            wl_emb[wl_idx] = 1.0

            # --- Normalized scalars ---
            lat_norm = float(min(result.total_latency_cycles / 1e8, 1.0))
            eng_norm = float(min(result.total_energy_pj / 1e9, 1.0))

            # --- Mapping vector (capped at 64 entries) ---
            max_map_len = 64
            map_arr = mapping[:max_map_len].astype(np.float32)
            map_arr = map_arr / max(num_macs - 1, 1)
            if len(map_arr) < max_map_len:
                map_arr = np.pad(map_arr, (0, max_map_len - len(map_arr)))
            mapping_tensor = torch.tensor(map_arr, dtype=torch.float32)

            data = Data(
                x=pyg_data.x,                          # [N, 8]
                x_history=x_history,                   # [N, HISTORY_LENGTH, 8]
                temperature_history_k=temperature_history_k,  # [N, HISTORY_LENGTH]
                edge_index=pyg_data.edge_index,        # [2, E] long
                edge_attr=pyg_data.edge_attr,          # [E, 2]
                y=torch.tensor(
                    aging_score, dtype=torch.float32
                ).unsqueeze(1),                        # [N, 1]
                y_mechanisms=torch.tensor(
                    mechanisms, dtype=torch.float32
                ),                                     # [N, 3]
                y_trajectory=y_trajectory,             # [N, HORIZON]
                y_mechanism_trajectory=y_mechanism_trajectory,  # [N, HORIZON, 3]
                workload_emb=wl_emb,                   # [5]
                mapping=mapping_tensor,                # [64]
                stress_time=torch.tensor([stress_time], dtype=torch.float32),
                latency=torch.tensor([lat_norm]),      # [1]
                energy=torch.tensor([eng_norm]),       # [1]
                num_nodes=num_nodes,                   # explicit — required by PyG 2.7
            )
            data_list.append(data)

        # PyG 2.7.0 save API
        self.save(data_list, self.processed_paths[0])
        logger.info(
            f"Saved {len(data_list)} samples → {self.processed_paths[0]}"
        )

    def add_sample(self, data: Data) -> None:
        self._dynamic_samples.append(data)

    def finalize_and_save(self) -> None:
        if self._dynamic_samples:
            self.save(self._dynamic_samples, self.processed_paths[0])
        else:
            self._save_empty_dataset()
        self.load(self.processed_paths[0])
        self._dynamic_samples = [self.get(i) for i in range(len(self))]

    @staticmethod
    def _has_required_sections(cfg: Any) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return "accelerator" in cfg and "workloads" in cfg
        try:
            return "accelerator" in cfg and "workloads" in cfg
        except TypeError:
            return hasattr(cfg, "accelerator") and hasattr(cfg, "workloads")

    def _save_empty_dataset(self) -> None:
        torch.save((Data().to_dict(), {}, Data), self.processed_paths[0])

# Alias for backward compatibility
AcceleratorGraphDataset = AgingDataset
