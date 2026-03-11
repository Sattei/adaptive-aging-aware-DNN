# graph/graph_dataset.py — COMPLETE IMPLEMENTATION
"""
PyTorch Geometric InMemoryDataset for aging graph samples.
Compatible with torch_geometric==2.7.0
"""

from __future__ import annotations
import logging
import random
from pathlib import Path
from typing import List, Optional, Callable
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AgingDataset(InMemoryDataset):
    """
    Dataset of accelerator hardware graphs labeled with aging scores.

    Each Data object contains:
      x              : node features [N, 7]   float32
      edge_index     : graph edges   [2, E]   long
      edge_attr      : edge features [E, 2]   float32
      y              : aging score   [N, 1]   float32
      y_trajectory   : future aging  [N, k]   float32
      workload_emb   : one-hot       [5]      float32
      latency        : scalar        [1]      float32
      energy         : scalar        [1]      float32
    """

    WORKLOAD_LIST = [
        "ResNet50", "MobileNetV2", "EfficientNetB4", "BERTBase", "ViTB16"
    ]
    FEATURE_DIM = 7
    HORIZON = 10
    SECONDS_PER_STEP = 3600.0

    def __init__(
        self,
        root: str,
        split: str,          # 'train' | 'val' | 'test'
        size: int,
        cfg: DictConfig,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.size = size
        self.cfg = cfg
        self.seed = seed
        self._acc_cfg = cfg.accelerator
        super().__init__(root=root, transform=transform)
        # PyG 2.7.0: load() replaces the old from_data_list pattern
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        n_mac = self._acc_cfg.num_mac_clusters
        return [f"aging_{self.split}_{self.size}_mac{n_mac}.pt"]

    def download(self) -> None:
        pass  # Synthetic dataset — no download needed

    def process(self) -> None:
        """
        Generate all graph samples and save as PyG dataset.
        Runs once; cached on disk afterward.
        """
        # Lazy imports to avoid circular deps
        from simulator.timeloop_runner import TimeloopRunner
        from simulator.workload_runner import WorkloadRunner, WORKLOAD_SPECS
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

        for idx in tqdm(range(self.size), desc=f"AgingDataset[{self.split}]"):
            # --- Select workload ---
            wl_name = self.WORKLOAD_LIST[int(rng.integers(0, len(self.WORKLOAD_LIST)))]
            wl_idx = self.WORKLOAD_LIST.index(wl_name)
            layers = workload_runner.get_workload_layers(wl_name)
            n_layers = len(layers)

            # --- Random mapping ---
            mapping = rng.integers(
                0, self._acc_cfg.num_mac_clusters, size=n_layers
            ).astype(np.int32)

            # --- Simulate ---
            result = simulator.run_workload(layers, mapping)
            activity = {
                "switching_activity": result.switching_activity,
                "mac_utilization":    result.mac_utilization,
                "sram_access_rate":   result.sram_access_rate,
                "noc_traffic":        result.noc_traffic,
            }

            # --- Node features [N, 7] ---
            node_features = feature_builder.build_node_features(
                activity_dict=activity,
                workload_name=wl_name,
                latency=result.total_latency_cycles,
                energy=result.total_energy_pj,
            )

            # --- Current aging score [N] ---
            aging_score = aging_gen.compute_aging_score(
                activity, self.SECONDS_PER_STEP
            )

            # --- Future trajectory [N, HORIZON] ---
            future_acts = []
            for h in range(self.HORIZON):
                noise = rng.normal(0, 0.03, size=result.switching_activity.shape)
                future_act = float(h + 1) / self.HORIZON
                future_acts.append({
                    "switching_activity": np.clip(
                        result.switching_activity * (1.0 + future_act * 0.2) + noise,
                        0.0, 1.0
                    ).astype(np.float32),
                    "mac_utilization":    result.mac_utilization,
                    "sram_access_rate":   result.sram_access_rate,
                    "noc_traffic":        result.noc_traffic,
                })
            trajectory = aging_gen.generate_trajectory_labels(
                future_acts, self.SECONDS_PER_STEP
            )  # [HORIZON, N]

            # Transpose: [HORIZON, N] → [N, HORIZON]
            y_trajectory = torch.tensor(
                trajectory.T, dtype=torch.float32
            )  # [N, HORIZON]

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
            map_arr = map_arr / max(self._acc_cfg.num_mac_clusters - 1, 1)
            if len(map_arr) < max_map_len:
                map_arr = np.pad(map_arr, (0, max_map_len - len(map_arr)))
            mapping_tensor = torch.tensor(map_arr, dtype=torch.float32)

            data = Data(
                x=pyg_data.x,                          # [N, 7]
                edge_index=pyg_data.edge_index,        # [2, E] long
                edge_attr=pyg_data.edge_attr,          # [E, 2]
                y=torch.tensor(
                    aging_score, dtype=torch.float32
                ).unsqueeze(1),                        # [N, 1]
                y_trajectory=y_trajectory,             # [N, HORIZON]
                workload_emb=wl_emb,                   # [5]
                mapping=mapping_tensor,                # [64]
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
