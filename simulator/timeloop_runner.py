import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
from omegaconf import DictConfig

@dataclass
class LayerResult:
    latency_cycles: int
    energy_pj: float
    mac_utilization: np.ndarray
    sram_access_rate: np.ndarray
    noc_traffic: np.ndarray
    switching_activity: np.ndarray
    total_macs: int = 0
    total_memory_accesses: int = 0

@dataclass
class WorkloadResult:
    total_latency_cycles: int
    total_energy_pj: float
    mac_utilization: np.ndarray
    sram_access_rate: np.ndarray
    noc_traffic: np.ndarray
    switching_activity: np.ndarray

class TimeloopRunner:
    """
    Analytical accelerator simulator.
    """
    def __init__(self, accelerator_config: DictConfig):
        self.config = accelerator_config
        # Extract constants
        pe_dims = self.config.get('pe_array', [64, 64])
        self.total_pes = int(pe_dims[0] * pe_dims[1])
        
        self.num_clusters = self.config.get('mac_clusters', self.config.get('num_mac_clusters', 64))
        self.num_banks = self.config.get('sram_banks', self.config.get('num_sram_banks', 16))
        self.num_routers = self.config.get('noc_routers', self.config.get('num_noc_routers', 8))
        
        # Energy constants
        self.mac_energy_pj = self.config.get('mac_energy_pj_per_op', 0.1)
        self.sram_read_energy_pj = self.config.get('sram_read_energy_pj', 2.0)
        self.noc_hop_energy_pj = self.config.get('noc_hop_energy_pj', 0.5)
        self.ops_per_cycle = self.config.get('ops_per_cycle', 2)
        self.clock_ghz = self.config.get('clock_ghz', 1.0)
        self.max_macs_per_cluster = self.config.get('max_macs_per_cluster', 256)
        
        # Physical constraints
        self.pes_per_cluster = self.total_pes // max(self.num_clusters, 1)
        self.num_mac_clusters = self.num_clusters
        self.num_sram_banks = self.num_banks
        self.num_noc_routers = self.num_routers
        
        # Derived nodes total = macs + srams + routers
        self.num_nodes_total = self.num_clusters + self.num_banks + self.num_routers

    def run_layer(self, layer_cfg: dict, mapping: np.ndarray) -> "LayerResult":
        """
        Simulate one layer. Returns LayerResult with all activity metrics.
        mapping: int array of cluster assignments for this workload's layers.
        """
        layer_type = layer_cfg.get("type", "conv2d")
    
        # --- Compute total MACs and memory accesses ---
        if layer_type == "conv2d":
            K = int(layer_cfg.get("K", 64))
            C = int(layer_cfg.get("C", 3))
            R = int(layer_cfg.get("R", 3))
            S = int(layer_cfg.get("S", 3))
            P = int(layer_cfg.get("P", 56))
            Q = int(layer_cfg.get("Q", 56))
            total_macs = K * C * R * S * P * Q
            total_mem  = C * R * S * P * Q + K * P * Q + K * C * R * S
        elif layer_type == "matmul":
            M = int(layer_cfg.get("M", 512))
            Kd = int(layer_cfg.get("K", 768))
            N = int(layer_cfg.get("N", 768))
            total_macs = M * Kd * N
            total_mem  = M * Kd + Kd * N + M * N
        elif layer_type == "depthwise_conv":
            C = int(layer_cfg.get("C", 64))
            R = int(layer_cfg.get("R", 3))
            S = int(layer_cfg.get("S", 3))
            P = int(layer_cfg.get("P", 56))
            Q = int(layer_cfg.get("Q", 56))
            total_macs = C * R * S * P * Q
            total_mem  = C * R * S * P * Q + C * P * Q
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
        total_macs = max(total_macs, 1)
        total_mem  = max(total_mem,  1)
    
        # --- Distribute MACs across clusters based on mapping ---
        mapping_clipped = np.clip(
            np.asarray(mapping, dtype=np.int32), 0, self.num_mac_clusters - 1
        )
        cluster_counts = np.bincount(
            mapping_clipped, minlength=self.num_mac_clusters
        ).astype(np.float64)
        cluster_fracs  = cluster_counts / (cluster_counts.sum() + 1e-8)
        macs_per_cluster = cluster_fracs * total_macs
        mac_utilization  = np.clip(
            macs_per_cluster / (self.max_macs_per_cluster + 1e-8), 0.0, 1.0
        ).astype(np.float32)
    
        # --- Roofline latency ---
        ops_per_s = self.ops_per_cycle * self.clock_ghz * 1e9
        compute_cycles = int(np.ceil(total_macs / ops_per_s))
        # Memory bound: assume 64-bit (8 byte) words
        mem_bw_bytes_per_cycle = (
            self.num_sram_banks * 64.0 * 1e9 / 8 / (self.clock_ghz * 1e9)
        )
        mem_cycles = int(np.ceil((total_mem * 4) / (mem_bw_bytes_per_cycle + 1e-8)))
        latency_cycles = max(compute_cycles, mem_cycles, 1)
    
        # --- Energy ---
        mac_energy_pj  = self.mac_energy_pj   * total_macs
        sram_energy_pj = self.sram_read_energy_pj * total_mem
        noc_traffic_ops = total_mem * 0.12
        noc_energy_pj  = self.noc_hop_energy_pj * noc_traffic_ops * 2.0
        total_energy_pj = mac_energy_pj + sram_energy_pj + noc_energy_pj
    
        # --- SRAM access distribution ---
        sram_access_rate = np.zeros(self.num_sram_banks, dtype=np.float32)
        for k in range(self.num_mac_clusters):
            if mac_utilization[k] > 0:
                b1 = (k * self.num_sram_banks // self.num_mac_clusters) % self.num_sram_banks
                b2 = (b1 + 1) % self.num_sram_banks
                sram_access_rate[b1] += mac_utilization[k] * 0.7
                sram_access_rate[b2] += mac_utilization[k] * 0.3
        sram_max = sram_access_rate.max()
        sram_access_rate = np.clip(
            sram_access_rate / (sram_max + 1e-8), 0.0, 1.0
        )
    
        # --- NoC traffic distribution ---
        noc_traffic = np.zeros(self.num_noc_routers, dtype=np.float32)
        for b in range(self.num_sram_banks):
            r = b % max(self.num_noc_routers, 1)
            noc_traffic[r] += sram_access_rate[b]
        noc_max = noc_traffic.max()
        noc_traffic = np.clip(noc_traffic / (noc_max + 1e-8), 0.0, 1.0)
    
        # --- Unified activity vector [num_mac + num_sram + num_routers] ---
        switching_activity = np.concatenate([
            mac_utilization,   # [num_mac_clusters]
            sram_access_rate,  # [num_sram_banks]
            noc_traffic,       # [num_noc_routers]
        ]).astype(np.float32)
    
        return LayerResult(
            latency_cycles=latency_cycles,
            energy_pj=total_energy_pj,
            mac_utilization=mac_utilization,
            sram_access_rate=sram_access_rate,
            noc_traffic=noc_traffic,
            switching_activity=switching_activity,
            total_macs=total_macs,
            total_memory_accesses=total_mem,
        )

    def run_workload(self, workload_layers: list, mapping: np.ndarray) -> "WorkloadResult":
        n = len(workload_layers)
        if len(mapping) != n:
            mapping = (np.resize(mapping, n) % self.num_mac_clusters).astype(np.int32)
    
        total_lat  = 0
        total_eng  = 0.0
        cum_mac    = np.zeros(self.num_mac_clusters, dtype=np.float32)
        cum_sram   = np.zeros(self.num_sram_banks,   dtype=np.float32)
        cum_noc    = np.zeros(self.num_noc_routers,   dtype=np.float32)
        layer_results = []
    
        for i, layer_cfg in enumerate(workload_layers):
            layer_map = np.array([mapping[i] % self.num_mac_clusters], dtype=np.int32)
            r = self.run_layer(layer_cfg, layer_map)
            layer_results.append(r)
            total_lat += r.latency_cycles
            total_eng += r.energy_pj
            cum_mac   += r.mac_utilization
            cum_sram  += r.sram_access_rate
            cum_noc   += r.noc_traffic
    
        norm = lambda x: np.clip(x / (x.max() + 1e-8), 0.0, 1.0).astype(np.float32)
        cum_mac  = norm(cum_mac)
        cum_sram = norm(cum_sram)
        cum_noc  = norm(cum_noc)
    
        return WorkloadResult(
            total_latency_cycles=total_lat,
            total_energy_pj=total_eng,
            mac_utilization=cum_mac,
            sram_access_rate=cum_sram,
            noc_traffic=cum_noc,
            switching_activity=np.concatenate([cum_mac, cum_sram, cum_noc]),
        )
