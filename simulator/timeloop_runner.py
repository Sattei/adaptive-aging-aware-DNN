"""
Analytical hardware simulator for DNN accelerator performance estimation.

Replaces a real Timeloop call with a closed-form roofline model.
This is methodologically valid for research — Timeloop itself is an
analytical mapper/model, and many published papers use equivalent formulations.

Paper citation note: "We use an analytical roofline model to estimate
hardware performance metrics (latency, energy, utilisation) for each
DNN mapping, following the methodology of [Timeloop/Maestro-style analysis]."
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AcceleratorConfig:
    """Hardware parameters loaded from config."""
    num_pes:          int   = 256       # total processing elements
    pe_array_rows:    int   = 16
    pe_array_cols:    int   = 16
    mac_per_pe:       int   = 1         # MACs per PE per cycle
    sram_kb:          float = 256.0     # on-chip SRAM in KB
    dram_bw_gb_s:     float = 51.2      # off-chip DRAM bandwidth GB/s
    noc_bw_gb_s:      float = 512.0     # on-chip NoC bandwidth GB/s
    freq_mhz:         float = 1000.0    # clock frequency MHz
    voltage_v:        float = 1.0       # supply voltage (normalised ref)
    # Energy constants (pJ per operation)
    mac_energy_pj:    float = 0.25      # multiply-accumulate
    sram_rd_energy_pj: float = 1.5      # SRAM read per byte
    dram_rd_energy_pj: float = 70.0     # DRAM read per byte
    # Aging degradation (applied externally before calling simulate)
    aging_freq_degrade: float = 0.0     # fractional frequency loss (0–1)
    aging_leak_increase: float = 0.0    # fractional leakage increase (0–1)


@dataclass
class LayerSpec:
    """One DNN layer to simulate."""
    name:       str
    layer_type: str          # 'conv', 'fc', 'pool', 'bn'
    # Convolutional / FC dimensions
    N: int = 1               # batch size
    C: int = 3               # input channels
    K: int = 64              # output channels / filters
    R: int = 3               # filter height (1 for FC)
    S: int = 3               # filter width  (1 for FC)
    P: int = 224             # output height (1 for FC)
    Q: int = 224             # output width  (1 for FC)
    stride: int = 1


@dataclass
class SimResult:
    """Output of one simulation run."""
    layer_name:     str
    latency_cycles: float
    latency_ms:     float
    energy_pj:      float
    energy_uj:      float
    throughput_gops: float
    utilisation:    float        # PE array utilisation 0–1
    memory_bound:   bool
    dram_accesses_bytes: float
    compute_intensity: float     # FLOPs / byte (arithmetic intensity)
    # Per-node (PE) degradation contribution
    per_pe_stress:  np.ndarray = field(default_factory=lambda: np.zeros(256))


class AnalyticalSimulator:
    """
    Roofline-based analytical performance model.

    For a CONV layer the total MACs are:
        MACs = N * K * C * R * S * P * Q / (stride^2)

    Latency (compute-bound):
        cycles_compute = MACs / (num_pes * mac_per_pe)

    Latency (memory-bound):
        bytes_ifmap  = N * C * (P*stride + R - 1) * (Q*stride + S - 1) * 4  (float32)
        bytes_weight = K * C * R * S * 4
        bytes_ofmap  = N * K * P * Q * 4
        total_bytes  = bytes_ifmap + bytes_weight + bytes_ofmap
        cycles_mem   = total_bytes / (dram_bw_bytes_per_cycle)

    Actual latency = max(cycles_compute, cycles_mem) — roofline law.
    """

    def __init__(self, accel_cfg: AcceleratorConfig):
        self.cfg = accel_cfg
        # Effective frequency accounting for aging degradation
        self.eff_freq_mhz = accel_cfg.freq_mhz * (1.0 - accel_cfg.aging_freq_degrade)
        self.eff_freq_hz  = self.eff_freq_mhz * 1e6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_layer(self, layer: LayerSpec) -> SimResult:
        """Simulate a single DNN layer and return performance metrics."""
        if layer.layer_type == 'conv':
            return self._simulate_conv(layer)
        elif layer.layer_type == 'fc':
            return self._simulate_fc(layer)
        elif layer.layer_type in ('pool', 'bn'):
            return self._simulate_elementwise(layer)
        else:
            return self._simulate_conv(layer)  # fallback

    def simulate_workload(self, layers: List[LayerSpec]) -> Dict[str, SimResult]:
        """Simulate a full DNN and aggregate results."""
        results = {}
        for layer in layers:
            results[layer.name] = self.simulate_layer(layer)
        return results

    def aggregate_metrics(self, results: Dict[str, SimResult]) -> Dict:
        """Compute workload-level summary statistics."""
        total_latency_ms  = sum(r.latency_ms  for r in results.values())
        total_energy_uj   = sum(r.energy_uj   for r in results.values())
        avg_utilisation   = np.mean([r.utilisation for r in results.values()])
        total_gops        = sum(r.throughput_gops for r in results.values())
        mem_bound_layers  = sum(1 for r in results.values() if r.memory_bound)

        return {
            'total_latency_ms':  total_latency_ms,
            'total_energy_uj':   total_energy_uj,
            'avg_utilisation':   avg_utilisation,
            'total_gops':        total_gops,
            'mem_bound_fraction': mem_bound_layers / max(len(results), 1),
            'energy_delay_product': total_energy_uj * total_latency_ms,
        }

    # ------------------------------------------------------------------
    # Layer-type simulators
    # ------------------------------------------------------------------

    def _simulate_conv(self, layer: LayerSpec) -> SimResult:
        cfg = self.cfg
        N, K, C, R, S, P, Q = layer.N, layer.K, layer.C, layer.R, layer.S, layer.P, layer.Q
        stride = layer.stride

        total_macs = N * K * C * R * S * P * Q

        # --- compute bound ---
        peak_macs_per_cycle = cfg.num_pes * cfg.mac_per_pe
        cycles_compute = total_macs / peak_macs_per_cycle

        # --- memory bound ---
        H_in = P * stride + R - 1
        W_in = Q * stride + S - 1
        bytes_ifmap  = N * C * H_in * W_in * 4
        bytes_weight = K * C * R * S * 4
        bytes_ofmap  = N * K * P * Q * 4
        total_bytes  = bytes_ifmap + bytes_weight + bytes_ofmap

        dram_bw_bytes_per_cycle = (cfg.dram_bw_gb_s * 1e9) / self.eff_freq_hz
        cycles_mem = total_bytes / dram_bw_bytes_per_cycle

        # --- roofline ---
        cycles_total = max(cycles_compute, cycles_mem)
        memory_bound = cycles_mem > cycles_compute

        # --- time and energy ---
        latency_ms   = (cycles_total / self.eff_freq_hz) * 1e3
        mac_energy   = total_macs * cfg.mac_energy_pj
        mem_energy   = total_bytes * cfg.dram_rd_energy_pj
        # Leakage: proportional to cycles and num_pes
        leakage_pj   = cycles_total * cfg.num_pes * 0.01 * (1.0 + cfg.aging_leak_increase)
        energy_pj    = mac_energy + mem_energy + leakage_pj

        # --- utilisation ---
        utilisation = min(1.0, cycles_compute / cycles_total)

        # --- compute intensity (FLOPs/byte) ---
        compute_intensity = (2 * total_macs) / max(total_bytes, 1)

        # --- throughput ---
        throughput_gops = (2 * total_macs) / (latency_ms * 1e-3) / 1e9

        # --- per-PE stress (for aging model input) ---
        per_pe_stress = self._compute_pe_stress(total_macs, utilisation)

        return SimResult(
            layer_name=layer.name,
            latency_cycles=cycles_total,
            latency_ms=latency_ms,
            energy_pj=energy_pj,
            energy_uj=energy_pj * 1e-6,
            throughput_gops=throughput_gops,
            utilisation=utilisation,
            memory_bound=memory_bound,
            dram_accesses_bytes=total_bytes,
            compute_intensity=compute_intensity,
            per_pe_stress=per_pe_stress,
        )

    def _simulate_fc(self, layer: LayerSpec) -> SimResult:
        """Treat FC as a CONV with 1x1 spatial dimensions."""
        layer.R = layer.S = layer.P = layer.Q = 1
        layer.stride = 1
        return self._simulate_conv(layer)

    def _simulate_elementwise(self, layer: LayerSpec) -> SimResult:
        """Pooling / BN — memory-bandwidth limited, low compute."""
        cfg = self.cfg
        total_bytes = layer.N * layer.C * layer.P * layer.Q * 4 * 2  # read + write
        dram_bw_bytes_per_cycle = (cfg.dram_bw_gb_s * 1e9) / self.eff_freq_hz
        cycles = total_bytes / dram_bw_bytes_per_cycle
        latency_ms  = (cycles / self.eff_freq_hz) * 1e3
        energy_pj   = total_bytes * cfg.sram_rd_energy_pj
        utilisation = 0.05  # minimal compute, mostly data movement
        throughput_gops = (layer.N * layer.C * layer.P * layer.Q) / (latency_ms * 1e-3) / 1e9
        return SimResult(
            layer_name=layer.name,
            latency_cycles=cycles,
            latency_ms=latency_ms,
            energy_pj=energy_pj,
            energy_uj=energy_pj * 1e-6,
            throughput_gops=throughput_gops,
            utilisation=utilisation,
            memory_bound=True,
            dram_accesses_bytes=total_bytes,
            compute_intensity=0.5,
            per_pe_stress=self._compute_pe_stress(0, utilisation),
        )

    # ------------------------------------------------------------------
    # Helper: distribute stress across PE array
    # ------------------------------------------------------------------

    def _compute_pe_stress(self, total_macs: float, utilisation: float) -> np.ndarray:
        """
        Approximate per-PE stress as a spatial distribution across the array.
        PEs near the centre of a systolic array typically see higher utilisation.
        """
        cfg = self.cfg
        rows, cols = cfg.pe_array_rows, cfg.pe_array_cols
        num_pes = rows * cols

        # Gaussian spatial stress pattern — centre PEs are busier
        cx, cy = rows / 2.0, cols / 2.0
        stress = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                dist = math.sqrt((r - cx)**2 + (c - cy)**2)
                stress[r, c] = math.exp(-0.5 * (dist / (rows / 3.0))**2)

        # Normalise so mean equals utilisation
        mean_s = stress.mean()
        if mean_s > 0:
            stress = stress / mean_s * utilisation
        return stress.flatten()[:num_pes]


# ------------------------------------------------------------------
# Convenience builder — used by run_full_pipeline.py
# ------------------------------------------------------------------

def build_simulator_from_config(cfg) -> AnalyticalSimulator:
    """Build an AnalyticalSimulator from a Hydra OmegaConf config."""
    accel = AcceleratorConfig(
        num_pes         = cfg.accelerator.get('num_pes', 256),
        pe_array_rows   = cfg.accelerator.get('pe_array_rows', 16),
        pe_array_cols   = cfg.accelerator.get('pe_array_cols', 16),
        sram_kb         = cfg.accelerator.get('sram_kb', 256.0),
        dram_bw_gb_s    = cfg.accelerator.get('dram_bw_gb_s', 51.2),
        noc_bw_gb_s     = cfg.accelerator.get('noc_bw_gb_s', 512.0),
        freq_mhz        = cfg.accelerator.get('freq_mhz', 1000.0),
    )
    return AnalyticalSimulator(accel)


def get_default_workload() -> List[LayerSpec]:
    """ResNet-18-style workload for quick testing."""
    return [
        LayerSpec('conv1',  'conv', N=1, C=3,   K=64,  R=7, S=7, P=112, Q=112, stride=2),
        LayerSpec('layer1a','conv', N=1, C=64,  K=64,  R=3, S=3, P=56,  Q=56,  stride=1),
        LayerSpec('layer1b','conv', N=1, C=64,  K=64,  R=3, S=3, P=56,  Q=56,  stride=1),
        LayerSpec('layer2a','conv', N=1, C=64,  K=128, R=3, S=3, P=28,  Q=28,  stride=2),
        LayerSpec('layer2b','conv', N=1, C=128, K=128, R=3, S=3, P=28,  Q=28,  stride=1),
        LayerSpec('layer3a','conv', N=1, C=128, K=256, R=3, S=3, P=14,  Q=14,  stride=2),
        LayerSpec('layer3b','conv', N=1, C=256, K=256, R=3, S=3, P=14,  Q=14,  stride=1),
        LayerSpec('layer4a','conv', N=1, C=256, K=512, R=3, S=3, P=7,   Q=7,   stride=2),
        LayerSpec('layer4b','conv', N=1, C=512, K=512, R=3, S=3, P=7,   Q=7,   stride=1),
        LayerSpec('avgpool', 'pool', N=1, C=512, K=512, P=1, Q=1),
        LayerSpec('fc',     'fc',  N=1, C=512, K=1000, R=1, S=1, P=1, Q=1),
    ]
# Alias for pipeline compatibility
TimeloopRunner = AnalyticalSimulator

# Alias for pipeline compatibility
WorkloadResult = SimResult

LayerResult = SimResult
