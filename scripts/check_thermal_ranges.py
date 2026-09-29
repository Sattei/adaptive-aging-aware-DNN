import numpy as np
from omegaconf import OmegaConf

from simulator.timeloop_runner import TimeloopRunner
from simulator.workload_runner import WorkloadRunner


acc_cfg = OmegaConf.load("configs/accelerator.yaml").accelerator
wl_cfg = OmegaConf.load("configs/workloads.yaml")

sim = TimeloopRunner(acc_cfg)

try:
    runner = WorkloadRunner(wl_cfg.workloads)
except Exception:
    runner = WorkloadRunner(wl_cfg)

names = [
    "ResNet-50",
    "MobileNetV2",
    "EfficientNet-B4",
    "BERT-Base",
    "ViT-B/16",
]

rng = np.random.default_rng(42)

for name in names:
    layers = runner.get_workload_layers(name)
    n = len(layers)
    m = int(acc_cfg.mac_clusters)

    mappings = {
        "single": np.zeros(n, dtype=np.int32),
        "round_robin": np.arange(n, dtype=np.int32) % m,
        "random": rng.integers(0, m, size=n, dtype=np.int32),
    }

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    for map_name, mapping in mappings.items():
        r = sim.run_workload(layers, mapping)

        temp = np.asarray(r.temperature_k)
        power = np.asarray(r.node_power_w)

        print(f"\nMapping: {map_name}")
        print(f"Latency cycles : {r.total_latency_cycles:.2f}")
        print(f"On-chip energy : {r.onchip_energy_pj:.2f} pJ")
        print(
            "Energy buckets : "
            f"compute={r.compute_energy_pj:.2f} pJ, "
            f"sram={r.sram_energy_pj:.2f} pJ, "
            f"noc={r.noc_energy_pj:.2f} pJ, "
            f"leakage={r.leakage_energy_pj:.2f} pJ"
        )
        print(f"Average power   : {r.avg_power_w:.6f} W")

        print(
            "Temperature K  : "
            f"min={temp.min():.2f}, "
            f"mean={temp.mean():.2f}, "
            f"max={temp.max():.2f}"
        )

        print(
            "Temperature C  : "
            f"min={temp.min() - 273.15:.2f}, "
            f"mean={temp.mean() - 273.15:.2f}, "
            f"max={temp.max() - 273.15:.2f}"
        )

        print(
            "Node power W   : "
            f"min={power.min():.6f}, "
            f"mean={power.mean():.6f}, "
            f"max={power.max():.6f}"
        )

        hot = int(np.argmax(temp))

        if hot < int(acc_cfg.mac_clusters):
            node_type = "MAC"
        elif hot < int(acc_cfg.mac_clusters) + int(acc_cfg.sram_banks):
            node_type = "SRAM"
        else:
            node_type = "NoC"

        print(
            f"Hottest node   : {hot} "
            f"({node_type}) -> {temp[hot]:.2f} K"
        )
