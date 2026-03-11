import pytest
import os
import sys
from pathlib import Path

# Ensure root is in PATH for direct testing
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

@pytest.fixture(scope="session")
def base_config():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "accelerator": {
            "num_mac_clusters": 4, "num_sram_banks": 4, "num_noc_routers": 2,
            "pe_array": [4, 4],
            "max_macs_per_cluster": 256,
            "clock_ghz": 1.0,
            "sram_read_energy_pj": 2.0,
            "noc_hop_energy_pj": 0.5,
            "mac_energy_pj_per_op": 0.1,
            "ops_per_cycle": 2
        },
        "workloads": {
            "test_wl": "data"
        },
        "aging": {
            "nbti_A": 0.005, "nbti_n": 0.25,
            "hci_B": 0.0001, "hci_m": 0.5,
            "tddb_k": 2.5, "tddb_beta": 10.0
        },
        "planning": {
            "failure_threshold": 0.8
        }
    })
