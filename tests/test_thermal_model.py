import numpy as np

from simulator.timeloop_runner import AnalyticalSimulator, get_default_workload
from features.feature_builder import FeatureBuilder


def _cfg():
    return {
        "pe_array": [4, 4],
        "pe_array_rows": 4,
        "pe_array_cols": 4,
        "num_pes": 16,
        "mac_clusters": 16,
        "sram_banks": 8,
        "noc_routers": 4,
        "freq_mhz": 1000.0,
        "voltage_v": 0.8,
        "mac_energy_pj": 0.25,
        "sram_rd_energy_pj": 1.5,
        "dram_rd_energy_pj": 70.0,
        "noc_energy_per_byte_pj": 2.0,
        "idle_leakage_pj_per_cycle": 0.005,
        "ambient_temperature_k": 318.15,
        "thermal_package_rth_k_per_w": 1.5,
        "thermal_local_rth_k_per_w": 60.0,
        "max_temperature_k": 398.15,
    }


def test_workload_thermal_state_shapes_and_bounds():
    cfg = _cfg()
    sim = AnalyticalSimulator(cfg)
    result = sim.run_workload(get_default_workload())
    n = cfg["mac_clusters"] + cfg["sram_banks"] + cfg["noc_routers"]

    assert result.node_power_w.shape == (n,)
    assert result.temperature_k.shape == (n,)
    assert np.all(result.node_power_w >= 0.0)
    assert np.all(result.temperature_k >= cfg["ambient_temperature_k"])
    assert np.all(result.temperature_k <= cfg["max_temperature_k"])
    assert result.avg_power_w >= 0.0
    assert np.isclose(result.node_power_w.sum(), result.avg_power_w, rtol=1e-6, atol=1e-12)
    assert np.isclose(
        result.compute_energy_pj
        + result.sram_energy_pj
        + result.noc_energy_pj
        + result.leakage_energy_pj,
        result.onchip_energy_pj,
        rtol=1e-12,
        atol=1e-9,
    )
    # DRAM is retained in the workload-system metric, not spread onto die nodes.
    assert result.energy_pj > result.onchip_energy_pj


def test_component_power_stays_with_its_component_type_and_has_a_fallback():
    cfg = _cfg()
    sim = AnalyticalSimulator(cfg)
    mac = np.zeros(cfg["mac_clusters"], dtype=np.float32)
    mac[3] = 1.0
    zero_sram = np.zeros(cfg["sram_banks"], dtype=np.float32)
    zero_noc = np.zeros(cfg["noc_routers"], dtype=np.float32)

    node_power, temperature, avg_power = sim._estimate_thermal_state(
        compute_energy_pj=100.0,
        sram_energy_pj=0.0,
        noc_energy_pj=0.0,
        active_leakage_energy_pj=0.0,
        idle_leakage_energy_pj=0.0,
        latency_cycles=100.0,
        mac_util=mac,
        sram_access=zero_sram,
        noc_traffic=zero_noc,
    )
    mac_end = cfg["mac_clusters"]
    assert np.isclose(node_power.sum(), avg_power, rtol=1e-6, atol=1e-12)
    assert node_power[3] > 0.0
    assert np.all(node_power[:3] == 0.0)
    assert np.all(node_power[4:] == 0.0)
    assert temperature[3] > temperature[0]

    node_power, _, avg_power = sim._estimate_thermal_state(
        compute_energy_pj=0.0,
        sram_energy_pj=80.0,
        noc_energy_pj=0.0,
        active_leakage_energy_pj=0.0,
        idle_leakage_energy_pj=0.0,
        latency_cycles=100.0,
        mac_util=mac,
        sram_access=zero_sram,
        noc_traffic=zero_noc,
    )
    sram_power = node_power[mac_end:mac_end + cfg["sram_banks"]]
    assert np.isclose(sram_power.sum(), avg_power, rtol=1e-6, atol=1e-12)
    assert np.allclose(sram_power, sram_power[0])
    assert np.all(node_power[:mac_end] == 0.0)
    assert np.all(node_power[mac_end + cfg["sram_banks"]:] == 0.0)


def test_empty_workload_has_ambient_temperature_and_zero_power():
    cfg = _cfg()
    result = AnalyticalSimulator(cfg).run_workload([])
    assert result.avg_power_w == 0.0
    assert np.all(result.node_power_w == 0.0)
    assert np.all(result.temperature_k == cfg["ambient_temperature_k"])


def test_active_nodes_are_not_colder_than_idle_nodes():
    cfg = _cfg()
    sim = AnalyticalSimulator(cfg)
    result = sim.run_workload(get_default_workload(), mapping=np.zeros(len(get_default_workload()), dtype=np.int32))

    active_mac_temp = result.temperature_k[0]
    idle_mac_temps = result.temperature_k[1:cfg["mac_clusters"]]
    assert np.all(active_mac_temp >= idle_mac_temps)


def test_feature_builder_uses_simulated_temperature():
    cfg = _cfg()
    sim = AnalyticalSimulator(cfg)
    result = sim.run_workload(get_default_workload())
    n = result.switching_activity.shape[0]
    activity = {
        "switching_activity": result.switching_activity,
        "mac_utilization": result.mac_utilization,
        "sram_access_rate": result.sram_access_rate,
        "noc_traffic": result.noc_traffic,
        "temperature_k": result.temperature_k,
    }
    features = FeatureBuilder(cfg).build_node_features(
        activity,
        "ResNet-50",
        result.total_latency_cycles,
        result.total_energy_pj,
        3600.0,
    )
    expected = np.clip(
        (result.temperature_k - cfg["ambient_temperature_k"])
        / (cfg["max_temperature_k"] - cfg["ambient_temperature_k"]),
        0.0,
        1.0,
    )
    assert features.shape == (n, 8)
    assert np.allclose(features[:, 4].numpy(), expected, atol=1e-6)
