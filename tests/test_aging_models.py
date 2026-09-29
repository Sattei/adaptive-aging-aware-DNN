import pytest
import numpy as np

from aging_models.nbti_model import NBTIModel
from aging_models.hci_model import HCIModel
from aging_models.tddb_model import TDDBModel
from aging_models.aging_label_generator import AgingLabelGenerator

def test_nbti_monotonicity():
    model = NBTIModel(A=5e-3, n=0.25)
    stress_times = np.array([100.0, 1000.0, 10000.0])
    activity = np.array([0.5, 0.5, 0.5])
    
    degradation = model.compute_degradation(stress_times, activity)
    assert degradation[0] < degradation[1] < degradation[2], "NBTI degradation must be monotonically increasing with time"
    
def test_hci_monotonicity():
    model = HCIModel(B=1e-4, m=0.5)
    stress_times = np.array([100.0, 1000.0, 10000.0])
    current_density = np.array([0.8, 0.8, 0.8])
    
    degradation = model.compute_degradation(current_density, stress_times)
    assert degradation[0] < degradation[1] < degradation[2], "HCI degradation must increase with time"

def test_tddb_probabilities():
    model = TDDBModel(k=2.5, beta=2.0)
    e_field = np.array([0.2, 0.5, 0.8])
    stress_time = np.array([1_000_000.0] * 3)
    temperature = np.array([373.0] * 3)

    probs = model.failure_probability(e_field, stress_time, temperature)
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be in [0, 1]"
    assert probs[0] < probs[1] < probs[2], "Failure probability must increase with electric stress"

def test_aging_generator_trajectory():
    nbti = NBTIModel(A=5e-3, n=0.25)
    hci = HCIModel(B=1e-4, m=0.5)
    tddb = TDDBModel(k=2.5, beta=2.0)
    
    gen = AgingLabelGenerator(nbti, hci, tddb, weights={'nbti': 0.4, 'hci': 0.4, 'tddb': 0.2})
    
    # Simulate 5 steps for 2 nodes
    seq = []
    for _ in range(5):
        seq.append({
            'switching_activity': np.array([0.2, 0.8]),
            'mac_utilization': np.array([0.1, 0.9]),
            'voltage': np.array([0.8, 0.8]),
            'temperature_k': np.array([373.0, 373.0]),
        })
        
    traj = gen.generate_trajectory_labels(seq, timestep_s=3600.0)
    assert traj.shape == (5, 2), "Shape must be [T, N]"
    
    # Node 1 (index 1) has higher activity, should age faster
    for t in range(5):
        assert traj[t, 0] < traj[t, 1], "Higher activity node must age faster"
        
    # Aging must monotonically increase for both nodes
    for i in range(4):
        assert traj[i, 0] <= traj[i+1, 0]
        assert traj[i, 1] <= traj[i+1, 1]

def test_aging_generator_mechanisms_and_composite():
    nbti = NBTIModel(A=5e-3, n=0.25)
    hci = HCIModel(B=1e-4, m=0.5)
    tddb = TDDBModel(k=2.5, beta=2.0)

    weights = {'nbti': 0.4, 'hci': 0.4, 'tddb': 0.2}
    gen = AgingLabelGenerator(nbti, hci, tddb, weights=weights)

    activity = {
        'switching_activity': np.array([0.2, 0.8]),
        'mac_utilization': np.array([0.1, 0.9]),
        'voltage': np.array([0.8, 0.8]),
        'temperature_k': np.array([373.0, 373.0]),
    }

    mechanisms = gen.compute_mechanisms(activity, stress_time_s=3600.0)
    score = gen.compute_aging_score(activity, stress_time_s=3600.0)

    assert mechanisms.shape == (2, 3)
    assert np.all((mechanisms >= 0.0) & (mechanisms <= 1.0))

    expected = (
        weights['nbti'] * mechanisms[:, 0] +
        weights['hci'] * mechanisms[:, 1] +
        weights['tddb'] * mechanisms[:, 2]
    )

    assert score.shape == (2,)
    assert np.allclose(score, expected)
    assert np.allclose(score, gen.combine_mechanisms(mechanisms))



def test_aging_generator_requires_explicit_stress_arrays():
    nbti = NBTIModel(A=5e-3, n=0.25)
    hci = HCIModel(B=1e-4, m=0.5)
    tddb = TDDBModel(k=2.5, beta=2.0)
    gen = AgingLabelGenerator(nbti, hci, tddb, weights={'nbti': 0.4, 'hci': 0.4, 'tddb': 0.2})

    base = {
        'switching_activity': np.array([0.2, 0.8]),
        'mac_utilization': np.array([0.1, 0.9]),
    }

    with pytest.raises(KeyError, match='voltage'):
        gen.compute_mechanisms(base, stress_time_s=3600.0)

    with_voltage = dict(base)
    with_voltage['voltage'] = np.array([0.8, 0.8])

    with pytest.raises(KeyError, match='temperature_k'):
        gen.compute_mechanisms(with_voltage, stress_time_s=3600.0)


def test_aging_generator_validates_stress_array_shapes():
    nbti = NBTIModel(A=5e-3, n=0.25)
    hci = HCIModel(B=1e-4, m=0.5)
    tddb = TDDBModel(k=2.5, beta=2.0)
    gen = AgingLabelGenerator(nbti, hci, tddb, weights={'nbti': 0.4, 'hci': 0.4, 'tddb': 0.2})

    activity = {
        'switching_activity': np.array([0.2, 0.8]),
        'mac_utilization': np.array([0.1, 0.9]),
        'voltage': np.array([0.8]),
        'temperature_k': np.array([373.0, 373.0]),
    }

    with pytest.raises(ValueError, match='voltage must have shape'):
        gen.compute_mechanisms(activity, stress_time_s=3600.0)


def test_nbti_voltage_and_temperature_response():
    model = NBTIModel(A=5e-3, n=0.25)
    time = np.array([10_000.0, 10_000.0, 10_000.0])
    activity = np.array([0.5, 0.5, 0.5])

    by_voltage = model.compute_degradation(
        time,
        activity,
        np.array([0.7, 0.8, 0.9]),
        np.array([373.0, 373.0, 373.0]),
    )
    assert by_voltage[0] < by_voltage[1] < by_voltage[2]

    by_temperature = model.compute_degradation(
        time,
        activity,
        np.array([0.8, 0.8, 0.8]),
        np.array([348.0, 373.0, 398.0]),
    )
    assert by_temperature[0] < by_temperature[1] < by_temperature[2]


def test_hci_voltage_and_temperature_response():
    model = HCIModel(B=1e-4, m=0.5)
    current_proxy = np.array([0.8, 0.8, 0.8])
    time = np.array([10_000.0, 10_000.0, 10_000.0])

    by_voltage = model.compute_degradation(
        current_proxy,
        time,
        np.array([0.7, 0.8, 0.9]),
        np.array([373.0, 373.0, 373.0]),
    )
    assert by_voltage[0] < by_voltage[1] < by_voltage[2]

    by_temperature = model.compute_degradation(
        current_proxy,
        time,
        np.array([0.8, 0.8, 0.8]),
        np.array([348.0, 373.0, 398.0]),
    )
    assert by_temperature[0] < by_temperature[1] < by_temperature[2]


def test_tddb_time_and_temperature_response():
    model = TDDBModel(k=2.5, beta=2.0)

    by_time = model.failure_probability(
        np.array([0.8, 0.8, 0.8]),
        np.array([10_000.0, 100_000.0, 1_000_000.0]),
        np.array([373.0, 373.0, 373.0]),
    )
    assert by_time[0] < by_time[1] < by_time[2]

    by_temperature = model.failure_probability(
        np.array([0.8, 0.8, 0.8]),
        np.array([1_000_000.0, 1_000_000.0, 1_000_000.0]),
        np.array([348.0, 373.0, 398.0]),
    )
    assert by_temperature[0] < by_temperature[1] < by_temperature[2]


def test_reference_conditions_preserve_nbti_hci_baseline():
    time = np.array([100.0, 1000.0, 10000.0])
    activity = np.array([0.5, 0.5, 0.5])
    voltage = np.array([0.8, 0.8, 0.8])
    temperature = np.array([373.0, 373.0, 373.0])

    nbti_model = NBTIModel(A=5e-3, n=0.25)
    old_nbti = 5e-3 * np.power(activity * time, 0.25)
    new_nbti = nbti_model.compute_degradation(time, activity, voltage, temperature)
    assert np.allclose(old_nbti, new_nbti)

    current_proxy = np.array([0.8, 0.8, 0.8])
    hci_model = HCIModel(B=1e-4, m=0.5)
    old_hci = 1e-4 * np.power(current_proxy + 1e-12, 0.5) * np.sqrt(time)
    new_hci = hci_model.compute_degradation(current_proxy, time, voltage, temperature)
    assert np.allclose(old_hci, new_hci)

def _stage3_activity(
    switching=np.array([0.2, 0.8]),
    utilization=np.array([0.1, 0.9]),
    voltage=np.array([0.8, 0.8]),
    temperature=np.array([373.0, 373.0]),
):
    return {
        "switching_activity": np.asarray(switching, dtype=np.float64),
        "mac_utilization": np.asarray(utilization, dtype=np.float64),
        "voltage": np.asarray(voltage, dtype=np.float64),
        "temperature_k": np.asarray(temperature, dtype=np.float64),
    }


def _stage3_generator():
    return AgingLabelGenerator(
        NBTIModel(A=5e-3, n=0.25),
        HCIModel(B=1e-4, m=0.5),
        TDDBModel(k=2.5, beta=2.0),
        weights={"nbti": 0.4, "hci": 0.4, "tddb": 0.2},
    )


def test_state_initialization():
    gen = _stage3_generator()
    state = gen.initialize_state(5)
    assert state.shape == (5, 3)
    assert state.dtype == np.float64
    assert np.all(state == 0.0)


def test_state_zero_delta_is_unchanged():
    gen = _stage3_generator()
    activity = _stage3_activity()
    state = np.array([
        [0.02, 0.01, 0.001],
        [0.03, 0.02, 0.002],
    ])
    updated = gen.step_state(state, activity, delta_t_s=0.0)
    assert np.array_equal(updated, state)
    assert updated is not state


def test_state_is_persistent_and_monotonic():
    gen = _stage3_generator()
    activity = _stage3_activity()

    state0 = gen.initialize_state(2)
    state1 = gen.step_state(state0, activity, delta_t_s=100.0)
    state2 = gen.step_state(state1, activity, delta_t_s=100.0)

    assert np.all(state1 >= state0)
    assert np.all(state2 >= state1)
    assert np.any(state2 > state1)


def test_state_constant_stress_chunking_consistency():
    gen = _stage3_generator()
    activity = _stage3_activity()

    one_step = gen.step_state(
        gen.initialize_state(2),
        activity,
        delta_t_s=1000.0,
    )

    chunked = gen.initialize_state(2)
    for _ in range(10):
        chunked = gen.step_state(
            chunked,
            activity,
            delta_t_s=100.0,
        )

    assert np.allclose(one_step, chunked, rtol=1e-10, atol=1e-12)


def test_state_higher_second_interval_stress_adds_more_damage():
    gen = _stage3_generator()

    base = _stage3_activity(
        switching=np.array([0.4, 0.4]),
        utilization=np.array([0.4, 0.4]),
    )
    low = _stage3_activity(
        switching=np.array([0.2, 0.2]),
        utilization=np.array([0.2, 0.2]),
    )
    high = _stage3_activity(
        switching=np.array([0.8, 0.8]),
        utilization=np.array([0.8, 0.8]),
    )

    state1 = gen.step_state(gen.initialize_state(2), base, delta_t_s=100.0)
    low_state = gen.step_state(state1, low, delta_t_s=100.0)
    high_state = gen.step_state(state1, high, delta_t_s=100.0)

    assert np.all(high_state >= low_state)
    assert np.any(high_state > low_state)


def test_state_zero_dynamic_stress_does_not_grow():
    gen = _stage3_generator()
    idle = _stage3_activity(
        switching=np.zeros(2),
        utilization=np.zeros(2),
    )
    state = np.array([
        [0.02, 0.01, 0.001],
        [0.03, 0.02, 0.002],
    ])

    updated = gen.step_state(state, idle, delta_t_s=1000.0)
    assert np.allclose(updated, state)


def test_state_to_mechanisms_and_tddb_probability_bounds():
    gen = _stage3_generator()
    activity = _stage3_activity()

    state = gen.step_state(
        gen.initialize_state(2),
        activity,
        delta_t_s=1_000_000.0,
    )
    mechanisms = gen.state_to_mechanisms(state)

    assert mechanisms.shape == (2, 3)
    assert np.all(state[:, 2] >= 0.0)
    assert np.all((mechanisms >= 0.0) & (mechanisms <= 1.0))


def test_state_shape_validation():
    gen = _stage3_generator()
    activity = _stage3_activity()

    with pytest.raises(ValueError, match="previous_state must have shape"):
        gen.step_state(np.zeros((3, 3)), activity, delta_t_s=100.0)

    with pytest.raises(ValueError, match=r"state must have shape \[N, 3\]"):
        gen.state_to_mechanisms(np.zeros((2, 2)))

def test_first_stateful_step_matches_static_constant_stress():
    gen = _stage3_generator()
    activity = _stage3_activity()

    static_mechanisms = gen.compute_mechanisms(
        activity,
        stress_time_s=3600.0,
    )
    state = gen.step_state(
        gen.initialize_state(2),
        activity,
        delta_t_s=3600.0,
    )
    stateful_mechanisms = gen.state_to_mechanisms(state)

    assert np.allclose(
        stateful_mechanisms,
        static_mechanisms,
        rtol=1e-10,
        atol=1e-12,
    )


def test_raw_static_state_matches_static_mechanisms_and_sequential_trajectory():
    gen = _stage3_generator()
    activity = _stage3_activity()
    raw_state = gen.compute_raw_state(activity, stress_time_s=3600.0)

    assert raw_state.shape == (2, 3)
    assert raw_state.dtype == np.float64
    assert np.allclose(
        gen.state_to_mechanisms(raw_state),
        gen.compute_mechanisms(activity, stress_time_s=3600.0),
        rtol=1e-10,
        atol=1e-12,
    )

    before = raw_state.copy()
    trajectory = gen.generate_trajectory_labels(
        [activity, activity],
        timestep_s=100.0,
        initial_state=raw_state,
    )
    assert np.array_equal(raw_state, before)
    assert trajectory.shape == (2, 2)
    assert np.all(trajectory[1] >= trajectory[0])


def test_state_rejects_nonfinite_inputs():
    gen = _stage3_generator()
    activity = _stage3_activity()

    with pytest.raises(ValueError, match="finite and nonnegative"):
        gen.step_state(gen.initialize_state(2), activity, delta_t_s=np.inf)

    nonfinite_voltage = dict(activity)
    nonfinite_voltage["voltage"] = np.array([0.8, np.nan])
    with pytest.raises(ValueError, match="voltage must contain only finite values"):
        gen.step_state(gen.initialize_state(2), nonfinite_voltage, delta_t_s=1.0)
