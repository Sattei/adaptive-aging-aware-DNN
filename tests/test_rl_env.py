import pytest
import numpy as np
from omegaconf import OmegaConf

from simulator.timeloop_runner import TimeloopRunner
from planning.lifetime_planner import LifetimePlanner
from rl.environment import AgingControlEnv
from rl.policy_network import ActorCritic
from graph.accelerator_graph import AcceleratorGraph

def test_gym_environment():
    # Setup dependencies
    cfg = OmegaConf.create({
        'pe_array': [8, 8],
        'mac_clusters': 4,
        'sram_banks': 2,
        'noc_routers': 2,
        'horizon_length': 5,
        'workload_feature_dim': 16,
        'max_layers': 10
    })
    
    graph = AcceleratorGraph(cfg)
    graph.build()
    
    sim = TimeloopRunner(cfg)
    planner = LifetimePlanner(graph, cfg)
    
    env = AgingControlEnv(sim, planner, None, None, graph, None, None, None, cfg)
    
    # 1. Test Reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert env.mechanism_state.shape == (8, 3)
    assert env.mechanism_state.dtype == np.float64
    assert np.all(env.mechanism_state == 0.0)
    
    # Expected N = 4 + 2 + 2 = 8
    # obs dim: 8 + (8*5) + 16 + 10 + 1 + 8 = 83
    assert len(obs) == 83
    
    # 2. Test Step
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(next_obs, np.ndarray)
    assert len(next_obs) == 83
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    first_state = env.mechanism_state.copy()
    assert np.all(first_state >= 0.0)

    env.step(env.action_space.sample())
    assert np.all(env.mechanism_state >= first_state)

    env.reset(seed=7)
    assert np.all(env.mechanism_state == 0.0)
