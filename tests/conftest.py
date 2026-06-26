"""Shared pytest fixtures for the ch04 test suite.

Module-specific fixtures land in each test_<module>.py.
Cross-module fixtures (action dims, fake stats, dummy chunks) live here.
"""

import numpy as np
import pytest

ACTION_DIM = 6  # SO-101: 5 arm joints + gripper (Ch 2 Table 2.2)
N_BINS = 256  # RT-1 / OpenVLA choice
SMOLLM_VOCAB = 49152  # native SmolLM-135M vocab (no expansion in Ch 3)


@pytest.fixture
def action_bounds():
    """Per-dimension (lo, hi) for a 6-DOF action, in [-1, 1]-ish range."""
    lo = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return lo, hi


@pytest.fixture
def fake_lerobot_stats():
    """Minimal stats dict shaped like LeRobot meta/stats.json['action']."""
    return {
        "action": {
            "q01": [-0.9, -0.8, -1.0, -0.7, -0.6, -1.0],
            "q99": [0.9, 0.8, 1.0, 0.7, 0.6, 1.0],
            "min": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    }


@pytest.fixture
def dummy_action_batch():
    """[8, 6] random continuous actions in [-1, 1]."""
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=(8, ACTION_DIM))
