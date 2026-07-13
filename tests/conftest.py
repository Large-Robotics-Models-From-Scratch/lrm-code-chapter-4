"""Shared pytest fixtures for the ch04 test suite.

Module-specific fixtures land in each test_<module>.py. Cross-module
fixtures (action bounds, fake stats, dummy chunks) live here.

Note on ``fake_stats``: the real ``lerobot/svla_so101_pickplace``
stats lack ``q01``/``q99`` (only min/max/mean/std/count). The
tokenizer computes the percentiles itself (PR 1); this fixture is the
shape the tokenizer produces after that pass, not what the dataset
ships.
"""

import numpy as np
import pytest
import torch


@pytest.fixture
def action_bounds():
    lo = -np.ones(6, dtype=np.float32)
    hi = np.ones(6, dtype=np.float32)
    return lo, hi


@pytest.fixture
def fake_stats():
    return {
        "action": {
            "q01": [-1.0, -0.5, -2.0, -1.0, -1.0, 0.0],
            "q99": [1.0, 0.5, 2.0, 1.0, 1.0, 1.0],
        }
    }


@pytest.fixture
def dummy_chunk():
    torch.manual_seed(0)
    return torch.rand(2, 16, 6) * 2 - 1  # [B, H, D] in [-1, 1]
