import ast
from pathlib import Path

import numpy as np
import pytest

from ch04 import ActionTokenizer, fit_action_tokenizer


def test_vectorized_round_trip_and_dtype(action_bounds):
    lo, hi = action_bounds
    tokenizer = ActionTokenizer(lo, hi)
    values = np.linspace(-0.9, 0.9, 2 * 16 * 6).reshape(2, 16, 6)
    bins = tokenizer.encode(values)
    decoded = tokenizer.decode(bins)
    assert bins.shape == values.shape
    assert bins.dtype == np.int64
    assert decoded.shape == values.shape
    assert decoded.dtype == np.float32
    assert np.max(np.abs(decoded - values)) <= 1 / 256 + 1e-6


def test_edges_clip_and_hi_maps_to_last_bin(action_bounds):
    tokenizer = ActionTokenizer(*action_bounds)
    assert np.all(tokenizer.encode(np.full(6, -5.0)) == 0)
    assert np.all(tokenizer.encode(np.full(6, 5.0)) == 255)
    assert np.all(tokenizer.encode(action_bounds[1]) == 255)


def test_q01_q99_fit():
    values = np.arange(100 * 6, dtype=np.float32).reshape(100, 6)
    tokenizer = fit_action_tokenizer(values)
    np.testing.assert_allclose(
        tokenizer.lo, np.percentile(values, 1, axis=0)
    )
    np.testing.assert_allclose(
        tokenizer.hi, np.percentile(values, 99, axis=0)
    )


def test_native_vocab_mapping_is_exact(action_bounds):
    tokenizer = ActionTokenizer(*action_bounds)
    bins = np.array([0, 1, 127, 255], dtype=np.int64)
    tokens = tokenizer.bins_to_tokens(bins)
    assert tokens.tolist() == [48896, 48897, 49023, 49151]
    np.testing.assert_array_equal(tokenizer.tokens_to_bins(tokens), bins)


def test_invalid_bounds_rejected(action_bounds):
    lo, hi = action_bounds
    with pytest.raises(ValueError):
        ActionTokenizer(hi, lo)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_actions_rejected(action_bounds, bad):
    tokenizer = ActionTokenizer(*action_bounds)
    action = np.zeros(6, dtype=np.float32)
    action[2] = bad
    with pytest.raises(ValueError, match="finite"):
        tokenizer.encode(action)


def test_tokenizer_module_has_no_torch_import():
    path = Path(__file__).parents[1] / "src/ch04/action_tokenizer.py"
    tree = ast.parse(path.read_text())
    names = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in names
