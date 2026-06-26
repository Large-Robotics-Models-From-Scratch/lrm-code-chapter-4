"""Tests for ch04.action_tokenizer.

Covers the round trip, the worked numeric example from the chapter, the
upper-edge clip bug, batched shapes/dtypes, Q01/Q99 construction, and the
reserved-vocabulary token mapping the autoregressive head consumes.
"""

import numpy as np
import pytest

from ch04.action_tokenizer import ActionTokenizer

N_BINS = 256
SMOLLM_VOCAB = 49152


def test_worked_numeric_example():
    """The chapter's example: 0.347 rad over [-pi, pi] -> bin 142.

    The decoded value is the bin *center*, derived from the formula rather
    than a hand-computed constant so the test cannot drift from the math.
    """
    lo, hi, n = -np.pi, np.pi, 256
    tok = ActionTokenizer(lo=[lo], hi=[hi], n_bins=n)
    ids = tok.encode([0.347])
    assert ids.tolist() == [142]
    expected_center = lo + (142 + 0.5) * (hi - lo) / n
    recovered = tok.decode(ids)
    assert recovered[0] == pytest.approx(expected_center)
    # Round trip is within half a bin width of the input.
    assert abs(recovered[0] - 0.347) <= (hi - lo) / (2 * n)


def test_roundtrip_error_bounded_by_half_bin(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    rng = np.random.default_rng(1)
    actions = rng.uniform(lo, hi, size=(2000, len(lo)))
    err = np.abs(tok.decode(tok.encode(actions)) - actions)
    half_bin = (hi - lo) / (2 * N_BINS)
    # Every reconstruction is within half a bin width of the input.
    assert np.all(err <= half_bin + 1e-9)


def test_upper_edge_clips_to_last_bin(action_bounds):
    """encode(hi) must return n_bins - 1, never n_bins (out of range)."""
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    ids = tok.encode(hi)
    assert ids.tolist() == [N_BINS - 1] * len(hi)


def test_out_of_range_inputs_are_clipped(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    below = tok.encode(lo - 5.0)
    above = tok.encode(hi + 5.0)
    assert below.tolist() == [0] * len(lo)
    assert above.tolist() == [N_BINS - 1] * len(hi)


def test_batched_shape_and_dtype(dummy_action_batch, action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    ids = tok.encode(dummy_action_batch)
    assert ids.shape == dummy_action_batch.shape
    assert ids.dtype == np.int64
    assert ids.min() >= 0 and ids.max() <= N_BINS - 1
    assert tok.decode(ids).shape == dummy_action_batch.shape


def test_chunk_shape_BHD(action_bounds):
    """A [B, H, D] action chunk tokenizes to [B, H, D] bin ids."""
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    rng = np.random.default_rng(2)
    chunk = rng.uniform(-1, 1, size=(4, 16, len(lo)))
    ids = tok.encode(chunk)
    assert ids.shape == (4, 16, len(lo))
    assert tok.decode(ids).shape == (4, 16, len(lo))


def test_scalar_and_batched_agree(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    rng = np.random.default_rng(3)
    rows = rng.uniform(-1, 1, size=(5, len(lo)))
    batched = tok.encode(rows)
    one_by_one = np.stack([tok.encode(r) for r in rows])
    assert np.array_equal(batched, one_by_one)


def test_from_lerobot_stats(fake_lerobot_stats):
    tok = ActionTokenizer.from_lerobot_stats(
        fake_lerobot_stats, key="action", n_bins=N_BINS
    )
    assert tok.action_dim == 6
    np.testing.assert_allclose(tok.lo, fake_lerobot_stats["action"]["q01"])
    np.testing.assert_allclose(tok.hi, fake_lerobot_stats["action"]["q99"])


def test_reserved_token_ids_in_range(dummy_action_batch, action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    start, end = tok.reserved_range(SMOLLM_VOCAB)
    assert (start, end) == (SMOLLM_VOCAB - N_BINS, SMOLLM_VOCAB)
    token_ids = tok.to_token_ids(dummy_action_batch, SMOLLM_VOCAB)
    assert token_ids.min() >= start
    assert token_ids.max() < end


def test_token_id_roundtrip_eq_decode(dummy_action_batch, action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    via_tokens = tok.from_token_ids(
        tok.to_token_ids(dummy_action_batch, SMOLLM_VOCAB), SMOLLM_VOCAB
    )
    via_bins = tok.decode(tok.encode(dummy_action_batch))
    np.testing.assert_array_equal(via_tokens, via_bins)


def test_shared_vocab_across_dimensions(action_bounds):
    """Same bin in different dimensions maps to the same reserved id."""
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    # An action at the low edge of every dim -> bin 0 in every dim ->
    # the same reserved token id in every dim (position disambiguates).
    token_ids = tok.to_token_ids(lo, SMOLLM_VOCAB)
    assert len(set(token_ids.tolist())) == 1
    assert token_ids[0] == SMOLLM_VOCAB - N_BINS


def test_validation_errors():
    with pytest.raises(ValueError):
        ActionTokenizer(lo=[0.0], hi=[1.0, 2.0])  # shape mismatch
    with pytest.raises(ValueError):
        ActionTokenizer(lo=[1.0], hi=[0.0])  # hi < lo
    with pytest.raises(ValueError):
        ActionTokenizer(lo=[0.0], hi=[1.0], n_bins=1)  # too few bins
    tok = ActionTokenizer(lo=[0.0], hi=[1.0], n_bins=8)
    with pytest.raises(ValueError):
        tok.encode([[0.1, 0.2]])  # wrong last dim
