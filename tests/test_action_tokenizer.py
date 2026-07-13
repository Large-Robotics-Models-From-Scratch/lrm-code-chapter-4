"""Tests for ch04.action_tokenizer.

Covers the round trip, the worked numeric example from the chapter,
the upper-edge clip bug, batched shapes/dtypes, Q01/Q99 construction
(both from stats that already carry them and from a raw dataset that
does not), and the reserved-vocabulary token mapping the
autoregressive head consumes.
"""

import numpy as np
import pytest

from ch04 import ACT_TOKEN_BASE, ACTION_DIM, N_BINS, SMOLLM_VOCAB
from ch04.action_tokenizer import ActionTokenizer


@pytest.mark.parametrize("lo,hi", [(-np.pi, np.pi), (-1.0, 1.0)])
def test_worked_numeric_example(lo, hi):
    """The chapter's example: 0.347 over [-pi, pi] (and [-1, 1]).

    The expected bin and center are derived from the same encode/
    decode formula the tokenizer uses, not hand-typed constants, so
    this test cannot silently drift from an old (wrong) manuscript
    number. See CLAUDE.md's "Known manuscript follow-up" note: the
    prose currently claims a different, incorrect center.
    """
    n = 256
    value = 0.347
    tok = ActionTokenizer(lo=[lo], hi=[hi], n_bins=n)

    expected_bin = int(np.floor((value - lo) / (hi - lo) * n))
    ids = tok.encode([value])
    assert ids.tolist() == [expected_bin]

    expected_center = lo + (expected_bin + 0.5) * (hi - lo) / n
    recovered = tok.decode(ids)
    assert recovered[0] == pytest.approx(expected_center)
    # Round trip is within half a bin width of the input.
    assert abs(recovered[0] - value) <= (hi - lo) / (2 * n)


def test_roundtrip_error_bounded_by_half_bin(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    rng = np.random.default_rng(1)
    actions = rng.uniform(lo, hi, size=(2000, len(lo)))
    err = np.abs(tok.decode(tok.encode(actions)) - actions)
    half_bin = (hi - lo) / (2 * N_BINS)
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


def test_batched_shape_and_dtype(dummy_chunk, action_bounds):
    """A torch tensor input round-trips through the pure-NumPy path."""
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    ids = tok.encode(dummy_chunk)
    assert ids.shape == tuple(dummy_chunk.shape)
    assert ids.dtype == np.int64
    assert ids.min() >= 0 and ids.max() <= N_BINS - 1
    assert tok.decode(ids).shape == tuple(dummy_chunk.shape)


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


def test_from_lerobot_stats(fake_stats):
    tok = ActionTokenizer.from_lerobot_stats(
        fake_stats, key="action", n_bins=N_BINS
    )
    assert tok.action_dim == ACTION_DIM
    np.testing.assert_allclose(tok.lo, fake_stats["action"]["q01"])
    np.testing.assert_allclose(tok.hi, fake_stats["action"]["q99"])


def test_from_lerobot_stats_missing_percentiles_raises():
    """The real dataset's stats have no q01/q99 (only min/max/etc).

    See docs/decisions/000-environment-pins.md. The error must name
    what's missing and point callers at from_lerobot_dataset.
    """
    stats = {
        "action": {
            "min": [-1.0] * ACTION_DIM,
            "max": [1.0] * ACTION_DIM,
            "mean": [0.0] * ACTION_DIM,
            "std": [1.0] * ACTION_DIM,
        }
    }
    with pytest.raises(ValueError, match="from_lerobot_dataset"):
        ActionTokenizer.from_lerobot_stats(stats, key="action")


class _FakeHFDataset:
    """Minimal stand-in for a HF ``Dataset`` column access."""

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        return self._rows


class _FakeLeRobotDataset:
    """Minimal stand-in for lerobot's ``LeRobotDataset`` (no network)."""

    def __init__(self, rows):
        self.hf_dataset = _FakeHFDataset(rows)


def test_from_lerobot_dataset_computes_percentiles():
    rng = np.random.default_rng(4)
    # 1200 frames of 6-dim actions with a few extreme outliers, so a
    # min/max fallback would visibly differ from the 1st/99th
    # percentile bounds this classmethod must produce instead.
    rows = rng.normal(loc=0.0, scale=1.0, size=(1200, ACTION_DIM))
    rows[0] = 500.0
    rows[1] = -500.0
    ds = _FakeLeRobotDataset(rows)

    tok = ActionTokenizer.from_lerobot_dataset(ds, key="action")

    expected_lo = np.percentile(rows, 1, axis=0)
    expected_hi = np.percentile(rows, 99, axis=0)
    np.testing.assert_allclose(tok.lo, expected_lo)
    np.testing.assert_allclose(tok.hi, expected_hi)
    # The saturated outliers must not have stretched the bounds.
    assert tok.hi.max() < 500.0
    assert tok.lo.min() > -500.0


def test_from_lerobot_dataset_accepts_torch_rows():
    """Per-frame torch tensors (the real access pattern) also work."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    base = rng.normal(size=(300, ACTION_DIM))
    rows = [torch.tensor(row, dtype=torch.float32) for row in base]
    ds = _FakeLeRobotDataset(rows)

    tok = ActionTokenizer.from_lerobot_dataset(ds, key="action")

    expected_lo = np.percentile(base, 1, axis=0)
    expected_hi = np.percentile(base, 99, axis=0)
    np.testing.assert_allclose(tok.lo, expected_lo, atol=1e-4)
    np.testing.assert_allclose(tok.hi, expected_hi, atol=1e-4)


def test_reserved_range_defaults_match_chapter_constants(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    ids = tok.encode(lo)  # all bin 0
    token_ids = tok.to_token_ids(ids)
    assert token_ids.min() >= ACT_TOKEN_BASE
    assert token_ids.max() < SMOLLM_VOCAB


def test_to_token_ids_range_and_formula(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    rng = np.random.default_rng(6)
    bin_ids = rng.integers(0, N_BINS, size=(10, len(lo)))
    token_ids = tok.to_token_ids(bin_ids, vocab_size=SMOLLM_VOCAB)
    np.testing.assert_array_equal(
        token_ids, bin_ids + SMOLLM_VOCAB - N_BINS
    )
    assert token_ids.min() >= SMOLLM_VOCAB - N_BINS
    assert token_ids.max() <= SMOLLM_VOCAB - 1


def test_token_id_roundtrip_is_exact_inverse(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    rng = np.random.default_rng(7)
    bin_ids = rng.integers(0, N_BINS, size=(50, len(lo)))
    token_ids = tok.to_token_ids(bin_ids, vocab_size=SMOLLM_VOCAB)
    recovered = tok.from_token_ids(token_ids, vocab_size=SMOLLM_VOCAB)
    np.testing.assert_array_equal(recovered, bin_ids)


def test_to_token_ids_rejects_out_of_range_bin_ids(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    with pytest.raises(ValueError):
        tok.to_token_ids([-1])
    with pytest.raises(ValueError):
        tok.to_token_ids([N_BINS])


def test_from_token_ids_rejects_out_of_range_token_ids(action_bounds):
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    below = SMOLLM_VOCAB - N_BINS - 1
    above = SMOLLM_VOCAB
    with pytest.raises(ValueError):
        tok.from_token_ids([below])
    with pytest.raises(ValueError):
        tok.from_token_ids([above])


def test_shared_vocab_across_dimensions(action_bounds):
    """Same bin in different dimensions maps to the same reserved id."""
    lo, hi = action_bounds
    tok = ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)
    bin_ids = tok.encode(lo)  # bin 0 in every dimension
    token_ids = tok.to_token_ids(bin_ids)
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
