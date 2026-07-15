"""Tests for the diagnostics dashboard and figure generators (PR 8).

The numeric diagnostics (``softmax_entropy``, ``bin_frequency_histogram``,
``canary_snapshot``) are checked against hand-derived expectations on
toy inputs and the ``FakeBackbone``. The three ``plot_*`` figure
generators (§4.6.5-4.6.7, Figures 4.8-4.10) are smoke-tested: each must
write a nonempty, valid PNG to a ``tmp_path`` and return that path.
Grayscale-safety is enforced by construction (the plotting code uses
only gray levels, hatching, and line styles -- never color as the sole
differentiator), so the tests assert the file is produced rather than
re-deriving the style guide.
"""

import math

import numpy as np
import pytest
import torch
from fakes import FakeBackbone

from ch04.action_tokenizer import ActionTokenizer
from ch04.autoregressive_action_head import AutoregressiveActionHead
from ch04.diagnostics import (
    bin_frequency_histogram,
    canary_snapshot,
    plot_bimodal_comparison,
    plot_convergence_ridges,
    plot_joint_coordination,
    softmax_entropy,
)
from ch04.fusion_adapter import FusionAdapter

# -- softmax_entropy -------------------------------------------------


def test_softmax_entropy_uniform_is_log_n_bins():
    n_bins = 256
    logits = torch.zeros(1, n_bins)  # uniform softmax
    ent = softmax_entropy(logits)
    assert ent.shape == (1,)
    assert float(ent[0]) == pytest.approx(math.log(n_bins), abs=1e-4)


def test_softmax_entropy_one_hot_is_zero():
    logits = torch.full((1, 256), -1e4)
    logits[0, 7] = 1e4  # essentially one-hot
    ent = softmax_entropy(logits)
    assert float(ent[0]) == pytest.approx(0.0, abs=1e-3)


def test_softmax_entropy_preserves_leading_dims():
    logits = torch.zeros(2, 3, 16)
    ent = softmax_entropy(logits)
    assert ent.shape == (2, 3)


def test_softmax_entropy_accepts_numpy():
    logits = np.zeros((1, 8))
    ent = softmax_entropy(logits)
    assert float(np.asarray(ent)[0]) == pytest.approx(math.log(8))


# -- bin_frequency_histogram -----------------------------------------


class _FakeLoader:
    """Yields batches carrying an ``action`` chunk ``[B, H, D]``."""

    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


def test_bin_frequency_histogram_sums_to_token_count():
    lo = -np.ones(6)
    hi = np.ones(6)
    tok = ActionTokenizer(lo, hi, n_bins=256)
    b, h, d = 2, 16, 6
    batches = [
        {"action": torch.rand(b, h, d) * 2 - 1} for _ in range(3)
    ]
    hist = bin_frequency_histogram(_FakeLoader(batches), tok)
    assert hist.shape == (256,)
    # Every action token counted exactly once.
    assert int(hist.sum()) == 3 * b * h * d


def test_bin_frequency_histogram_respects_n_batches():
    lo = -np.ones(6)
    hi = np.ones(6)
    tok = ActionTokenizer(lo, hi, n_bins=256)
    batches = [
        {"action": torch.zeros(1, 16, 6)} for _ in range(5)
    ]
    hist = bin_frequency_histogram(
        _FakeLoader(batches), tok, n_batches=2
    )
    assert int(hist.sum()) == 2 * 1 * 16 * 6


# -- canary_snapshot -------------------------------------------------


def _fake_stack():
    torch.manual_seed(0)
    backbone = FakeBackbone(causal=True)
    fusion = FusionAdapter(backbone)
    head = AutoregressiveActionHead(
        fusion, d_embed=8, n_bins=8, act_token_base=56, bos_id=1
    )
    return fusion, head


def _fake_batch(fusion):
    return {
        "observation.images.up": torch.rand(1, 3, 8, 8),
        "observation.images.side": torch.rand(1, 3, 8, 8),
        "observation.state": torch.zeros(1, 6),
        "task": ["do the thing"],
    }


def test_canary_snapshot_shape_and_normalized():
    fusion, head = _fake_stack()
    batch = _fake_batch(fusion)
    h, d, n_bins = 2, 6, 8
    target_bins = torch.zeros(1, h * d, dtype=torch.long)
    probs = canary_snapshot(head, fusion, batch, target_bins)
    assert probs.shape == (h * d, n_bins)
    # Each position's softmax sums to 1.
    row_sums = probs.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


# -- plot smoke tests ------------------------------------------------


def _valid_png(path):
    assert path.exists()
    data = path.read_bytes()
    assert len(data) > 0
    # PNG magic number.
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_convergence_ridges_writes_png(tmp_path):
    n_bins = 64
    centers = np.linspace(-1, 1, n_bins)
    # Three snapshots: uniform -> emerging bimodal -> sharp bimodal.
    snaps = []
    for sharp in (0.0, 4.0, 12.0):
        left = np.exp(-sharp * (centers + 0.5) ** 2)
        right = np.exp(-sharp * (centers - 0.5) ** 2)
        p = left + right + 1e-3
        snaps.append(p / p.sum())
    out = tmp_path / "fig48.png"
    result = plot_convergence_ridges(
        np.array(snaps), out, steps=[0, 5000, 20000],
        centers=centers,
    )
    _valid_png(out)
    assert str(result) == str(out)


def test_plot_bimodal_comparison_writes_png(tmp_path):
    n_bins = 64
    centers = np.linspace(-1, 1, n_bins)
    left = np.exp(-8 * (centers + 0.5) ** 2)
    right = np.exp(-8 * (centers - 0.5) ** 2)
    cat_probs = (left + right)
    cat_probs = cat_probs / cat_probs.sum()
    out = tmp_path / "fig49.png"
    result = plot_bimodal_comparison(
        mse_pred=0.0, cat_probs=cat_probs, centers=centers,
        out_path=out,
    )
    _valid_png(out)
    assert str(result) == str(out)


def test_plot_joint_coordination_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    # AR samples on the diagonal; parallel leaks off-diagonal.
    diag = rng.integers(0, 32, size=200)
    ar = np.stack([diag, diag], axis=1)
    par = np.stack(
        [rng.integers(0, 32, 200), rng.integers(0, 32, 200)],
        axis=1,
    )
    out = tmp_path / "fig410.png"
    result = plot_joint_coordination(
        par_samples=par, ar_samples=ar, out_path=out, n_bins=32
    )
    _valid_png(out)
    assert str(result) == str(out)
