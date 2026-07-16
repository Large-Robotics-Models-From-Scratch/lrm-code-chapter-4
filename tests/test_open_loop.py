"""Tests for the open-loop eval helpers (PR 9c).

Open-loop validation loss is Chapter 4's *primary* honest metric: it
feeds the policy held-out ground-truth states and scores the head's
teacher-forced cross-entropy on data it never trained on (see
``docs/EVAL_INCONSISTENCY.md`` for why the ManiSkill closed-loop path
is demoted to optional). These tests exercise the two deliverables on
the toy ``FakeBackbone`` -- no network, no simulator:

* ``evaluate_open_loop`` returns a finite ``val_loss``, a
  ``per_dim_loss`` of length ``D``, an integer ``n_tokens``, and
  correctly drops fully-padded batches from the token-weighted mean
  (the pad mask must zero their contribution).
* ``plot_action_traces`` writes a valid PNG of predicted-vs-ground-truth
  joint traces, and ``open_loop_episode_predictions`` returns aligned
  ``[H, D]`` prediction/truth arrays from a real decode.
"""

import math

import numpy as np
import torch
from fakes import FAKE_WIDTH, FakeBackbone

from ch04.action_tokenizer import ActionTokenizer
from ch04.autoregressive_action_head import AutoregressiveActionHead
from ch04.diagnostics import (
    open_loop_episode_predictions,
    plot_action_traces,
)
from ch04.fusion_adapter import FusionAdapter
from ch04.policy import DiscretePolicy
from ch04.rollout import evaluate_open_loop

N_BINS = 8
ACT_TOKEN_BASE = 56
D_EMBED = FAKE_WIDTH
CHUNK_H = 2
ACTION_DIM = 6
BATCH = 2


def _head(causal=False):
    fusion = FusionAdapter(FakeBackbone(causal=causal))
    head = AutoregressiveActionHead(
        fusion,
        d_embed=D_EMBED,
        n_bins=N_BINS,
        act_token_base=ACT_TOKEN_BASE,
        bos_id=1,
    )
    return head, fusion


def _tokenizer():
    lo = -np.ones(ACTION_DIM, dtype=np.float32)
    hi = np.ones(ACTION_DIM, dtype=np.float32)
    return ActionTokenizer(lo, hi, n_bins=N_BINS)


def _batch(seed=0, all_pad=False):
    torch.manual_seed(seed)
    pad = torch.ones if all_pad else torch.zeros
    return {
        "action": torch.rand(BATCH, CHUNK_H, ACTION_DIM) * 2 - 1,
        "action_is_pad": pad(BATCH, CHUNK_H, dtype=torch.bool),
        "observation.state": torch.rand(BATCH, 1, ACTION_DIM),
        "observation.images.up": torch.rand(BATCH, 3, 8, 8),
        "observation.images.side": torch.rand(BATCH, 3, 8, 8),
        "task": ["pick up the cube"] * BATCH,
    }


# -- evaluate_open_loop ----------------------------------------------


def test_evaluate_open_loop_returns_finite_metrics():
    torch.manual_seed(0)
    head, fusion = _head()
    tokenizer = _tokenizer()
    loader = [_batch(seed=1), _batch(seed=2)]

    out = evaluate_open_loop(head, fusion, tokenizer, loader)

    assert math.isfinite(out["val_loss"])
    assert len(out["per_dim_loss"]) == ACTION_DIM
    assert all(math.isfinite(x) for x in out["per_dim_loss"])
    # Two full batches, no padding -> every action token counted.
    assert out["n_tokens"] == 2 * BATCH * CHUNK_H * ACTION_DIM
    # A fresh categorical head sits near the log(n_bins) uniform scale.
    assert abs(out["val_loss"] - math.log(N_BINS)) < 2.0


def test_evaluate_open_loop_respects_n_batches():
    torch.manual_seed(0)
    head, fusion = _head()
    tokenizer = _tokenizer()
    loader = [_batch(seed=i) for i in range(4)]

    out = evaluate_open_loop(
        head, fusion, tokenizer, loader, n_batches=2
    )
    assert out["n_tokens"] == 2 * BATCH * CHUNK_H * ACTION_DIM


def test_evaluate_open_loop_ignores_fully_padded_batch():
    """A fully-padded batch must contribute 0 tokens and must not move
    the token-weighted mean -- the pad mask has to work."""
    torch.manual_seed(0)
    head, fusion = _head()
    tokenizer = _tokenizer()
    real = _batch(seed=1)
    padded = _batch(seed=1, all_pad=True)

    only_real = evaluate_open_loop(head, fusion, tokenizer, [real])
    with_pad = evaluate_open_loop(
        head, fusion, tokenizer, [real, padded]
    )

    assert with_pad["n_tokens"] == only_real["n_tokens"]
    assert with_pad["val_loss"] == only_real["val_loss"]


def test_evaluate_open_loop_all_padded_is_zero_tokens():
    torch.manual_seed(0)
    head, fusion = _head()
    tokenizer = _tokenizer()
    padded = _batch(seed=1, all_pad=True)

    out = evaluate_open_loop(head, fusion, tokenizer, [padded])
    assert out["n_tokens"] == 0
    assert math.isfinite(out["val_loss"])  # clamped, not div-by-zero


# -- plot_action_traces ----------------------------------------------


def _valid_png(path):
    assert path.exists()
    data = path.read_bytes()
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_action_traces_writes_png(tmp_path):
    steps, dim = 16, 6
    rng = np.random.default_rng(0)
    true = np.cumsum(rng.normal(size=(steps, dim)), axis=0)
    pred = true + rng.normal(scale=0.2, size=(steps, dim))
    out = tmp_path / "traces.png"
    result = plot_action_traces(
        pred, true, out,
        joint_names=["shoulder", "upper", "elbow", "wrist",
                     "roll", "gripper"],
    )
    _valid_png(out)
    assert str(result) == str(out)


def test_plot_action_traces_defaults_joint_names(tmp_path):
    out = tmp_path / "traces2.png"
    pred = np.zeros((8, 3))
    true = np.ones((8, 3))
    plot_action_traces(pred, true, out)
    _valid_png(out)


# -- open_loop_episode_predictions -----------------------------------


def test_open_loop_episode_predictions_shapes():
    torch.manual_seed(0)
    head, fusion = _head(causal=True)
    tokenizer = _tokenizer()
    policy = DiscretePolicy(
        fusion, head, tokenizer,
        chunk_h=CHUNK_H, action_dim=ACTION_DIM,
        strategy="argmax",
    )
    batch = _batch(seed=3)

    pred, true = open_loop_episode_predictions(policy, batch, index=0)

    assert pred.shape == (CHUNK_H, ACTION_DIM)
    assert true.shape == (CHUNK_H, ACTION_DIM)
    assert np.all(np.isfinite(pred))
