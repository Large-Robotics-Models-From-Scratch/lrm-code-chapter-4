"""Tests for the discrete policy and decode-time sampling (PR 7).

Two surfaces are exercised. ``sample_bin`` is the pure decode-time
sampler behind Table 4.4 (argmax / temperature / nucleus /
expected-value); its tests derive expectations by hand on toy logits.
``DiscretePolicy`` is the deployment wrapper (Listing 4.14): unit tests
run against ``FusionAdapter`` on the toy ``FakeBackbone`` and count
``fusion.forward`` calls to prove the KV-cached decode fires exactly
once per chunk and replays the buffer for the next ``H - 1`` calls.

The decode-count test uses ``FakeBackbone(causal=True)`` so a cached
incremental step reproduces a full forward (see ``fakes._CausalFakeLM``);
a position-wise fake would make the cache meaningless.
"""

import math

import numpy as np
import pytest
import torch
from fakes import FAKE_WIDTH, FakeBackbone

from ch04.action_tokenizer import ActionTokenizer
from ch04.autoregressive_action_head import AutoregressiveActionHead
from ch04.fusion_adapter import FusionAdapter
from ch04.policy import DiscretePolicy, sample_bin

N_BINS = 8
ACT_TOKEN_BASE = 56  # 56..63 reserved inside the fake's 64-wide vocab
BOS_ID = 1
D_EMBED = FAKE_WIDTH
CHUNK_H = 4
ACTION_DIM = 6


# -- sample_bin -----------------------------------------------------


def test_sample_bin_argmax_returns_argmax():
    logits = torch.tensor(
        [[0.1, 5.0, 0.2, -1.0], [3.0, 0.0, 0.1, 0.2]]
    )
    out = sample_bin(logits, strategy="argmax")
    assert out.dtype == torch.int64
    assert out.shape == (2,)
    assert out.tolist() == [1, 0]


def test_sample_bin_temperature_low_approaches_argmax():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 1.0, 2.0, 5.0]])
    out = sample_bin(
        logits, strategy="temperature", temperature=1e-4
    )
    assert out.dtype == torch.int64
    assert out.tolist() == [3]


def test_sample_bin_temperature_zero_is_argmax_no_div0():
    logits = torch.tensor([[0.0, 9.0, 1.0, 2.0]])
    out = sample_bin(
        logits, strategy="temperature", temperature=0.0
    )
    assert out.tolist() == [1]


def test_sample_bin_top_p_never_samples_tail():
    """Peaked dist: one bin holds >p mass, tail mass < 1 - p.

    With p = 0.95 and a distribution whose top bin already exceeds
    0.95, the nucleus is that single bin -- 200 draws must never land
    on a tail bin.
    """
    torch.manual_seed(0)
    # softmax of these logits: bin 2 ~ 0.976, others tiny.
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0]]).repeat(200, 1)
    out = sample_bin(
        logits, strategy="temperature_top_p",
        temperature=1.0, top_p=0.95,
    )
    assert out.dtype == torch.int64
    assert out.shape == (200,)
    assert set(out.tolist()) == {2}


def test_sample_bin_expected_value_exact():
    """expected_value == sum_b softmax(logits)_b * centers_b."""
    logits = torch.tensor([[math.log(1.0), math.log(3.0)]])
    centers = torch.tensor([10.0, 20.0])
    out = sample_bin(
        logits, strategy="expected_value", centers=centers
    )
    # p = [0.25, 0.75]; 0.25*10 + 0.75*20 = 17.5
    assert out.dtype == torch.float32
    assert out.shape == (1,)
    assert torch.allclose(out, torch.tensor([17.5]))


def test_sample_bin_expected_value_needs_centers():
    logits = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError):
        sample_bin(logits, strategy="expected_value")


def test_sample_bin_unknown_strategy_raises():
    with pytest.raises(ValueError):
        sample_bin(torch.zeros(1, 4), strategy="nope")


# -- DiscretePolicy -------------------------------------------------


class _CountingFusion(FusionAdapter):
    """Wrap the adapter to count ``forward`` (decode-step) calls."""

    def __init__(self, backbone):
        super().__init__(backbone)
        self.forward_calls = 0
        self.encode_calls = 0

    def encode_prefix(self, batch):
        self.encode_calls += 1
        return super().encode_prefix(batch)

    def forward(self, seq_embeds, past_key_values=None, use_cache=False):
        self.forward_calls += 1
        return super().forward(
            seq_embeds, past_key_values=past_key_values,
            use_cache=use_cache,
        )


def _obs():
    torch.manual_seed(0)
    return {
        "observation.images.up": torch.rand(1, 3, 224, 224),
        "observation.images.side": torch.rand(1, 3, 224, 224),
        "observation.state": torch.rand(1, 6),
        "task": ["pick up the cube"],
    }


def _tokenizer():
    lo = -np.ones(ACTION_DIM, dtype=np.float64)
    hi = np.ones(ACTION_DIM, dtype=np.float64)
    return ActionTokenizer(lo=lo, hi=hi, n_bins=N_BINS)


def _policy(strategy="argmax", counting=True):
    bb = FakeBackbone(causal=True)
    fusion = _CountingFusion(bb) if counting else FusionAdapter(bb)
    head = AutoregressiveActionHead(
        fusion, d_embed=D_EMBED, n_bins=N_BINS,
        act_token_base=ACT_TOKEN_BASE, bos_id=BOS_ID,
    )
    tokenizer = _tokenizer()
    policy = DiscretePolicy(
        fusion, head, tokenizer, chunk_h=CHUNK_H,
        action_dim=ACTION_DIM, strategy=strategy,
    )
    return policy, fusion


def test_select_action_shape_and_dtype():
    policy, _ = _policy()
    action = policy.select_action(_obs())
    assert isinstance(action, np.ndarray)
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(action))


def test_decode_fires_once_per_chunk():
    """One decode per chunk, buffer replay for the next H-1 calls."""
    policy, fusion = _policy()
    obs = _obs()

    policy.select_action(obs)
    # 1 seed forward + (H*D - 1) incremental steps = H*D forwards.
    expected_forwards = CHUNK_H * ACTION_DIM
    assert fusion.encode_calls == 1
    assert fusion.forward_calls == expected_forwards

    # Next H-1 calls replay the buffer: zero new work.
    for _ in range(CHUNK_H - 1):
        policy.select_action(obs)
    assert fusion.encode_calls == 1
    assert fusion.forward_calls == expected_forwards

    # Call H+1 exhausts the buffer -> fresh decode.
    policy.select_action(obs)
    assert fusion.encode_calls == 2
    assert fusion.forward_calls == 2 * expected_forwards


def test_reset_forces_redecode():
    policy, fusion = _policy()
    obs = _obs()
    policy.select_action(obs)
    base_forwards = fusion.forward_calls
    policy.reset()
    policy.select_action(obs)
    assert fusion.forward_calls == 2 * base_forwards
    assert fusion.encode_calls == 2


def test_actions_equal_tokenizer_decode_of_sampled_bins():
    """Returned chunk == tokenizer.decode of the argmax bins.

    Re-run the decode by hand (argmax is deterministic) and confirm the
    H actions the policy hands back over one chunk equal
    ``tokenizer.decode`` of the bins the head would pick.
    """
    policy, _ = _policy(strategy="argmax")
    obs = _obs()
    tokenizer = policy.tokenizer

    got = np.stack(
        [policy.select_action(obs) for _ in range(CHUNK_H)], axis=0
    )
    bins = _reference_argmax_bins(policy, obs)
    expected = tokenizer.decode(bins)  # [H, D]
    assert got.shape == (CHUNK_H, ACTION_DIM)
    assert np.allclose(got, expected)


def _reference_argmax_bins(policy, obs):
    """Independently reproduce the policy's KV-cached argmax decode."""
    fusion = policy.fusion
    head = policy.head
    horizon = policy.chunk_h * policy.action_dim
    with torch.no_grad():
        batch = policy._obs_to_batch(obs)
        prefix = fusion.encode_prefix(batch)
        bos = torch.full((1, 1), head.bos_id, dtype=torch.long)
        step_emb = fusion.embed(bos).to(prefix.dtype)
        seq = torch.cat([prefix, step_emb], dim=1)
        hidden, past = fusion.forward(seq, use_cache=True)
        bins = []
        for _ in range(horizon):
            logits = head.readout(hidden[:, -1, :])
            b = int(torch.argmax(logits, dim=-1).item())
            bins.append(b)
            tok = torch.full(
                (1, 1), head.act_token_base + b, dtype=torch.long
            )
            step_emb = fusion.embed(tok).to(prefix.dtype)
            hidden, past = fusion.forward(
                step_emb, past_key_values=past, use_cache=True
            )
    return np.array(bins).reshape(policy.chunk_h, policy.action_dim)


def test_buffer_order_matches_h_by_d_rows():
    """A rigged head forcing bin = (token index % n_bins) yields a
    buffer whose row h, column d holds bin (h*D + d) % n_bins."""
    policy, _ = _policy(strategy="argmax")
    obs = _obs()
    horizon = policy.chunk_h * policy.action_dim

    # Rig readout so step i (0-indexed over the decode) is argmax at
    # bin (i % n_bins): a monotonic ramp keyed to the running position.
    head = policy.head
    real_readout = head.readout

    class _RiggedReadout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.i = 0

        def forward(self, hidden_last):
            logits = torch.full((1, N_BINS), -10.0)
            logits[0, self.i % N_BINS] = 10.0
            self.i += 1
            return logits

    head.readout = _RiggedReadout()
    try:
        rows = np.stack(
            [policy.select_action(obs) for _ in range(policy.chunk_h)],
            axis=0,
        )
    finally:
        head.readout = real_readout

    expected_bins = (
        np.arange(horizon) % N_BINS
    ).reshape(policy.chunk_h, policy.action_dim)
    expected = policy.tokenizer.decode(expected_bins)
    assert np.allclose(rows, expected)


# -- integration ----------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_integration_real_backbone_one_decode():
    from ch03 import UnifiedEmbeddingBackbone

    from ch04 import ACT_TOKEN_BASE as REAL_BASE
    from ch04 import ACTION_DIM as REAL_DIM
    from ch04 import CHUNK_H as REAL_H
    from ch04 import N_BINS as REAL_BINS

    torch.manual_seed(0)
    backbone = UnifiedEmbeddingBackbone().float()
    backbone.eval()
    fusion = _CountingFusion(backbone)
    head = AutoregressiveActionHead(
        fusion, d_embed=576, n_bins=REAL_BINS,
        act_token_base=REAL_BASE, bos_id=1,
    )
    lo = -np.ones(REAL_DIM, dtype=np.float64)
    hi = np.ones(REAL_DIM, dtype=np.float64)
    tokenizer = ActionTokenizer(lo=lo, hi=hi, n_bins=REAL_BINS)
    policy = DiscretePolicy(
        fusion, head, tokenizer, chunk_h=REAL_H,
        action_dim=REAL_DIM, strategy="argmax",
    )
    obs = {
        "observation.images.up": torch.rand(1, 3, 224, 224),
        "observation.images.side": torch.rand(1, 3, 224, 224),
        "observation.state": torch.rand(1, 6),
        "task": ["pick up the cube"],
    }

    action = policy.select_action(obs)
    assert action.shape == (REAL_DIM,)
    assert np.all(np.isfinite(action))

    expected_forwards = REAL_H * REAL_DIM
    assert fusion.encode_calls == 1
    assert fusion.forward_calls == expected_forwards
    # A full chunk cycle (H calls) triggers exactly one decode.
    for _ in range(REAL_H - 1):
        policy.select_action(obs)
    assert fusion.encode_calls == 1
    assert fusion.forward_calls == expected_forwards
