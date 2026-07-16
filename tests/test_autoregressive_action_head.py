"""Tests for the autoregressive action head (the shipped head).

Unit tests run against ``FusionAdapter`` on the toy ``FakeBackbone``
(no network). The head maps action bins to reserved vocabulary ids,
shift-rights with a BOS token, appends the embeddings to the fused
prefix, and reads out the last ``T`` positions of a causal forward
(manuscript Listing 4.10). Because the fake vocab is only 64 wide, the
unit tests use a small ``n_bins`` / ``act_token_base`` (8 / 56) so the
reserved ids stay inside the table; the integration test exercises the
chapter's real 256-bin / 48896-base configuration.

The causality test runs against ``FakeBackbone(causal=True)`` -- the
default fake LM is a position-wise map, so a causality check against it
would be vacuous (see ``fakes._CausalFakeLM``).
"""

import math

import pytest
import torch
from fakes import FAKE_WIDTH, FakeBackbone

from ch04.autoregressive_action_head import AutoregressiveActionHead
from ch04.fusion_adapter import FusionAdapter

N_BINS = 8
ACT_TOKEN_BASE = 56  # 56..63 reserved inside the fake's 64-wide vocab
BOS_ID = 1
D_EMBED = FAKE_WIDTH
BATCH = 2
HORIZON = 5  # T = H * D; toy value, real head uses 16 * 6 = 96


def _head(causal: bool = False):
    fusion = FusionAdapter(FakeBackbone(causal=causal))
    return AutoregressiveActionHead(
        fusion,
        d_embed=D_EMBED,
        n_bins=N_BINS,
        act_token_base=ACT_TOKEN_BASE,
        bos_id=BOS_ID,
    )


def _prefix(batch: int = BATCH, prefix_len: int = 4):
    return torch.randn(batch, prefix_len, D_EMBED) * 0.1


def test_forward_returns_scalar_near_uniform_entropy():
    """Zero-init readout bias => fresh loss near ``log(n_bins)``.

    The manuscript's sanity check (scoped there to "a fake batch"):
    an untrained categorical head reports loss close to the entropy of
    a uniform distribution over ``n_bins``, not an arbitrary number.
    The readout's random weight (default ``nn.Linear`` init) over the
    small width-8 fake hidden perturbs the logits only slightly, so the
    loss lands within ~0.1 of ``log(8)`` (measured ~0.06-0.11 across
    seeds); tolerance 0.2 leaves margin. On the real 576-wide backbone
    the same random weight lifts the loss further above uniform -- see
    the integration test.
    """
    torch.manual_seed(0)
    head = _head()
    prefix = _prefix()
    targets = torch.randint(0, N_BINS, (BATCH, HORIZON))

    loss = head(prefix, targets)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert abs(loss.item() - math.log(N_BINS)) < 0.2


def test_logits_shape():
    torch.manual_seed(0)
    head = _head()
    prefix = _prefix()
    targets = torch.randint(0, N_BINS, (BATCH, HORIZON))

    logits = head.logits(prefix, targets)

    assert logits.shape == (BATCH, HORIZON, N_BINS)


def test_causality_last_input_token_influences_only_final_readout():
    """The last token that enters the shifted input influences only
    the final readout position; every earlier position is unchanged.

    Shift-right + a causal LM means readout position ``i`` depends only
    on target tokens strictly before ``i``. The true last target
    (index ``-1``) is a label only -- the shift drops it, so it feeds
    no input; the last token that actually enters the sequence is index
    ``-2``, and it must reach only the final readout position. Run
    against ``FakeBackbone(causal=True)`` -- the default position-wise
    fake LM would pass this vacuously.
    """
    torch.manual_seed(0)
    head = _head(causal=True)
    prefix = _prefix(batch=1)
    targets = torch.randint(0, N_BINS, (1, HORIZON))

    base = head.logits(prefix, targets)

    perturbed = targets.clone()
    perturbed[0, -2] = (targets[0, -2] + 1) % N_BINS
    after = head.logits(prefix, perturbed)

    assert torch.allclose(base[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(base[:, -1], after[:, -1])


def test_token_mapping_shift_right_with_bos():
    """Ids handed to ``fusion.embed`` are ``act_token_base +
    target_bins`` shifted right, with ``bos_id`` first."""
    torch.manual_seed(0)
    head = _head()
    prefix = _prefix()
    targets = torch.randint(0, N_BINS, (BATCH, HORIZON))

    captured = {}
    real_embed = head.fusion.embed

    def recording_embed(token_ids):
        captured["ids"] = token_ids.clone()
        return real_embed(token_ids)

    head.fusion.embed = recording_embed
    head.logits(prefix, targets)

    tok = ACT_TOKEN_BASE + targets
    start = torch.full((BATCH, 1), BOS_ID, dtype=tok.dtype)
    expected = torch.cat([start, tok[:, :-1]], dim=1)
    assert torch.equal(captured["ids"], expected)


def test_pad_mask_ignores_masked_positions():
    """A masked (padded) target must not affect the loss.

    lerobot repeats the last action past an episode boundary and flags
    those steps in ``action_is_pad``; the head drops them from the CE
    mean. We corrupt the *last* target position -- teacher forcing
    shifts the input right and drops the last target, so changing it
    touches no logits -- and mask that position: the loss must be
    identical to the uncorrupted run, and differ from an unmasked run.
    """
    torch.manual_seed(0)
    head = _head()
    prefix = _prefix()
    targets = torch.randint(0, N_BINS, (BATCH, HORIZON))

    mask = torch.zeros(BATCH, HORIZON, dtype=torch.bool)
    mask[:, -1] = True  # mark the last position as padding

    corrupted = targets.clone()
    corrupted[:, -1] = (targets[:, -1] + 1) % N_BINS

    masked_clean = head(prefix, targets, pad_mask=mask)
    masked_corrupt = head(prefix, corrupted, pad_mask=mask)
    assert torch.allclose(masked_clean, masked_corrupt, atol=1e-6)

    unmasked = head(prefix, targets)
    assert not torch.allclose(masked_clean, unmasked)


def test_out_of_range_targets_raise():
    head = _head()
    prefix = _prefix()

    too_high = torch.randint(0, N_BINS, (BATCH, HORIZON))
    too_high[0, 0] = N_BINS  # one past the last valid bin
    with pytest.raises(ValueError, match="target_bins"):
        head.logits(prefix, too_high)

    negative = torch.randint(0, N_BINS, (BATCH, HORIZON))
    negative[0, 0] = -1
    with pytest.raises(ValueError, match="target_bins"):
        head.logits(prefix, negative)


def test_float_targets_raise():
    head = _head()
    prefix = _prefix()
    targets = torch.rand(BATCH, HORIZON)  # float dtype
    with pytest.raises(ValueError, match="integer"):
        head.logits(prefix, targets)


def test_int32_targets_train_ok():
    """int32 bins are valid (``_validate`` accepts any integer dtype),
    so ``forward`` must not crash inside ``cross_entropy`` -- it coerces
    the CE targets to int64 and returns a finite scalar loss.
    """
    torch.manual_seed(0)
    head = _head()
    prefix = _prefix()
    targets = torch.randint(
        0, N_BINS, (BATCH, HORIZON), dtype=torch.int32
    )

    loss = head(prefix, targets)

    assert loss.shape == ()
    assert torch.isfinite(loss)


@pytest.mark.integration
@pytest.mark.slow
def test_integration_real_backbone_fresh_loss():
    """Real backbone + adapter: finite scalar loss, ``[1, T, 256]``
    logits, fresh loss in the uniform-scale band.

    The manuscript's tight "within ~0.1 of log(256)" is scoped to a
    *fake batch* (small hidden width); on the real 576-wide backbone
    the default random ``readout`` weight lifts the fresh loss above
    pure uniform (measured ~6.8 vs log(256) ~5.545). We cannot zero
    the readout weight to force exact uniform -- that would make the
    logits input-independent and void the causality test. So we assert
    the loss is finite and still in the uniform *scale* band (well
    below the ~11-13 nats a confidently-wrong head would report),
    matching the manuscript's fake-batch sanity intent without the
    fake-only precision. See task-5 report for the manuscript follow-up.
    """
    from ch03 import UnifiedEmbeddingBackbone

    from ch04 import ACT_TOKEN_BASE as REAL_BASE
    from ch04 import ACTION_DIM, CHUNK_H
    from ch04 import N_BINS as REAL_BINS

    torch.manual_seed(0)
    backbone = UnifiedEmbeddingBackbone().float()
    backbone.eval()
    fusion = FusionAdapter(backbone)
    head = AutoregressiveActionHead(
        fusion,
        d_embed=576,
        n_bins=REAL_BINS,
        act_token_base=REAL_BASE,
        bos_id=1,
    )

    instruction = "pick up the cube"
    batch = {
        "observation.images.up": torch.rand(1, 3, 224, 224),
        "observation.images.side": torch.rand(1, 3, 224, 224),
        "observation.state": torch.rand(1, 6),
        "task": [instruction],
    }
    horizon = CHUNK_H * ACTION_DIM
    targets = torch.randint(0, REAL_BINS, (1, horizon))

    with torch.no_grad():
        prefix = fusion.encode_prefix(batch)
        logits = head.logits(prefix, targets)
        assert logits.shape == (1, horizon, REAL_BINS)

        loss = head(prefix, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        # Uniform-scale band (see docstring): fresh loss stays near
        # the log(256) uniform entropy, not collapsed low or blown up.
        assert abs(loss.item() - math.log(REAL_BINS)) < 2.0
