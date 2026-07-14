"""Tests for the training loop (PR 6).

Unit tests run on CPU against the toy ``FakeBackbone`` (no network):
the fake vocab is only 64 wide, so the head uses a small
``n_bins``/``act_token_base`` (8 / 56), matching the AR-head unit tests.
The synthetic loader is a plain list of batch dicts carrying exactly the
keys the real lerobot loader emits -- including the ``[B, 1, 6]`` state
(state-squeeze) and an ``action_is_pad`` mask (pad-masking).
"""

import math

import numpy as np
import pytest
import torch
from fakes import FAKE_WIDTH, FakeBackbone

from ch04.action_tokenizer import ActionTokenizer
from ch04.autoregressive_action_head import AutoregressiveActionHead
from ch04.fusion_adapter import FusionAdapter
from ch04.train import (
    TrainConfig,
    autocast_context,
    build_optimizer,
    load_checkpoint,
    load_config,
    save_checkpoint,
    train,
    unfreeze_backbone,
    warmup_cosine,
)

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


def _batch(seed=0):
    torch.manual_seed(seed)
    return {
        "action": torch.rand(BATCH, CHUNK_H, ACTION_DIM) * 2 - 1,
        "action_is_pad": torch.zeros(
            BATCH, CHUNK_H, dtype=torch.bool
        ),
        "observation.state": torch.rand(BATCH, 1, ACTION_DIM),
        "observation.images.up": torch.rand(BATCH, 3, 224, 224),
        "observation.images.side": torch.rand(BATCH, 3, 224, 224),
        "task": ["pick up the cube"] * BATCH,
    }


# -- warmup_cosine ----------------------------------------------------


def test_warmup_cosine_boundaries():
    assert warmup_cosine(0, warmup=500, total=20_000) == 0.0
    assert warmup_cosine(500, warmup=500, total=20_000) == 1.0
    assert warmup_cosine(20_000, warmup=500, total=20_000) < 1e-9


def test_warmup_cosine_monotone_rise_then_fall():
    warmup, total = 100, 1_000
    rise = [warmup_cosine(s, warmup, total) for s in range(0, 101, 10)]
    assert all(a <= b for a, b in zip(rise, rise[1:]))
    fall = [
        warmup_cosine(s, warmup, total)
        for s in range(100, 1_001, 50)
    ]
    assert all(a >= b for a, b in zip(fall, fall[1:]))


# -- optimizer freeze / unfreeze --------------------------------------


def test_build_optimizer_group_lrs_and_freeze():
    head, fusion = _head()
    opt = build_optimizer(head, fusion, head_lr=1e-4)

    assert opt.param_groups[0]["lr"] == 1e-4
    assert opt.param_groups[1]["lr"] == 0.0
    # Backbone params frozen after build.
    assert all(
        not p.requires_grad for p in fusion.parameters_backbone()
    )


def test_build_optimizer_group1_excludes_siglip():
    head, fusion = _head()
    opt = build_optimizer(head, fusion)

    group1_ids = {id(p) for p in opt.param_groups[1]["params"]}
    vision = fusion.backbone.vision_encoder
    assert group1_ids  # non-empty
    for p in vision.parameters():
        assert id(p) not in group1_ids
    # And the head group must not double-count fusion params.
    group0_ids = {id(p) for p in opt.param_groups[0]["params"]}
    fusion_ids = {id(p) for p in fusion.parameters()}
    assert not (group0_ids & fusion_ids)


def test_unfreeze_backbone_flips_grad_and_sets_lr():
    head, fusion = _head()
    opt = build_optimizer(head, fusion)

    unfreeze_backbone(opt, fusion, backbone_lr=1e-5)

    assert opt.param_groups[1]["lr"] == 1e-5
    assert all(
        p.requires_grad for p in fusion.parameters_backbone()
    )


# -- autocast helper --------------------------------------------------


def test_autocast_context_cpu_is_noop():
    # On CPU the helper is a no-op context: entering it must not change
    # the default dtype of a fresh tensor.
    with autocast_context(torch.device("cpu"), bf16=True):
        x = torch.ones(2, 2)
    assert x.dtype == torch.float32


def test_autocast_context_cuda_branch_is_autocast():
    # Constructing (not entering) the CUDA autocast needs no GPU; this
    # exercises the branch structurally. Torch warns that CUDA is
    # unavailable and disables the autocast -- expected here, silenced.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = autocast_context(torch.device("cuda"), bf16=True)
    assert isinstance(ctx, torch.autocast)


# -- synthetic train() run --------------------------------------------


def test_train_reduces_loss_and_starts_near_uniform():
    torch.manual_seed(0)
    head, fusion = _head()
    tokenizer = _tokenizer()
    # Repeat one fixed batch so the tiny head can overfit and the loss
    # trend is unambiguous; log every step for a dense history.
    loader = [_batch(seed=1)] * 4
    cfg = TrainConfig(
        n_epochs=8,
        grad_accum=1,
        warmup_steps=2,
        total_steps=30,
        steps_per_checkpoint=10_000,
        log_every=1,
        bf16=False,
    )

    history = train(
        head, fusion, tokenizer, loader, cfg, log_fn=lambda *_: None
    )

    assert len(history["loss"]) == 30
    # Fresh head starts near log(n_bins) uniform entropy.
    assert abs(history["loss"][0] - math.log(N_BINS)) < 0.3
    # Training reduces the loss on the overfit batch.
    assert history["loss"][-1] < history["loss"][0]
    # Entropy is a finite canary metric.
    assert all(math.isfinite(e) for e in history["entropy"])


def test_train_scheduler_advances_per_optimizer_step():
    """With ``grad_accum`` microbatches per optimizer step, the number
    of optimizer steps (and logged steps) is microbatches // grad_accum.
    """
    torch.manual_seed(0)
    head, fusion = _head()
    tokenizer = _tokenizer()
    grad_accum = 3
    n_microbatches = 12  # 1 epoch * 12 batches
    loader = [_batch(seed=i) for i in range(n_microbatches)]
    cfg = TrainConfig(
        n_epochs=1,
        grad_accum=grad_accum,
        warmup_steps=1,
        total_steps=1_000,  # not the limiter; loader is
        steps_per_checkpoint=10_000,
        log_every=1,
        bf16=False,
    )

    history = train(
        head, fusion, tokenizer, loader, cfg, log_fn=lambda *_: None
    )

    assert len(history["step"]) == n_microbatches // grad_accum
    assert history["step"] == [1, 2, 3, 4]


def test_train_pad_mask_and_state_squeeze_are_exercised():
    """A batch with all-padded actions and a ``[B, 1, 6]`` state must
    run: the state-squeeze assert passes and the fully-masked loss is
    finite (denominator clamped, not divide-by-zero)."""
    torch.manual_seed(0)
    head, fusion = _head()
    tokenizer = _tokenizer()
    batch = _batch(seed=2)
    batch["action_is_pad"] = torch.ones(
        BATCH, CHUNK_H, dtype=torch.bool
    )
    cfg = TrainConfig(
        n_epochs=1,
        grad_accum=1,
        warmup_steps=1,
        total_steps=2,
        steps_per_checkpoint=10_000,
        log_every=1,
        bf16=False,
    )

    history = train(
        head, fusion, tokenizer, [batch, batch], cfg,
        log_fn=lambda *_: None,
    )
    assert all(math.isfinite(x) for x in history["loss"])


# -- checkpointing ----------------------------------------------------


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    head, fusion = _head()
    # Perturb the readout so save/load has something to verify.
    with torch.no_grad():
        head.readout.weight.add_(1.234)
    path = str(tmp_path / "ckpt.pt")

    save_checkpoint(path, head, fusion, step=42)

    fresh_head, fresh_fusion = _head()
    step = load_checkpoint(path, fresh_head, fresh_fusion)

    assert step == 42
    assert torch.allclose(
        fresh_head.readout.weight, head.readout.weight
    )
    assert torch.allclose(
        fresh_fusion.backbone.embed_tokens.weight,
        fusion.backbone.embed_tokens.weight,
    )


# -- config loading ---------------------------------------------------


@pytest.mark.parametrize(
    "name,microbatch,total,warmup,bf16",
    [
        ("configs/demo.yaml", 8, 800, 100, False),
        ("configs/full.yaml", 16, 20_000, 500, True),
    ],
)
def test_load_config(name, microbatch, total, warmup, bf16):
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = load_config(str(root / name))

    assert isinstance(cfg, TrainConfig)
    assert cfg.microbatch == microbatch
    assert cfg.total_steps == total
    assert cfg.warmup_steps == warmup
    assert cfg.bf16 is bf16


def test_load_config_demo_has_episode_subset():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = load_config(str(root / "configs/demo.yaml"))
    assert cfg.episodes == list(range(10))


def test_load_config_rejects_unknown_key(tmp_path):
    """A typo'd key must raise (naming it), not silently fall back to
    the default and poison a locked-recipe run."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("microbatch_size: 8\ntotal_steps: 800\n")

    with pytest.raises(ValueError, match="microbatch_size"):
        load_config(str(bad))
