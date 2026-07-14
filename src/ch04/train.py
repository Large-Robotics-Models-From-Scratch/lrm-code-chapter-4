"""Training loop for the autoregressive action head (Listing 4.13).

This module turns the pieces PRs 1-5 built -- the action tokenizer, the
fusion adapter, and the autoregressive head -- into a training run. The
core loop stays faithful to the manuscript's Listing 4.13 (tokenize the
action chunk, flatten to a token sequence, ``encode_prefix``, take the
head's cross-entropy loss, clip, step) and adds the machinery a real run
needs: gradient accumulation, mixed-precision autocast, a warmup +
cosine learning-rate schedule, a two-group optimizer that freezes the
backbone for one epoch then unfreezes it, entropy/loss logging, and
checkpointing.

Two data-contract requirements from PR 5 (chunked loading) are honored
here, not inside the model modules:

* **Pad masking.** lerobot pads episode-boundary chunks by repeating the
  last action and flags those steps in ``action_is_pad`` (``[B, H]``
  bool). Training on the repeated rows teaches the policy to freeze at
  episode end, so we expand the frame mask to token level
  (``repeat_interleave`` by ``action_dim``) and pass it to the head,
  which drops the masked positions from the loss.
* **State squeeze.** ``batch["observation.state"]`` arrives ``[B, 1, 6]``
  (lerobot keeps the chunk dim for tabular keys) but the fusion adapter
  expects ``[B, 6]``; we squeeze dim 1 with a shape assert before
  encoding.

The recipe constants (manuscript Table 4.3) are locked module-level:
AdamW betas ``(0.9, 0.95)``, weight decay ``0.05``, head LR ``1e-4``,
backbone LR ``1e-5`` (after unfreeze), grad clip ``1.0``.

**Freeze pitfall (Table 4.3 / SS4.6):** epoch 0 trains only the fresh
head. If the backbone trained from step 0, the random-init head's large
early gradients would flow back and corrupt ch3's pretrained SmolLM2 /
projection weights before the head learned anything useful. So
``build_optimizer`` freezes the backbone group (``requires_grad=False``,
LR ``0.0``) and ``unfreeze_backbone`` flips it on at epoch 1.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from ch04 import ACTION_DIM

HEAD_LR = 1e-4
BACKBONE_LR = 1e-5
GRAD_CLIP = 1.0
ADAMW_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.05


def warmup_cosine(step, warmup=500, total=20_000):
    """LR multiplier: linear 0->1 over warmup, cosine 1->0 after.

    Returns a scalar in ``[0, 1]`` to scale each group's base LR.
    ``warmup_cosine(0) == 0`` (the very first update barely moves),
    ``warmup_cosine(warmup) == 1`` (peak), and it decays to ``0`` at
    ``total`` and stays there. Rises monotonically over the warmup
    window, then falls monotonically over the cosine window.
    """
    if warmup > 0 and step < warmup:
        return step / warmup
    if step >= total:
        return 0.0
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(head, fusion, head_lr=HEAD_LR):
    """Two AdamW groups: head @ ``head_lr``, backbone @ ``0.0``.

    Group 0 is the head's own parameters (the readout) at ``head_lr``.
    Group 1 is the trainable backbone (``fusion.parameters_backbone``:
    ``img_proj``, ``state_proj``, ``embed_tokens``, ``language_backbone``
    -- frozen SigLIP excluded by module). The backbone group starts
    frozen: we set ``requires_grad=False`` on those params *before*
    building and give the group LR ``0.0`` (the freeze pitfall above).

    ``head.parameters()`` includes the fusion submodule (the AR head
    holds a reference to it), so we exclude every fusion parameter from
    group 0 by identity -- otherwise the backbone would be optimized
    twice, once at ``head_lr``.
    """
    for p in fusion.parameters_backbone():
        p.requires_grad = False

    fusion_param_ids = {id(p) for p in fusion.parameters()}
    head_params = [
        p for p in head.parameters()
        if id(p) not in fusion_param_ids
    ]
    backbone_params = list(fusion.parameters_backbone())
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr},
            {"params": backbone_params, "lr": 0.0},
        ],
        betas=ADAMW_BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def unfreeze_backbone(opt, fusion, backbone_lr=BACKBONE_LR):
    """Unfreeze the backbone group at epoch 1.

    Flips ``requires_grad=True`` on ``fusion.parameters_backbone`` and
    raises group 1's LR from ``0.0`` to ``backbone_lr`` so the backbone
    starts adapting once the head is no longer emitting random-init
    gradients.
    """
    for p in fusion.parameters_backbone():
        p.requires_grad = True
    opt.param_groups[1]["lr"] = backbone_lr


@dataclass
class TrainConfig:
    n_epochs: int = 4
    steps_per_checkpoint: int = 5_000
    grad_accum: int = 2
    microbatch: int = 16
    warmup_steps: int = 500
    total_steps: int = 20_000
    bf16: bool = True
    log_every: int = 50
    out_dir: str = "checkpoints"
    # Episode subset for the dataset builder (None = all episodes).
    # The demo config trains on a handful of episodes so a smoke run
    # finishes quickly; the full recipe leaves this None.
    episodes: list[int] | None = None


@contextmanager
def _null_context():
    yield


def autocast_context(device, bf16):
    """Mixed-precision context for the forward/loss.

    On CUDA, autocast in ``bfloat16`` when ``bf16`` (Ampere+), else
    ``float16``. On CPU (where the unit tests run) there is no autocast
    -- return a no-op context. Factored out so the branch is unit
    testable without a GPU.
    """
    if device.type == "cuda":
        dtype = torch.bfloat16 if bf16 else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return _null_context()


def _prepare_batch(batch, tokenizer, device):
    """Listing 4.13 encoding path + the PR 5 data-contract fixes.

    Returns ``(model_batch, bins, pad_mask)`` where ``model_batch`` has
    a squeezed ``[B, 6]`` state, ``bins`` is the flattened target token
    ids ``[B, H * D]``, and ``pad_mask`` is the token-level ``[B, H * D]``
    ignore mask (or ``None`` when the batch carries no ``action_is_pad``).
    """
    state = batch["observation.state"]
    assert state.dim() == 3 and state.shape[1] == 1, (
        f"expected state [B, 1, 6]; got {tuple(state.shape)}"
    )
    model_batch = dict(batch)
    model_batch["observation.state"] = state.squeeze(1).to(device)

    action = batch["action"]  # [B, H, D]
    bins_np = tokenizer.encode(action.numpy())  # [B, H, D] int
    bins = torch.from_numpy(bins_np).long().to(device)
    batch_size = bins.shape[0]
    bins = bins.reshape(batch_size, -1)  # [B, H * D]

    pad_mask = None
    if batch.get("action_is_pad") is not None:
        frame_mask = batch["action_is_pad"].to(device)  # [B, H] bool
        pad_mask = frame_mask.repeat_interleave(ACTION_DIM, dim=1)
    return model_batch, bins, pad_mask


def _batch_entropy(head, prefix, bins):
    """Mean softmax entropy of the head's logits (a canary metric).

    A fresh head sits near ``log(n_bins)`` (uniform); a collapsing or
    over-confident head drifts far from it. Computed under ``no_grad``
    on logging steps only, reusing the head's ``logits`` path so it adds
    no gradient work.
    """
    with torch.no_grad():
        logits = head.logits(prefix, bins).float()
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs + 1e-9)).sum(-1)
        return ent.mean().item()


def _trainable_params(head, fusion):
    """Head + currently-trainable backbone params for grad clipping."""
    seen = set()
    params = []
    for p in list(head.parameters()) + list(
        fusion.parameters_backbone()
    ):
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return params


def train(head, fusion, tokenizer, loader, cfg, log_fn=print):
    """Train the AR head (Listing 4.13 + the run machinery).

    ``head``, ``fusion``, ``tokenizer`` and ``loader`` are injected (the
    real dataset/backbone are built in ``main``, never here). Runs up to
    ``cfg.total_steps`` optimizer steps, cycling the loader across
    ``cfg.n_epochs``; each optimizer step consumes ``cfg.grad_accum``
    microbatches. The backbone is frozen for epoch 0 and unfrozen at the
    start of epoch 1. Returns a ``history`` dict with ``step``, ``loss``,
    ``lr``, ``entropy`` recorded every ``cfg.log_every`` optimizer steps.
    """
    device = next(head.parameters()).device
    opt = build_optimizer(head, fusion, head_lr=HEAD_LR)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    if device.type == "cuda":
        use_bf16 = cfg.bf16 and torch.cuda.is_bf16_supported()
    else:
        use_bf16 = False
    # GradScaler is a no-op passthrough when disabled (fp16 on CUDA is
    # the only case that needs it), so the same code path runs on CPU.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and not use_bf16)
    )

    history = {"step": [], "loss": [], "lr": [], "entropy": []}
    opt_step = 0
    microbatch = 0
    clip_params = _trainable_params(head, fusion)
    done = False

    for epoch in range(cfg.n_epochs):
        if epoch == 1:
            # Unfreeze at epoch 1: backbone joins the optimization once
            # the head has stopped emitting random-init gradients.
            unfreeze_backbone(opt, fusion, BACKBONE_LR)
            base_lrs[1] = BACKBONE_LR
            clip_params = _trainable_params(head, fusion)

        for batch in loader:
            if opt_step >= cfg.total_steps:
                done = True
                break

            model_batch, bins, pad_mask = _prepare_batch(
                batch, tokenizer, device
            )
            with autocast_context(device, use_bf16):
                prefix = fusion.encode_prefix(model_batch)
                raw_loss = head(prefix, bins, pad_mask=pad_mask)
            scaler.scale(raw_loss / cfg.grad_accum).backward()
            microbatch += 1

            if microbatch % cfg.grad_accum != 0:
                continue

            # One optimizer step per grad_accum microbatches; the LR
            # schedule advances here (per optimizer step, not per
            # microbatch).
            opt_step += 1
            mult = warmup_cosine(
                opt_step, cfg.warmup_steps, cfg.total_steps
            )
            for pg, base in zip(opt.param_groups, base_lrs):
                pg["lr"] = base * mult

            scaler.unscale_(opt)
            clip_grad_norm_(clip_params, GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if opt_step % cfg.log_every == 0:
                entropy = _batch_entropy(head, prefix, bins)
                history["step"].append(opt_step)
                history["loss"].append(raw_loss.item())
                history["lr"].append(opt.param_groups[0]["lr"])
                history["entropy"].append(entropy)
                log_fn(
                    f"step {opt_step} loss {raw_loss.item():.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} "
                    f"entropy {entropy:.4f}"
                )

            if opt_step % cfg.steps_per_checkpoint == 0:
                os.makedirs(cfg.out_dir, exist_ok=True)
                path = os.path.join(
                    cfg.out_dir, f"step_{opt_step}.pt"
                )
                save_checkpoint(path, head, fusion, opt_step)
                log_fn(f"saved checkpoint {path}")

            if opt_step >= cfg.total_steps:
                done = True
                break
        if done:
            break

    return history


# -- checkpointing ----------------------------------------------------
#
# The fusion checkpoint stores only the adapter's *trainable* parts --
# the same modules ``parameters_backbone`` yields (img_proj, state_proj,
# embed_tokens, language_backbone). The frozen SigLIP vision encoder and
# any other frozen weights reload from Hugging Face, so there is no
# reason to duplicate them here. The head's own params (everything not
# under its ``fusion`` submodule) are saved separately.

_FUSION_MODULES = (
    "img_proj",
    "state_proj",
    "embed_tokens",
    "language_backbone",
)


def _fusion_trainable_state(fusion):
    return {
        name: getattr(fusion.backbone, name).state_dict()
        for name in _FUSION_MODULES
    }


def save_checkpoint(path, head, fusion, step):
    """Save head + trainable-fusion state dicts and the step count."""
    head_state = {
        k: v
        for k, v in head.state_dict().items()
        if not k.startswith("fusion.")
    }
    torch.save(
        {
            "step": step,
            "head": head_state,
            "fusion": _fusion_trainable_state(fusion),
        },
        path,
    )


def load_checkpoint(path, head, fusion):
    """Restore head + trainable-fusion state; return the saved step."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["head"], strict=False)
    for name in _FUSION_MODULES:
        getattr(fusion.backbone, name).load_state_dict(
            ckpt["fusion"][name]
        )
    return ckpt["step"]


# -- config + entry point ---------------------------------------------


def load_config(path):
    """Load a YAML config into a ``TrainConfig`` (strict keys).

    Unknown keys raise rather than being silently dropped: a typo'd
    key (say ``microbatch_size:``) would otherwise fall back to the
    default and quietly poison a locked-recipe run.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(data) - fields)
    if unknown:
        raise ValueError(
            f"unknown config key(s) {unknown} in {path}; "
            f"valid keys: {sorted(fields)}"
        )
    return TrainConfig(**data)


def main(argv=None):
    """Standalone entry: ``python -m ch04.train --config <path>``.

    Builds the *real* dataset, ch3 backbone, fusion adapter, tokenizer,
    and AR head, then hands them to ``train``. Imports live inside the
    function so the unit tests (which inject fakes into ``train``) never
    trigger a model download or the lerobot import.
    """
    parser = argparse.ArgumentParser(description="Train the AR head.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    from ch03 import UnifiedEmbeddingBackbone

    from ch04.action_tokenizer import ActionTokenizer
    from ch04.autoregressive_action_head import (
        AutoregressiveActionHead,
    )
    from ch04.chunk_data import make_chunk_dataset, make_chunk_loader
    from ch04.fusion_adapter import FusionAdapter

    backbone = UnifiedEmbeddingBackbone().float()
    fusion = FusionAdapter(backbone)
    head = AutoregressiveActionHead(fusion)
    dataset = make_chunk_dataset(episodes=cfg.episodes)
    tokenizer = ActionTokenizer.from_lerobot_stats(dataset.meta.stats)
    loader = make_chunk_loader(dataset, batch_size=cfg.microbatch)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    head.to(device)
    return train(head, fusion, tokenizer, loader, cfg)


if __name__ == "__main__":
    main()
