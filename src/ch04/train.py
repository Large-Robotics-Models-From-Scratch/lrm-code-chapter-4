"""Training utilities for manuscript listings 4.6 and 4.7."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path

import torch

from ch04.backbone_adapter import encode_prefix, gather_state_hidden
from ch04.data import action_targets, prepare_batch
from ch04.losses import categorical_entropy, masked_token_cross_entropy


def freeze_backbone(backbone) -> None:
    """Freeze every Chapter 3 parameter while training the action head."""
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()


def head_parameters(head, backbone) -> list[torch.nn.Parameter]:
    """Return trainable parameters owned outside the frozen backbone."""
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    return [
        parameter
        for parameter in head.parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]


def head_only_state_dict(head) -> dict[str, torch.Tensor]:
    """Return action-head state without its registered frozen backbone."""
    return {
        name: value
        for name, value in head.state_dict().items()
        if not name.startswith("backbone.")
    }


def load_head_only_state_dict(head, state_dict) -> None:
    """Load a compact head state while retaining the attached backbone."""
    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        name for name in missing if not name.startswith("backbone.")
    ]
    if invalid_missing or unexpected:
        raise ValueError(
            "invalid compact head state: "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )


def make_optimizer(
    head,
    backbone,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """Build AdamW over head-only parameters."""
    parameters = head_parameters(head, backbone)
    if not parameters:
        raise ValueError("the action head has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )


def warmup_cosine_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError("require 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = min(
        (step - warmup_steps) / (total_steps - warmup_steps), 1.0
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 500,
    total_steps: int = 20_000,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_multiplier(
            step, warmup_steps, total_steps
        ),
    )


def action_head_logits(head, backbone, model_inputs, target_bins):
    """Run any of the manuscript's three action-head variants."""
    if hasattr(head, "teacher_forced_logits"):
        return head.teacher_forced_logits(*model_inputs, target_bins)
    if hasattr(head, "backbone"):
        return head(*model_inputs)
    images, sequence_ids, state = model_inputs
    hidden = encode_prefix(backbone, images, sequence_ids, state)
    state_hidden = gather_state_hidden(backbone, hidden, sequence_ids)
    return head(state_hidden)


def train_action_head(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    total_steps: int = 20_000,
    warmup_steps: int = 500,
    learning_rate: float = 1e-4,
    label_smoothing: float = 0.05,
    grad_clip: float = 1.0,
    log_every: int = 100,
    checkpoint_every: int = 1_000,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> list[dict[str, float]]:
    """Train a factorized, autoregressive, or parallel action head."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    head.to(device)
    head.train()
    freeze_backbone(backbone)
    optimizer = make_optimizer(head, backbone, learning_rate)
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
    parameters = head_parameters(head, backbone)
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        expected_config = {
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "label_smoothing": label_smoothing,
            "grad_clip": grad_clip,
        }
        if checkpoint.get("config") != expected_config:
            raise ValueError("resume configuration differs from checkpoint")
        saved_tokenizer = checkpoint["tokenizer"]
        current_lo = torch.from_numpy(tokenizer.lo)
        current_hi = torch.from_numpy(tokenizer.hi)
        if (
            saved_tokenizer["n_bins"] != tokenizer.n_bins
            or not torch.equal(saved_tokenizer["lo"].cpu(), current_lo)
            or not torch.equal(saved_tokenizer["hi"].cpu(), current_hi)
        ):
            raise ValueError("resume tokenizer differs from checkpoint")
        load_head_only_state_dict(head, checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"]) + 1
    while step < total_steps:
        for batch in loader:
            model_inputs = prepare_batch(batch, stats, backbone, device)
            bins, pad = action_targets(batch, stats, tokenizer, device)
            use_amp = torch.device(device).type == "cuda"
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with amp:
                logits = action_head_logits(
                    head, backbone, model_inputs, bins
                )
            loss = masked_token_cross_entropy(
                logits,
                bins,
                pad,
                label_smoothing=label_smoothing,
            )
            step_learning_rate = optimizer.param_groups[0]["lr"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                record = {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "entropy": float(categorical_entropy(logits).detach()),
                    "lr": step_learning_rate,
                }
                history.append(record)
                print(
                    f"[{step:6d}] loss={record['loss']:.3f} "
                    f"ent={record['entropy']:.2f}"
                )
            if (
                checkpoint_path is not None
                and checkpoint_every > 0
                and step % checkpoint_every == checkpoint_every - 1
            ):
                torch.save(
                    {
                        "step": step,
                        "head": head_only_state_dict(head),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "tokenizer": {
                            "lo": torch.from_numpy(tokenizer.lo.copy()),
                            "hi": torch.from_numpy(tokenizer.hi.copy()),
                            "n_bins": tokenizer.n_bins,
                        },
                        "config": {
                            "total_steps": total_steps,
                            "warmup_steps": warmup_steps,
                            "learning_rate": learning_rate,
                            "label_smoothing": label_smoothing,
                            "grad_clip": grad_clip,
                        },
                    },
                    checkpoint_path / f"step{step:06d}.pt",
                )
            step += 1
            if step >= total_steps:
                break
        else:
            raise ValueError("loader produced no batches")
    return history


def train_parallel_head(*args, **kwargs):
    """Backward-compatible name for the manuscript's main training path."""
    return train_action_head(*args, **kwargs)
