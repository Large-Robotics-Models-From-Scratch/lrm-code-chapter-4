"""Training utilities aligned with manuscript listings 4.7 and 4.8."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path

import torch

from ch04.backbone_adapter import encode_prefix, gather_state_hidden
from ch04.data import action_targets, prepare_batch
from ch04.losses import masked_token_cross_entropy


def head_parameters(head, backbone) -> list[torch.nn.Parameter]:
    """Return trainable parameters owned by the new action head."""
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    return [
        parameter
        for parameter in head.parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]


def backbone_parameters(backbone) -> list[torch.nn.Parameter]:
    """Return Chapter 3 parameters that remain trainable in Chapter 4."""
    return [
        parameter
        for parameter in backbone.parameters()
        if parameter.requires_grad
    ]


def upcast_trainable_parameters(
    module,
    dtype: torch.dtype = torch.float32,
) -> list[str]:
    """Promote trainable low-precision parameters to full precision.

    SmolLM2's checkpoint declares ``bfloat16`` and transformers honours
    it, so the Chapter 3 trunk arrives in bfloat16. At the manuscript's
    ``backbone_learning_rate`` of 1e-5 an AdamW update is roughly 1e-5
    relative, which rounds to zero in bfloat16's eight-bit mantissa: the
    loss falls, the gradients look healthy, and the pretrained trunk
    barely moves. Keeping float32 master weights fixes that. Under CUDA
    autocast the matmuls still run in bfloat16, so the cost is memory,
    not throughput.

    Returns the names of the parameters that were promoted.
    """
    promoted = []
    for name, parameter in module.named_parameters():
        if (
            parameter.requires_grad
            and parameter.is_floating_point()
            and parameter.dtype != dtype
        ):
            parameter.data = parameter.data.to(dtype)
            promoted.append(name)
    return promoted


def policy_state_dict(head, backbone) -> dict[str, object]:
    """Serialize the complete policy without duplicating the backbone."""
    head_state = {
        name: value
        for name, value in head.state_dict().items()
        if not name.startswith("backbone.")
    }
    return {
        "head": head_state,
        "backbone": backbone.state_dict(),
    }


def load_policy_state_dict(head, backbone, state_dict) -> None:
    """Restore a policy produced by :func:`policy_state_dict`."""
    missing, unexpected = head.load_state_dict(
        state_dict["head"], strict=False
    )
    invalid_missing = [
        name for name in missing if not name.startswith("backbone.")
    ]
    if invalid_missing or unexpected:
        raise ValueError(
            "invalid policy head state: "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )
    backbone.load_state_dict(state_dict["backbone"])


def make_optimizer(
    head,
    backbone,
    learning_rate: float = 1e-4,
    backbone_learning_rate: float = 1e-5,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """Use separate learning rates for the new head and pretrained trunk."""
    head_params = head_parameters(head, backbone)
    backbone_params = backbone_parameters(backbone)
    if not head_params:
        raise ValueError("the action head has no trainable parameters")
    if not backbone_params:
        raise ValueError(
            "the Chapter 3 backbone has no trainable parameters"
        )
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": learning_rate},
            {"params": backbone_params, "lr": backbone_learning_rate},
        ],
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
        return (step + 1) / max(warmup_steps, 1)
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


def action_head_logits(
    head,
    backbone,
    model_inputs,
    target_bins: torch.Tensor | None = None,
):
    """Return training logits from any of the three action heads.

    The factorized and parallel heads need only the observation. The
    autoregressive head additionally consumes the expert grid for causal
    teacher forcing. Keeping that variation here lets training, validation,
    and visualization share the rest of their pipeline.
    """
    if hasattr(head, "teacher_forced_logits"):
        if target_bins is None:
            raise ValueError(
                "target_bins are required for autoregressive logits"
            )
        return head.teacher_forced_logits(*model_inputs, target_bins)
    if hasattr(head, "backbone"):
        return head(*model_inputs)
    hidden = encode_prefix(backbone, *model_inputs)
    return head(gather_state_hidden(hidden))


@torch.no_grad()
def held_out_loss(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    label_smoothing: float = 0.05,
    max_batches: int | None = None,
) -> float:
    """Dataset-level token CE on an episode-disjoint validation loader.

    ``max_batches`` bounds the evaluation. The SO-101 split holds roughly
    1,200 held-out frames, so an unbounded pass costs one backbone forward
    per frame every time a checkpoint is written. Bounding it keeps the
    validation signal cheap enough to run often; leave it ``None`` for the
    exact dataset-level number.
    """
    modules = {
        id(module): module
        for root in (head, backbone)
        for module in root.modules()
    }
    modes = [(module, module.training) for module in modules.values()]
    head.eval()
    backbone.eval()
    loss_sum = 0.0
    valid_count = 0
    try:
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            model_inputs = prepare_batch(
                batch, stats, device, backbone
            )
            bins, pad = action_targets(batch, stats, tokenizer, device)
            logits = action_head_logits(
                head, backbone, model_inputs, bins
            )
            keep = ~pad
            loss = masked_token_cross_entropy(
                logits,
                bins,
                pad,
                label_smoothing=label_smoothing,
            )
            count = int(keep.sum())
            loss_sum += float(loss) * count
            valid_count += count
    finally:
        for module, training in modes:
            module.training = training
    if valid_count == 0:
        raise ValueError(
            "validation loader produced no valid action tokens"
        )
    return loss_sum / valid_count


def _checkpoint_payload(
    *,
    step: int,
    head,
    backbone,
    optimizer,
    scheduler,
    validation_loss: float | None,
    stats,
    tokenizer,
    config,
) -> dict[str, object]:
    return {
        "step": step,
        "model": policy_state_dict(head, backbone),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "validation_loss": validation_loss,
        "normalization": stats,
        "tokenizer": {
            "lo": torch.from_numpy(tokenizer.lo.copy()),
            "hi": torch.from_numpy(tokenizer.hi.copy()),
            "n_bins": tokenizer.n_bins,
        },
        "config": config,
    }


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
    backbone_learning_rate: float = 1e-5,
    label_smoothing: float = 0.05,
    grad_clip: float = 1.0,
    log_every: int = 100,
    checkpoint_every: int = 1_000,
    checkpoint_dir: str | Path | None = None,
    validation_loader: Iterable[Mapping[str, object]] | None = None,
    resume_from: str | Path | None = None,
    upcast_backbone: bool = True,
    validation_batches: int | None = 32,
) -> list[dict[str, float]]:
    """Train a factorized, autoregressive, or parallel action head."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    backbone.to(device)
    head.to(device)
    if upcast_backbone:
        promoted = upcast_trainable_parameters(backbone)
        promoted += upcast_trainable_parameters(head)
        if promoted:
            print(
                f"promoted {len(promoted)} trainable parameters to "
                "float32 so low-precision weights receive their updates"
            )
    head.train()
    backbone.train()
    optimizer = make_optimizer(
        head,
        backbone,
        learning_rate,
        backbone_learning_rate,
    )
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    config = {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "learning_rate": learning_rate,
        "backbone_learning_rate": backbone_learning_rate,
        "label_smoothing": label_smoothing,
        "grad_clip": grad_clip,
        "horizon": getattr(head, "horizon", None),
        "action_dim": getattr(head, "action_dim", None),
    }
    history: list[dict[str, float]] = []
    step = 0
    best_validation = float("inf")
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        if checkpoint.get("config") != config:
            raise ValueError("resume configuration differs from checkpoint")
        saved_tokenizer = checkpoint["tokenizer"]
        if (
            saved_tokenizer["n_bins"] != tokenizer.n_bins
            or not torch.equal(
                torch.as_tensor(saved_tokenizer["lo"]),
                torch.from_numpy(tokenizer.lo),
            )
            or not torch.equal(
                torch.as_tensor(saved_tokenizer["hi"]),
                torch.from_numpy(tokenizer.hi),
            )
        ):
            raise ValueError("resume tokenizer differs from checkpoint")
        load_policy_state_dict(head, backbone, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
        saved_validation = checkpoint.get("validation_loss")
        if saved_validation is not None:
            best_validation = float(saved_validation)

    while step < total_steps:
        saw_batch = False
        for batch in loader:
            saw_batch = True
            optimizer.zero_grad(set_to_none=True)
            model_inputs = prepare_batch(
                batch, stats, device, backbone
            )
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
            step_head_lr = optimizer.param_groups[0]["lr"]
            step_backbone_lr = optimizer.param_groups[1]["lr"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            scheduler.step()

            if step % log_every == 0:
                log_probs = logits.float().log_softmax(-1)
                token_entropy = -(log_probs.exp() * log_probs).sum(-1)
                record = {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "entropy": float(token_entropy[~pad].mean().detach()),
                    "head_lr": step_head_lr,
                    "backbone_lr": step_backbone_lr,
                }
                history.append(record)
                print(
                    f"[{step:6d}] loss={record['loss']:.3f} "
                    f"ent={record['entropy']:.2f}"
                )

            completed_step = step + 1
            should_checkpoint = (
                checkpoint_path is not None
                and checkpoint_every > 0
                and completed_step % checkpoint_every == 0
            )
            if should_checkpoint:
                validation = (
                    held_out_loss(
                        head,
                        backbone,
                        validation_loader,
                        stats,
                        tokenizer,
                        device,
                        label_smoothing,
                        validation_batches,
                    )
                    if validation_loader is not None
                    else None
                )
                payload = _checkpoint_payload(
                    step=completed_step,
                    head=head,
                    backbone=backbone,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    validation_loss=validation,
                    stats=stats,
                    tokenizer=tokenizer,
                    config=config,
                )
                torch.save(payload, checkpoint_path / "latest.pt")
                if validation is not None and validation < best_validation:
                    best_validation = validation
                    torch.save(payload, checkpoint_path / "best.pt")
                if completed_step in {5_000, total_steps}:
                    torch.save(
                        payload,
                        checkpoint_path / f"step{completed_step:06d}.pt",
                    )
            step = completed_step
            if step >= total_steps:
                break
        if not saw_batch:
            raise ValueError("loader produced no batches")
    return history
