"""Training utilities aligned with manuscript listings 4.7 and 4.8."""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path

import torch

from ch04.backbone_adapter import encode_prefix, gather_state_hidden
from ch04.data import (
    action_targets,
    denormalize_from_stats,
    normalize_from_stats,
    prepare_batch,
)
from ch04.losses import masked_token_cross_entropy


def _make_summary_writer(log_dir: str | Path, purge_step: int | None):
    """Construct TensorBoard lazily so head-only use stays lightweight."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError(
            "TensorBoard logging requires `pip install tensorboard`"
        ) from error
    return SummaryWriter(log_dir=str(log_dir), purge_step=purge_step)


def _checkpoint_array_equal(saved, current) -> bool:
    """Compare checkpoint metadata without inheriting its load device."""
    saved_tensor = torch.as_tensor(saved).detach().cpu()
    current_tensor = torch.as_tensor(current).detach().cpu()
    return torch.equal(saved_tensor, current_tensor)


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


def action_metrics(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    pad: torch.Tensor,
    raw_actions,
    stats,
    tokenizer,
    predicted_bins: torch.Tensor | None = None,
) -> dict[str, object]:
    """Token accuracy and decoded MAE, overall and per control.

    ``mae_in_std`` compares decoded bin centres with the demonstrated
    actions after Chapter 2 z-score normalization. A value of 1.0 means
    one training-set standard deviation. ``mae_raw`` inverts that
    normalization and is reported in the dataset's native joint units.

    Pass ``predicted_bins`` to score a grid the head actually generated.
    Without it the bins are taken from ``logits``, which for the
    autoregressive head are teacher-forced: every cell would be scored with
    the true earlier bins supplied, collapsing its decoded error toward the
    quantization floor instead of measuring open-loop error.
    """
    if (
        logits.shape[:-1] != target_bins.shape
        or pad.shape != target_bins.shape
    ):
        raise ValueError("logits, targets, and padding shapes do not align")
    keep = ~pad
    if not bool(keep.any()):
        raise ValueError("the batch contains no valid action tokens")
    if predicted_bins is None:
        predictions = logits.argmax(dim=-1)
    else:
        if predicted_bins.shape != target_bins.shape:
            raise ValueError(
                "predicted_bins must match the target grid shape"
            )
        predictions = predicted_bins
    correct = (predictions == target_bins) & keep
    counts = keep.sum(dim=(0, 1))
    accuracy_by_control = correct.sum(dim=(0, 1)).float() / counts

    decoded = tokenizer.decode(predictions.detach().cpu().numpy())
    decoded_std = torch.from_numpy(decoded).to(
        target_bins.device, dtype=logits.dtype
    )
    targets_raw = torch.as_tensor(
        raw_actions, device=target_bins.device, dtype=decoded_std.dtype
    )
    targets_std = normalize_from_stats(targets_raw, stats, "action")
    decoded_raw = denormalize_from_stats(decoded_std, stats, "action")
    mae_std_by_control = (
        ((decoded_std - targets_std).abs() * keep).sum(dim=(0, 1))
        / counts
    )
    mae_raw_by_control = (
        ((decoded_raw - targets_raw).abs() * keep).sum(dim=(0, 1))
        / counts
    )
    return {
        "accuracy": float(correct.sum() / keep.sum()),
        "accuracy_by_control": accuracy_by_control.detach().cpu().tolist(),
        "mae_in_std": float(
            ((decoded_std - targets_std).abs() * keep).sum() / keep.sum()
        ),
        "mae_in_std_by_control": (
            mae_std_by_control.detach().cpu().tolist()
        ),
        "mae_raw_by_control": mae_raw_by_control.detach().cpu().tolist(),
    }


@torch.no_grad()
def held_out_metrics(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    label_smoothing: float = 0.05,
    max_batches: int | None = None,
    rollout_batches: int | None = None,
) -> dict[str, object]:
    """Loss, token accuracy, and decoded MAE on held-out episodes.

    Cross-entropy and the explicitly named ``teacher_forced_*`` metrics use
    expert prefixes, matching the training objective. The short
    ``accuracy`` and ``mae_in_std`` aliases instead score the grid each head
    actually generates, so the autoregressive head is rolled out through
    its own earlier choices rather than handed the expert prefix. Keeping
    both scopes prevents a one-minibatch training number from being compared
    with a small deployment-rollout validation slice.

    ``max_batches`` bounds the evaluation. The SO-101 split holds roughly
    1,200 held-out frames, so an unbounded pass costs one backbone forward
    per frame every time a checkpoint is written. Bounding it keeps the
    validation signal cheap enough to run often; leave it ``None`` for the
    exact dataset-level number.

    ``rollout_batches`` bounds the decoded metrics separately, because an
    autoregressive rollout costs ``H * D`` serial steps per batch and is
    far more expensive than the teacher-forced pass. The returned
    ``rollout_batches_used`` records how many batches it covered, so a
    bounded decoded metric is never mistaken for a full-split one. Pass
    ``0`` to skip decoding entirely and pay only for the loss, which is
    what :func:`held_out_loss` needs.
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
    teacher_correct = None
    teacher_mae_std = None
    teacher_mae_raw = None
    teacher_counts = None
    rollout_correct = None
    rollout_mae_std = None
    rollout_mae_raw = None
    rollout_counts = None
    rollout_used = 0
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

            teacher_metrics = action_metrics(
                logits,
                bins,
                pad,
                batch["action"],
                stats,
                tokenizer,
            )
            control_counts = keep.sum(dim=(0, 1)).cpu().double()
            batch_teacher_correct = (
                torch.tensor(teacher_metrics["accuracy_by_control"])
                * control_counts
            )
            batch_teacher_mae_std = (
                torch.tensor(teacher_metrics["mae_in_std_by_control"])
                * control_counts
            )
            batch_teacher_mae_raw = (
                torch.tensor(teacher_metrics["mae_raw_by_control"])
                * control_counts
            )
            if teacher_correct is None:
                teacher_correct = batch_teacher_correct
                teacher_mae_std = batch_teacher_mae_std
                teacher_mae_raw = batch_teacher_mae_raw
                teacher_counts = control_counts.clone()
            else:
                teacher_correct += batch_teacher_correct
                teacher_mae_std += batch_teacher_mae_std
                teacher_mae_raw += batch_teacher_mae_raw
                teacher_counts += control_counts

            if (
                rollout_batches is not None
                and rollout_used >= rollout_batches
            ):
                continue
            # Score what the head would actually emit at deployment: the
            # AR head generates through its own choices, the one-pass
            # heads simply argmax their logits as before.
            from ch04.decoding import action_head_bins

            predicted = action_head_bins(
                head, backbone, model_inputs, strategy="argmax"
            )
            rollout_used += 1
            metrics = action_metrics(
                logits,
                bins,
                pad,
                batch["action"],
                stats,
                tokenizer,
                predicted_bins=predicted,
            )
            batch_correct = (
                torch.tensor(metrics["accuracy_by_control"])
                * control_counts
            )
            batch_mae_std = (
                torch.tensor(metrics["mae_in_std_by_control"])
                * control_counts
            )
            batch_mae_raw = (
                torch.tensor(metrics["mae_raw_by_control"])
                * control_counts
            )
            if rollout_correct is None:
                rollout_correct = batch_correct
                rollout_mae_std = batch_mae_std
                rollout_mae_raw = batch_mae_raw
                rollout_counts = control_counts.clone()
            else:
                rollout_correct += batch_correct
                rollout_mae_std += batch_mae_std
                rollout_mae_raw += batch_mae_raw
                rollout_counts += control_counts
    finally:
        for module, training in modes:
            module.training = training
    if valid_count == 0:
        raise ValueError(
            "validation loader produced no valid action tokens"
        )
    teacher_accuracy_by_control = (
        teacher_correct / teacher_counts
    ).tolist()
    teacher_mae_std_by_control = (
        teacher_mae_std / teacher_counts
    ).tolist()
    teacher_mae_raw_by_control = (
        teacher_mae_raw / teacher_counts
    ).tolist()
    result = {
        "loss": loss_sum / valid_count,
        "teacher_forced_accuracy": float(
            teacher_correct.sum() / teacher_counts.sum()
        ),
        "teacher_forced_accuracy_by_control": (
            teacher_accuracy_by_control
        ),
        "teacher_forced_mae_in_std": float(
            teacher_mae_std.sum() / teacher_counts.sum()
        ),
        "teacher_forced_mae_in_std_by_control": (
            teacher_mae_std_by_control
        ),
        "teacher_forced_mae_raw_by_control": (
            teacher_mae_raw_by_control
        ),
        "teacher_forced_token_count": int(teacher_counts.sum()),
        "rollout_batches_used": rollout_used,
    }
    if rollout_correct is None:
        if rollout_batches != 0:
            raise ValueError(
                "rollout_batches excluded every batch, so no decoded "
                "metric could be computed"
            )
        return result
    accuracy_by_control = (rollout_correct / rollout_counts).tolist()
    mae_std_by_control = (rollout_mae_std / rollout_counts).tolist()
    mae_raw_by_control = (rollout_mae_raw / rollout_counts).tolist()
    # The short names remain deployment aliases for CLI compatibility.
    # Callers that compare train and validation curves should use the
    # explicit teacher_forced_* fields above.
    result.update({
        "accuracy": float(
            rollout_correct.sum() / rollout_counts.sum()
        ),
        "accuracy_by_control": accuracy_by_control,
        "mae_in_std": float(
            rollout_mae_std.sum() / rollout_counts.sum()
        ),
        "mae_in_std_by_control": mae_std_by_control,
        "mae_raw_by_control": mae_raw_by_control,
        "rollout_token_count": int(rollout_counts.sum()),
    })
    return result


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
    """Backward-compatible scalar view of :func:`held_out_metrics`."""
    return float(
        held_out_metrics(
            head,
            backbone,
            loader,
            stats,
            tokenizer,
            device,
            label_smoothing,
            max_batches,
            rollout_batches=0,
        )["loss"]
    )


def _checkpoint_payload(
    *,
    step: int,
    head,
    backbone,
    optimizer,
    scheduler,
    validation_loss: float | None,
    validation_metrics: Mapping[str, object] | None,
    best_validation_loss: float,
    history: list[dict[str, float]],
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
        "validation_metrics": validation_metrics,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "normalization": stats,
        "tokenizer": {
            "lo": torch.from_numpy(tokenizer.lo.copy()),
            "hi": torch.from_numpy(tokenizer.hi.copy()),
            "n_bins": tokenizer.n_bins,
        },
        "config": config,
    }


def _mirror_checkpoint(source: Path, mirror_dir: Path) -> bool:
    """Atomically mirror one local checkpoint to durable storage.

    Colab's Google Drive mount is much slower and less reliable than its
    local disk, so training writes locally first. A failed mirror leaves the
    local checkpoint intact and is retried naturally at the next interval.
    """
    destination = mirror_dir / source.name
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    except OSError as error:
        print(
            f"warning: could not mirror {source.name} to "
            f"{mirror_dir}: {error}"
        )
        return False
    print(f"mirrored checkpoint: {destination}")
    return True


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
    checkpoint_mirror_dir: str | Path | None = None,
    validation_loader: Iterable[Mapping[str, object]] | None = None,
    resume_from: str | Path | None = None,
    upcast_backbone: bool = True,
    snapshot_steps: tuple[int, ...] = (),
    validate_every: int | None = None,
    validation_batches: int | None = 32,
    validation_rollout_batches: int | None = 8,
    tensorboard_log_dir: str | Path | None = None,
) -> list[dict[str, float]]:
    """Train a factorized, autoregressive, or parallel action head.

    ``latest.pt`` is rewritten every ``checkpoint_every`` steps and
    ``best.pt`` whenever the held-out loss improves. If
    ``checkpoint_mirror_dir`` is set, each file is written to fast local
    storage first and then copied atomically to that durable location.
    Mirror failures warn rather than discarding the local checkpoint or
    stopping training. ``snapshot_steps`` additionally keeps a permanent
    copy at the given steps; it is empty by default because a policy
    checkpoint carries the whole float32 backbone and runs to roughly a
    gigabyte.

    Every history record carries ``validation_loss``: the held-out token
    cross-entropy measured just after that record's optimizer step, or NaN
    on steps where no validation ran. Matplotlib skips NaN, so plotting the
    column directly gives the measured points and nothing else. Validation
    runs every ``validate_every`` steps, defaulting to ``checkpoint_every``,
    and always once at the final step.

    When ``tensorboard_log_dir`` is set, loss, accuracy, decoded MAE,
    predictive entropy, per-control metrics, and both learning rates are
    streamed as scalars. The writer uses the restored step as
    ``purge_step`` when resuming.
    """
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
    mirror_path = None
    if checkpoint_mirror_dir is not None:
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_mirror_dir requires checkpoint_dir"
            )
        mirror_path = Path(checkpoint_mirror_dir)

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
    if validate_every is None:
        validate_every = checkpoint_every
    if validate_every < 0:
        raise ValueError("validate_every must be non-negative")
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        if checkpoint.get("config") != config:
            raise ValueError("resume configuration differs from checkpoint")
        saved_tokenizer = checkpoint["tokenizer"]
        if (
            saved_tokenizer["n_bins"] != tokenizer.n_bins
            or not _checkpoint_array_equal(
                saved_tokenizer["lo"], tokenizer.lo
            )
            or not _checkpoint_array_equal(
                saved_tokenizer["hi"], tokenizer.hi
            )
        ):
            raise ValueError("resume tokenizer differs from checkpoint")
        load_policy_state_dict(head, backbone, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
        history = [dict(record) for record in checkpoint.get("history", [])]
        saved_best = checkpoint.get(
            "best_validation_loss", checkpoint.get("validation_loss")
        )
        if saved_best is not None:
            best_validation = float(saved_best)

    writer = None
    if tensorboard_log_dir is not None:
        writer = _make_summary_writer(
            tensorboard_log_dir, step if step > 0 else None
        )

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

            completed_step = step + 1
            should_checkpoint = (
                checkpoint_path is not None
                and checkpoint_every > 0
                and completed_step % checkpoint_every == 0
            )
            should_validate = validation_loader is not None and (
                should_checkpoint
                or completed_step == total_steps
                or (
                    validate_every > 0
                    and completed_step % validate_every == 0
                )
            )
            log_now = (
                completed_step == 1
                or completed_step % log_every == 0
                or completed_step == total_steps
                or should_validate
            )
            if log_now:
                log_probs = logits.float().log_softmax(-1)
                token_entropy = -(log_probs.exp() * log_probs).sum(-1)
                metrics = action_metrics(
                    logits,
                    bins,
                    pad,
                    batch["action"],
                    stats,
                    tokenizer,
                )
                record = {
                    "step": float(completed_step),
                    "loss": float(loss.detach()),
                    "entropy": float(token_entropy[~pad].mean().detach()),
                    "head_lr": step_head_lr,
                    "backbone_lr": step_backbone_lr,
                    "validation_loss": float("nan"),
                    "accuracy": metrics["accuracy"],
                    "mae_in_std": metrics["mae_in_std"],
                    "validation_accuracy": float("nan"),
                    "validation_mae_in_std": float("nan"),
                    "validation_rollout_accuracy": float("nan"),
                    "validation_rollout_mae_in_std": float("nan"),
                }
                for index in range(len(metrics["accuracy_by_control"])):
                    record[f"accuracy_dim_{index}"] = metrics[
                        "accuracy_by_control"
                    ][index]
                    record[f"mae_in_std_dim_{index}"] = metrics[
                        "mae_in_std_by_control"
                    ][index]
                    record[f"validation_accuracy_dim_{index}"] = float(
                        "nan"
                    )
                    record[f"validation_mae_in_std_dim_{index}"] = float(
                        "nan"
                    )
                history.append(record)

            validation = None
            validation_metrics = None
            if should_validate:
                validation_metrics = held_out_metrics(
                    head,
                    backbone,
                    validation_loader,
                    stats,
                    tokenizer,
                    device,
                    label_smoothing,
                    validation_batches,
                    rollout_batches=validation_rollout_batches,
                )
                validation = float(validation_metrics["loss"])
                if history:
                    history[-1]["validation_loss"] = validation
                    history[-1]["validation_accuracy"] = (
                        validation_metrics["teacher_forced_accuracy"]
                    )
                    history[-1]["validation_mae_in_std"] = (
                        validation_metrics["teacher_forced_mae_in_std"]
                    )
                    history[-1]["validation_rollout_accuracy"] = (
                        validation_metrics.get("accuracy", float("nan"))
                    )
                    history[-1]["validation_rollout_mae_in_std"] = (
                        validation_metrics.get(
                            "mae_in_std", float("nan")
                        )
                    )
                    for index in range(
                        len(validation_metrics[
                            "teacher_forced_accuracy_by_control"
                        ])
                    ):
                        history[-1][
                            f"validation_accuracy_dim_{index}"
                        ] = validation_metrics[
                            "teacher_forced_accuracy_by_control"
                        ][index]
                        history[-1][
                            f"validation_mae_in_std_dim_{index}"
                        ] = validation_metrics[
                            "teacher_forced_mae_in_std_by_control"
                        ][index]
                head.train()
                backbone.train()
            if writer is not None and log_now:
                writer.add_scalar(
                    "loss/train", history[-1]["loss"], completed_step
                )
                writer.add_scalar(
                    "entropy/train",
                    history[-1]["entropy"],
                    completed_step,
                )
                writer.add_scalar(
                    "learning_rate/head", step_head_lr, completed_step
                )
                writer.add_scalar(
                    "learning_rate/backbone",
                    step_backbone_lr,
                    completed_step,
                )
                writer.add_scalar(
                    "accuracy/train",
                    history[-1]["accuracy"],
                    completed_step,
                )
                writer.add_scalar(
                    "control_mae/train",
                    history[-1]["mae_in_std"],
                    completed_step,
                )
                for index in range(getattr(head, "action_dim", 0)):
                    writer.add_scalar(
                        f"accuracy_by_control/train_{index}",
                        history[-1][f"accuracy_dim_{index}"],
                        completed_step,
                    )
                    writer.add_scalar(
                        f"mae_in_std_by_control/train_{index}",
                        history[-1][f"mae_in_std_dim_{index}"],
                        completed_step,
                    )
            if writer is not None and validation is not None:
                writer.add_scalar(
                    "loss/held_out", validation, completed_step
                )
                writer.add_scalar(
                    "accuracy/held_out",
                    validation_metrics["teacher_forced_accuracy"],
                    completed_step,
                )
                writer.add_scalar(
                    "accuracy/held_out_teacher_forced",
                    validation_metrics["teacher_forced_accuracy"],
                    completed_step,
                )
                if "accuracy" in validation_metrics:
                    writer.add_scalar(
                        "accuracy/held_out_rollout",
                        validation_metrics["accuracy"],
                        completed_step,
                    )
                writer.add_scalar(
                    "control_mae/held_out",
                    validation_metrics["teacher_forced_mae_in_std"],
                    completed_step,
                )
                for index in range(getattr(head, "action_dim", 0)):
                    writer.add_scalar(
                        f"accuracy_by_control/held_out_{index}",
                        validation_metrics[
                            "teacher_forced_accuracy_by_control"
                        ][index],
                        completed_step,
                    )
                    writer.add_scalar(
                        f"mae_in_std_by_control/held_out_{index}",
                        validation_metrics[
                            "teacher_forced_mae_in_std_by_control"
                        ][index],
                        completed_step,
                    )
            if log_now:
                message = (
                    f"[{completed_step:6d}] loss={history[-1]['loss']:.3f} "
                    f"train_batch_tf_acc={history[-1]['accuracy']:.1%} "
                    f"train_batch_tf_control_mae="
                    f"{history[-1]['mae_in_std']:.3f}"
                )
                if validation is not None:
                    message += (
                        f" val_tf_ce={validation:.3f} "
                        f"val_tf_acc="
                        f"{validation_metrics['teacher_forced_accuracy']:.1%}"
                    )
                    if "accuracy" in validation_metrics:
                        message += (
                            f" val_rollout_acc="
                            f"{validation_metrics['accuracy']:.1%} "
                            "(n="
                            f"{validation_metrics['rollout_token_count']} "
                            "tokens)"
                        )
                print(message)
            if should_checkpoint:
                best_updated = (
                    validation is not None
                    and validation < best_validation
                )
                if best_updated:
                    best_validation = validation
                payload = _checkpoint_payload(
                    step=completed_step,
                    head=head,
                    backbone=backbone,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    validation_loss=validation,
                    validation_metrics=validation_metrics,
                    best_validation_loss=best_validation,
                    history=history,
                    stats=stats,
                    tokenizer=tokenizer,
                    config=config,
                )
                latest_path = checkpoint_path / "latest.pt"
                torch.save(payload, latest_path)
                written_paths = [latest_path]
                if best_updated:
                    best_path = checkpoint_path / "best.pt"
                    torch.save(payload, best_path)
                    written_paths.append(best_path)
                if completed_step in snapshot_steps:
                    snapshot_path = (
                        checkpoint_path / f"step{completed_step:06d}.pt"
                    )
                    torch.save(
                        payload,
                        snapshot_path,
                    )
                    written_paths.append(snapshot_path)
                if mirror_path is not None:
                    for written_path in written_paths:
                        _mirror_checkpoint(written_path, mirror_path)
                if writer is not None:
                    writer.flush()
            step = completed_step
            if step >= total_steps:
                break
        if not saw_batch:
            raise ValueError("loader produced no batches")
    if writer is not None:
        writer.flush()
        writer.close()
    return history
