"""Temperature, top-p, and action-chunk decoding."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch

from ch04.constants import ACTION_DIM, ACTION_HORIZON
from ch04.data import denormalize_from_stats, prepare_batch
from ch04.train import action_head_logits


@contextmanager
def evaluation_mode(module):
    """Temporarily evaluate a module and restore every child mode."""
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, training in modes:
            child.training = training


def nucleus_probabilities(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Return normalized probabilities after inclusive nucleus filtering."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    probs = (logits.float() / temperature).softmax(dim=-1)
    if top_p == 1.0:
        return probs
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    keep = sorted_probs.cumsum(dim=-1) - sorted_probs <= top_p
    filtered = torch.where(keep, sorted_probs, 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    restored = torch.zeros_like(filtered)
    return restored.scatter(-1, sorted_ids, filtered)


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    greedy: bool = False,
) -> torch.Tensor:
    """Choose one bin from the final axis of ``logits``."""
    if greedy:
        return logits.argmax(dim=-1)
    probs = nucleus_probabilities(logits, temperature, top_p)
    picks = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1)
    return picks.reshape(probs.shape[:-1])


def select_bins(
    logits: torch.Tensor,
    strategy: str = "argmax",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Select one bin per cell using manuscript listing 4.10."""
    if strategy == "argmax":
        return logits.argmax(dim=-1)
    if strategy == "sample":
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        probabilities = (logits.float() / temperature).softmax(-1)
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        return torch.multinomial(flat, 1).view(logits.shape[:-1])
    raise ValueError(f"unknown strategy: {strategy}")


@torch.no_grad()
def action_head_bins(
    head,
    backbone,
    model_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    strategy: str = "argmax",
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Select a discrete grid through the head's real inference path."""
    with evaluation_mode(backbone), evaluation_mode(head):
        if hasattr(head, "generate"):
            if strategy == "argmax":
                return head.generate(*model_inputs, temperature=0.0)
            if strategy == "sample":
                return head.generate(
                    *model_inputs,
                    temperature=temperature,
                    top_p=top_p,
                )
            raise ValueError(f"unknown strategy: {strategy}")
        logits = action_head_logits(head, backbone, model_inputs)
    if strategy == "argmax":
        return logits.argmax(dim=-1)
    if strategy == "sample":
        return sample_logits(logits, temperature, top_p)
    raise ValueError(f"unknown strategy: {strategy}")


@torch.no_grad()
def sample_action_grids(
    head,
    backbone,
    model_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    n_samples: int = 128,
    example: int = 0,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Draw complete grids while preserving each head's dependencies.

    Factorized and parallel cells are sampled from their one-pass logits.
    Autoregressive grids are generated causally, so later draws condition on
    the bins sampled earlier in the same grid.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    batch_size = model_inputs[0].shape[0]
    if not 0 <= example < batch_size:
        raise IndexError("example is outside the batch")

    selected = tuple(
        value[example : example + 1].expand(
            n_samples, *value.shape[1:]
        )
        for value in model_inputs
    )
    if hasattr(head, "generate"):
        return action_head_bins(
            head,
            backbone,
            selected,
            strategy="sample",
            temperature=temperature,
            top_p=top_p,
        )

    with evaluation_mode(backbone), evaluation_mode(head):
        logits = action_head_logits(
            head,
            backbone,
            tuple(value[:1] for value in selected),
        )[0]
    probabilities = nucleus_probabilities(logits, temperature, top_p)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    draws = torch.multinomial(flat, n_samples, replacement=True).T
    return draws.reshape(n_samples, *logits.shape[:-1])


@torch.no_grad()
def decode_action_chunk(
    head,
    backbone,
    model_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tokenizer,
    stats: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    horizon: int | None = None,
    action_dim: int | None = None,
    strategy: str = "argmax",
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Decode any action head to normalized or raw continuous actions."""
    bins = action_head_bins(
        head,
        backbone,
        model_inputs,
        strategy,
        temperature,
        top_p,
    )
    horizon = horizon or getattr(head, "horizon", ACTION_HORIZON)
    action_dim = action_dim or getattr(head, "action_dim", ACTION_DIM)
    if bins.ndim != 3 or bins.shape[1:] != (horizon, action_dim):
        raise ValueError(
            "head must return logits shaped [B, H, D, bins]"
        )
    grid = bins.cpu().numpy()
    normalized = torch.from_numpy(tokenizer.decode(grid))
    if stats is None:
        return normalized
    return denormalize_from_stats(normalized, stats, "action")


@torch.no_grad()
def decode_parallel_chunk(
    head,
    model_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tokenizer,
    stats: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    horizon: int = ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
    strategy: str = "argmax",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Backward-compatible wrapper for the original parallel-only API."""
    return decode_action_chunk(
        head,
        head.backbone,
        model_inputs,
        tokenizer,
        stats,
        horizon,
        action_dim,
        strategy,
        temperature,
    )


@torch.no_grad()
def evaluate_open_loop(
    head,
    loader,
    tokenizer,
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    backbone,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Aggregate padding-aware MAE by horizon offset and control."""
    error_sum = None
    valid_count = None
    with evaluation_mode(head):
        for batch in loader:
            model_inputs = prepare_batch(
                batch, stats, device, backbone
            )
            predicted = decode_action_chunk(
                head,
                backbone,
                model_inputs,
                tokenizer,
                stats,
                strategy="argmax",
            ).cpu()
            expert = torch.as_tensor(batch["action"]).float().cpu()
            timestep_valid = ~torch.as_tensor(
                batch.get(
                    "action_is_pad",
                    torch.zeros(expert.shape[:2], dtype=torch.bool),
                ),
                dtype=torch.bool,
            ).cpu()
            valid = timestep_valid.unsqueeze(-1).expand_as(expert)
            if error_sum is None:
                error_sum = torch.zeros_like(expert[0])
                valid_count = torch.zeros_like(expert[0])
            error_sum += ((predicted - expert).abs() * valid).sum(0)
            valid_count += valid.sum(0)
    if error_sum is None or valid_count is None:
        raise ValueError("validation loader produced no batches")
    mae = torch.full_like(error_sum, float("nan"))
    seen = valid_count > 0
    mae[seen] = error_sum[seen] / valid_count[seen]
    scale = torch.as_tensor(stats["action"]["std"]).float().clamp_min(1e-8)
    return {
        "mae": mae,
        "mae_in_standard_deviations": mae / scale[None],
        "valid_count": valid_count,
    }


def mean_absolute_error_by_timestep(
    predicted: torch.Tensor,
    expert: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error over batch and control dimensions."""
    if predicted.shape != expert.shape or predicted.ndim != 3:
        raise ValueError("predicted and expert must match [B, H, D]")
    errors = (predicted - expert).abs().mean(dim=-1)
    if pad_mask is None:
        return errors.mean(dim=0)
    if pad_mask.shape != errors.shape:
        raise ValueError("pad_mask must have shape [B, H]")
    valid = (~pad_mask).to(errors.dtype)
    counts = valid.sum(dim=0)
    if bool((counts == 0).any()):
        raise ValueError("at least one timestep is entirely padded")
    return (errors * valid).sum(dim=0) / counts
