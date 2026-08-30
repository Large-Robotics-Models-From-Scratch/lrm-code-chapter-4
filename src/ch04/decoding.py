"""Temperature, top-p, and action-chunk decoding."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch

from ch04.constants import ACTION_DIM, ACTION_HORIZON
from ch04.data import denormalize_from_stats, prepare_batch


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
    """Decode a parallel head to normalized or raw continuous actions."""
    with evaluation_mode(head):
        logits = head(*model_inputs)
    bins = select_bins(logits, strategy, temperature)
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
            predicted = decode_parallel_chunk(
                head,
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
