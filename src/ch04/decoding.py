"""Temperature, top-p, and action-chunk decoding."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch

from ch04.constants import ACTION_DIM, ACTION_HORIZON
from ch04.data import denormalize_from_stats


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


@torch.no_grad()
def decode_parallel_chunk(
    head,
    model_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tokenizer,
    stats: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    horizon: int = ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
    temperature: float = 1.0,
    top_p: float = 0.95,
    greedy: bool = False,
) -> torch.Tensor:
    """Decode a parallel head to normalized or raw continuous actions."""
    with evaluation_mode(head):
        logits = head(*model_inputs)
    bins = sample_logits(logits, temperature, top_p, greedy)
    expected = horizon * action_dim
    if bins.shape[1] != expected:
        raise ValueError(
            f"head returned {bins.shape[1]} cells, expected {expected}"
        )
    grid = bins.reshape(-1, horizon, action_dim).cpu().numpy()
    normalized = torch.from_numpy(tokenizer.decode(grid))
    if stats is None:
        return normalized
    return denormalize_from_stats(normalized, stats, "action")


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
