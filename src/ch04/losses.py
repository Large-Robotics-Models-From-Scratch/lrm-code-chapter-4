"""Action-grid shaping and masked categorical losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flatten_action_grid(grid: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B, H, D]`` to timestep-major ``[B, H * D]``."""
    if grid.ndim != 3:
        raise ValueError("action grid must have shape [B, H, D]")
    return grid.reshape(grid.shape[0], -1)


def expand_timestep_pad_mask(
    pad_mask: torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    """Expand ``[B, H]`` padding flags to grid order ``[B, H * D]``."""
    if pad_mask.ndim != 2 or pad_mask.dtype != torch.bool:
        raise ValueError("pad_mask must be a bool tensor shaped [B, H]")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    return pad_mask.repeat_interleave(action_dim, dim=1)


def masked_token_cross_entropy(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Mean token CE after excluding padded cells."""
    if logits.ndim != 3 or target_bins.shape != logits.shape[:2]:
        raise ValueError("expected logits [B, G, bins] and targets [B, G]")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must lie in [0, 1)")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        target_bins.reshape(-1).long(),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if pad_mask is None:
        return losses.mean()
    if pad_mask.shape != target_bins.shape or pad_mask.dtype != torch.bool:
        raise ValueError("pad_mask must be bool and match target shape")
    keep = ~pad_mask.reshape(-1)
    if not bool(keep.any()):
        raise ValueError("the batch contains no valid action tokens")
    return losses[keep].mean()


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean categorical entropy in nats across all grid cells."""
    log_probs = logits.float().log_softmax(dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).mean()
