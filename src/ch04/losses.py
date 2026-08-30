"""Action-grid shaping and masked categorical losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def expand_timestep_pad_mask(
    pad_mask: torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    """Expand ``[B, H]`` padding flags to ``[B, H, D]``."""
    if pad_mask.ndim != 2 or pad_mask.dtype != torch.bool:
        raise ValueError("pad_mask must be a bool tensor shaped [B, H]")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    return pad_mask.unsqueeze(-1).expand(-1, -1, action_dim)


def masked_token_cross_entropy(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Mean categorical CE after excluding padded action cells."""
    if (
        logits.ndim != target_bins.ndim + 1
        or target_bins.shape != logits.shape[:-1]
    ):
        raise ValueError("targets must match every logits axis except bins")
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
