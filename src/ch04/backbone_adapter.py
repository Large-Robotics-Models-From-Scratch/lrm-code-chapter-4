"""Small helpers around Chapter 3's two-stage backbone contract."""

from __future__ import annotations

import torch


def extend_position_ids(
    prefix_position_ids: torch.Tensor,
    extra_tokens: int,
) -> torch.Tensor:
    """Append compact action positions after each row's state position."""
    if prefix_position_ids.ndim != 2:
        raise ValueError("prefix_position_ids must have shape [B, N]")
    if extra_tokens < 0:
        raise ValueError("extra_tokens must be non-negative")
    if extra_tokens == 0:
        return prefix_position_ids
    offsets = torch.arange(
        1,
        extra_tokens + 1,
        device=prefix_position_ids.device,
        dtype=prefix_position_ids.dtype,
    )
    action_positions = prefix_position_ids[:, -1:] + offsets[None]
    return torch.cat([prefix_position_ids, action_positions], dim=1)


def encode_prefix(
    backbone,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    state: torch.Tensor,
    text_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Contextualize the Chapter 3 observation prefix."""
    return backbone(images, input_ids, state, text_attention_mask)


def gather_state_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Return the final state position from ``[image, text, state]``."""
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [B, N, D]")
    return hidden[:, -1]
