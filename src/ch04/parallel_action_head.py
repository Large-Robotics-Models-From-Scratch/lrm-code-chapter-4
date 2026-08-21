"""Bidirectional parallel action decoder from manuscript listing 4.5."""

import torch
import torch.nn as nn

from ch04.backbone_adapter import extend_position_ids
from ch04.constants import (
    ACTION_BINS,
    ACTION_DIM,
    ACTION_HORIZON,
    SMOLLM_WIDTH,
)


class ParallelDecodeActionHead(nn.Module):
    """Decode the full action grid in one bidirectional action block."""

    def __init__(
        self,
        backbone,
        d_embed: int = SMOLLM_WIDTH,
        horizon: int = ACTION_HORIZON,
        action_dim: int = ACTION_DIM,
        n_bins: int = ACTION_BINS,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        set_attention = getattr(
            backbone.language_backbone,
            "set_attn_implementation",
            None,
        )
        if set_attention is not None:
            set_attention("eager")
        self.horizon = horizon
        self.action_dim = action_dim
        self.grid = horizon * action_dim
        self.n_bins = n_bins
        self.slots = nn.Parameter(
            0.02 * torch.randn(self.grid, d_embed)
        )
        self.readout = nn.Linear(d_embed, n_bins)
        nn.init.normal_(self.readout.weight, std=0.02)
        nn.init.zeros_(self.readout.bias)

    def _mask(
        self,
        prefix_valid: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Causal prefix plus a fully connected action-slot block."""
        if prefix_valid.ndim != 2:
            raise ValueError("prefix_valid must have shape [B, N]")
        prefix_valid = prefix_valid.bool()
        batch_size, n_prefix = prefix_valid.shape
        n_total = n_prefix + self.grid
        mask = torch.zeros(
            batch_size,
            1,
            n_total,
            n_total,
            device=prefix_valid.device,
            dtype=dtype,
        )
        future = torch.triu(
            torch.ones(
                n_total,
                n_total,
                device=prefix_valid.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        mask.masked_fill_(future[None, None], -torch.inf)
        mask[:, :, n_prefix:, n_prefix:] = 0.0
        key_valid = torch.cat(
            [
                prefix_valid,
                torch.ones(
                    batch_size,
                    self.grid,
                    device=prefix_valid.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        mask.masked_fill_(~key_valid[:, None, None, :], -torch.inf)
        return mask

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix, prefix_valid, prefix_positions = (
            self.backbone.embed_inputs(
                images,
                input_ids,
                state,
                text_attention_mask,
            )
        )
        batch_size = prefix.shape[0]
        slots = self.slots[None].expand(batch_size, -1, -1)
        sequence = torch.cat([prefix, slots.to(prefix.dtype)], dim=1)
        hidden = self.backbone.contextualize(
            sequence,
            self._mask(prefix_valid, sequence.dtype),
            extend_position_ids(prefix_positions, self.grid),
        )
        action_hidden = hidden[:, -self.grid :].to(
            self.readout.weight.dtype
        )
        return self.readout(action_hidden).float()
