"""Bidirectional parallel action decoder from listing 4.4."""

import torch
import torch.nn as nn

from ch04.backbone_adapter import embed_inputs, prefix_valid_mask
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
        self.horizon = horizon
        self.action_dim = action_dim
        self.grid = horizon * action_dim
        self.n_bins = n_bins
        self.slot = nn.Parameter(0.02 * torch.randn(d_embed))
        self.readout = nn.Linear(d_embed, n_bins)
        nn.init.normal_(self.readout.weight, std=0.02)
        nn.init.zeros_(self.readout.bias)

    def _mask(
        self,
        n_prefix: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        prefix_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_total = n_prefix + self.grid
        mask = torch.zeros(n_total, n_total, device=device, dtype=dtype)
        above = torch.triu(
            torch.ones(n_total, n_total, dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask[above] = -torch.inf
        mask[n_prefix:, n_prefix:] = 0.0
        mask = mask[None, None]
        if prefix_valid is None:
            return mask
        if prefix_valid.shape[1] != n_prefix:
            raise ValueError("prefix_valid length must equal n_prefix")
        mask = mask.expand(prefix_valid.shape[0], -1, -1, -1).clone()
        valid_keys = torch.cat(
            [
                prefix_valid,
                torch.ones(
                    prefix_valid.shape[0],
                    self.grid,
                    device=device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        mask.masked_fill_(~valid_keys[:, None, None, :], -torch.inf)
        return mask

    def forward(
        self,
        images: torch.Tensor,
        sequence_ids: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        prefix = embed_inputs(self.backbone, images, sequence_ids, state)
        batch_size, n_prefix, _ = prefix.shape
        slots = self.slot.expand(batch_size, self.grid, -1)
        sequence = torch.cat([prefix, slots.to(prefix.dtype)], dim=1)
        valid = prefix_valid_mask(self.backbone, sequence_ids)
        mask = self._mask(
            n_prefix,
            sequence.device,
            sequence.dtype,
            prefix_valid=valid,
        )
        hidden = self.backbone.language_backbone(
            inputs_embeds=sequence,
            attention_mask=mask,
        ).last_hidden_state
        action_hidden = hidden[:, -self.grid :].to(
            self.readout.weight.dtype
        )
        return self.readout(action_hidden).float()
