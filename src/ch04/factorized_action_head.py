"""One-shot factorized baseline from listing 4.2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch04.constants import (
    ACTION_BINS,
    ACTION_DIM,
    ACTION_HORIZON,
    SMOLLM_WIDTH,
)


class FactorizedActionHead(nn.Module):
    """Predict every action-grid cell independently from one state."""

    def __init__(
        self,
        d_embed: int = SMOLLM_WIDTH,
        horizon: int = ACTION_HORIZON,
        action_dim: int = ACTION_DIM,
        n_bins: int = ACTION_BINS,
        d_slot: int = 256,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.grid = horizon * action_dim
        self.n_bins = n_bins
        self.slots = nn.Parameter(0.02 * torch.randn(self.grid, d_slot))
        self.projection = nn.Linear(d_embed, d_slot)
        self.action_decoder = nn.Linear(d_slot, n_bins)
        nn.init.normal_(self.action_decoder.weight, std=0.02)
        nn.init.zeros_(self.action_decoder.bias)

    def forward(self, state_hidden: torch.Tensor) -> torch.Tensor:
        if state_hidden.ndim != 2:
            raise ValueError("state_hidden must have shape [B, d_embed]")
        state_hidden = state_hidden.to(self.projection.weight.dtype)
        hidden = self.projection(state_hidden)[:, None, :]
        logits = self.action_decoder(F.gelu(hidden + self.slots))
        return logits.view(
            state_hidden.shape[0],
            self.horizon,
            self.action_dim,
            self.n_bins,
        ).float()
