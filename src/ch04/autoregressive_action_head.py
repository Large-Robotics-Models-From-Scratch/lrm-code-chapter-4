"""Autoregressive action decoder from listing 4.3."""

from __future__ import annotations

import torch
import torch.nn as nn

from ch04.backbone_adapter import (
    causal_mask_with_prefix_padding,
    embed_inputs,
    prefix_valid_mask,
)
from ch04.constants import ACTION_BINS, SMOLLM_VOCAB, SMOLLM_WIDTH
from ch04.losses import masked_token_cross_entropy


class AutoregressiveActionHead(nn.Module):
    """Teacher-forced training and sequential action-token decoding."""

    def __init__(
        self,
        backbone,
        d_embed: int = SMOLLM_WIDTH,
        n_bins: int = ACTION_BINS,
        token_base: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(n_bins, int) or not 2 <= n_bins <= SMOLLM_VOCAB:
            raise ValueError("n_bins must lie in [2, SMOLLM_VOCAB]")
        self.backbone = backbone
        self.n_bins = n_bins
        self.token_base = (
            SMOLLM_VOCAB - n_bins if token_base is None else token_base
        )
        if not 0 <= self.token_base <= SMOLLM_VOCAB - n_bins:
            raise ValueError(
                "action token range must fit the native vocabulary"
            )
        self.action_decoder = nn.Linear(d_embed, n_bins)
        nn.init.normal_(self.action_decoder.weight, std=0.02)
        nn.init.zeros_(self.action_decoder.bias)

    def teacher_forced_logits(
        self,
        images: torch.Tensor,
        sequence_ids: torch.Tensor,
        state: torch.Tensor,
        target_bins: torch.Tensor,
    ) -> torch.Tensor:
        """Predict all grid tokens in one causal teacher-forced pass."""
        if target_bins.ndim != 2:
            raise ValueError("target_bins must have shape [B, G]")
        if target_bins.dtype != torch.long:
            raise TypeError("target_bins must have dtype torch.long")
        if target_bins.numel() == 0:
            raise ValueError("target_bins cannot be empty")
        if bool(((target_bins < 0) | (target_bins >= self.n_bins)).any()):
            raise ValueError("target bins must lie in [0, n_bins)")
        action_tokens = self.token_base + target_bins
        input_ids = torch.cat([sequence_ids, action_tokens[:, :-1]], dim=1)
        embeddings = embed_inputs(self.backbone, images, input_ids, state)
        valid = prefix_valid_mask(self.backbone, sequence_ids)
        mask = causal_mask_with_prefix_padding(
            valid,
            target_bins.shape[1] - 1,
            embeddings.dtype,
        )
        hidden = self.backbone.language_backbone(
            inputs_embeds=embeddings,
            attention_mask=mask,
        ).last_hidden_state
        grid = target_bins.shape[1]
        state_positions = (
            (sequence_ids == self.backbone.state_id)
            .to(torch.int64)
            .argmax(dim=1)
        )
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        first_hidden = hidden[rows, state_positions][:, None]
        previous_action_hidden = (
            hidden[:, -(grid - 1) :] if grid > 1 else None
        )
        action_hidden = (
            torch.cat([first_hidden, previous_action_hidden], dim=1)
            if previous_action_hidden is not None
            else first_hidden
        ).to(self.action_decoder.weight.dtype)
        return self.action_decoder(action_hidden).float()

    def loss(
        self,
        images: torch.Tensor,
        sequence_ids: torch.Tensor,
        state: torch.Tensor,
        target_bins: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        logits = self.teacher_forced_logits(
            images, sequence_ids, state, target_bins
        )
        return masked_token_cross_entropy(
            logits,
            target_bins,
            pad_mask=pad_mask,
            label_smoothing=label_smoothing,
        )

    @torch.no_grad()
    def decode(
        self,
        images: torch.Tensor,
        sequence_ids: torch.Tensor,
        state: torch.Tensor,
        grid_size: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Pedagogical uncached left-to-right decode.

        This deliberately favors clarity over speed. Production systems
        should replace the repeated prefix passes with the model's KV
        cache while preserving the same token ordering.
        """
        from ch04.decoding import evaluation_mode, sample_logits

        ids = sequence_ids
        generated = []
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        with evaluation_mode(self):
            valid = prefix_valid_mask(self.backbone, sequence_ids)
            for index in range(grid_size):
                embeddings = embed_inputs(self.backbone, images, ids, state)
                mask = causal_mask_with_prefix_padding(
                    valid, index, embeddings.dtype
                )
                hidden = self.backbone.language_backbone(
                    inputs_embeds=embeddings,
                    attention_mask=mask,
                ).last_hidden_state
                if index == 0:
                    state_positions = (
                        (sequence_ids == self.backbone.state_id)
                        .to(torch.int64)
                        .argmax(dim=1)
                    )
                    rows = torch.arange(
                        hidden.shape[0], device=hidden.device
                    )
                    last_hidden = hidden[rows, state_positions]
                else:
                    last_hidden = hidden[:, -1]
                last_hidden = last_hidden.to(
                    self.action_decoder.weight.dtype
                )
                logits = self.action_decoder(last_hidden).float()
                bins = sample_logits(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    greedy=greedy,
                )
                generated.append(bins)
                ids = torch.cat(
                    [ids, (self.token_base + bins)[:, None]], dim=1
                )
        return torch.stack(generated, dim=1)
