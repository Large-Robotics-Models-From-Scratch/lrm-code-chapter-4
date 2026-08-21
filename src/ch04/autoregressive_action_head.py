"""Autoregressive action decoder from manuscript listings 4.3 and 4.4."""

from __future__ import annotations

import torch
import torch.nn as nn

from ch04.backbone_adapter import extend_position_ids
from ch04.constants import (
    ACTION_BINS,
    ACTION_DIM,
    ACTION_HORIZON,
    SMOLLM_WIDTH,
)
from ch04.losses import masked_token_cross_entropy


class AutoregressiveActionHead(nn.Module):
    """Teacher-forced training and cached action-token generation."""

    def __init__(
        self,
        backbone,
        d_embed: int = SMOLLM_WIDTH,
        horizon: int = ACTION_HORIZON,
        action_dim: int = ACTION_DIM,
        n_bins: int = ACTION_BINS,
    ) -> None:
        super().__init__()
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer greater than one")
        if horizon < 1 or action_dim < 1:
            raise ValueError("horizon and action_dim must be positive")
        self.backbone = backbone
        self.horizon = horizon
        self.action_dim = action_dim
        self.grid = horizon * action_dim
        self.n_bins = n_bins
        self.action_embeddings = nn.Embedding(n_bins, d_embed)
        self.action_decoder = nn.Linear(d_embed, n_bins)
        nn.init.normal_(self.action_embeddings.weight, std=0.02)
        nn.init.normal_(self.action_decoder.weight, std=0.02)
        nn.init.zeros_(self.action_decoder.bias)

    def _flatten_targets(self, target_bins: torch.Tensor) -> torch.Tensor:
        if target_bins.ndim == 3:
            target_bins = target_bins.flatten(1)
        if target_bins.ndim != 2:
            raise ValueError(
                "target_bins must have shape [B, H, D] or [B, G]"
            )
        if target_bins.shape[1] != self.grid:
            raise ValueError(f"target_bins must contain {self.grid} cells")
        if target_bins.dtype != torch.long:
            raise TypeError("target_bins must have dtype torch.long")
        if bool(((target_bins < 0) | (target_bins >= self.n_bins)).any()):
            raise ValueError("target bins must lie in [0, n_bins)")
        return target_bins

    def teacher_forced_logits(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        text_attention_mask: torch.Tensor,
        target_bins: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the full grid in one causal teacher-forced pass."""
        target_bins = self._flatten_targets(target_bins)
        prefix, prefix_valid, prefix_positions = (
            self.backbone.embed_inputs(
                images,
                input_ids,
                state,
                text_attention_mask,
            )
        )
        action_inputs = self.action_embeddings(target_bins[:, :-1])
        sequence = torch.cat(
            [prefix, action_inputs.to(prefix.dtype)], dim=1
        )
        attention_mask = torch.cat(
            [
                prefix_valid,
                torch.ones_like(target_bins[:, :-1], dtype=torch.bool),
            ],
            dim=1,
        )
        hidden = self.backbone.contextualize(
            sequence,
            attention_mask,
            extend_position_ids(prefix_positions, self.grid - 1),
        )
        action_hidden = hidden[:, prefix.shape[1] - 1 :].to(
            self.action_decoder.weight.dtype
        )
        return self.action_decoder(action_hidden).float()

    def loss(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        text_attention_mask: torch.Tensor,
        target_bins: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
    ) -> torch.Tensor:
        target_bins = self._flatten_targets(target_bins)
        if pad_mask is not None and pad_mask.shape[1] == self.horizon:
            pad_mask = pad_mask.repeat_interleave(self.action_dim, dim=1)
        logits = self.teacher_forced_logits(
            images,
            input_ids,
            state,
            text_attention_mask,
            target_bins,
        )
        return masked_token_cross_entropy(
            logits,
            target_bins,
            pad_mask=pad_mask,
            label_smoothing=label_smoothing,
        )

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        text_attention_mask: torch.Tensor,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Generate one ``[B, H, D]`` grid using SmolLM2's KV cache."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        from ch04.decoding import evaluation_mode

        with evaluation_mode(self):
            prefix, attention_mask, prefix_positions = (
                self.backbone.embed_inputs(
                    images,
                    input_ids,
                    state,
                    text_attention_mask,
                )
            )
            prefix_length = prefix.shape[1]
            cache_position = torch.arange(
                prefix_length, device=prefix.device
            )
            outputs = self.backbone.language_backbone(
                inputs_embeds=prefix,
                attention_mask=attention_mask,
                position_ids=prefix_positions,
                cache_position=cache_position,
                use_cache=True,
            )
            cache = outputs.past_key_values
            hidden = outputs.last_hidden_state[:, -1]
            generated = []

            for position in range(self.grid):
                logits = self.action_decoder(
                    hidden.to(self.action_decoder.weight.dtype)
                ).float()
                if temperature == 0.0:
                    next_bin = logits.argmax(dim=-1)
                else:
                    probabilities = (logits / temperature).softmax(-1)
                    next_bin = torch.multinomial(
                        probabilities, 1
                    ).squeeze(1)
                generated.append(next_bin)
                if position + 1 == self.grid:
                    break

                next_embedding = self.action_embeddings(next_bin[:, None])
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones_like(
                            next_bin[:, None], dtype=torch.bool
                        ),
                    ],
                    dim=1,
                )
                cache_position = torch.tensor(
                    [prefix_length + position],
                    device=prefix.device,
                )
                next_position = prefix_positions[:, -1:] + position + 1
                outputs = self.backbone.language_backbone(
                    inputs_embeds=next_embedding.to(prefix.dtype),
                    attention_mask=attention_mask,
                    position_ids=next_position,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True,
                )
                cache = outputs.past_key_values
                hidden = outputs.last_hidden_state[:, -1]

        bins = torch.stack(generated, dim=1)
        return bins.view(-1, self.horizon, self.action_dim)
