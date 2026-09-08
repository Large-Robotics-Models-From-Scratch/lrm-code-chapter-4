"""Autoregressive action decoder from manuscript listings 4.4 and 4.5."""

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
    """Teacher-forced training and cached action-token generation.

    The head appends one position per future timestep, matching the
    SmolVLA-style granularity :class:`~ch04.parallel_action_head.
    ParallelDecodeActionHead` already uses. A timestep's input embedding is
    the sum of its ``D`` per-control bin embeddings, and its hidden state is
    read out into all ``D`` control distributions at once. The two heads
    therefore differ only in causal versus bidirectional attention and in
    whether the appended positions carry learned slots or realized action
    bins, which is what makes the section 4.6 comparison honest.

    The cost is that the ``D`` controls of a single timestep are
    conditionally independent given the observation and every earlier
    timestep: nothing conditions the gripper on the wrist bin chosen for the
    same timestep. Autoregressive conditioning here buys temporal structure
    only. ``tests/test_attention_semantics.py`` pins both halves of that
    statement.
    """

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
        # One shared table over every (control, bin) pair. The offset makes
        # control identity part of the lookup, so summing the D embeddings
        # of a timestep cannot confuse "gripper bin 7" with "wrist bin 7".
        self.action_embeddings = nn.Embedding(action_dim * n_bins, d_embed)
        self.action_decoder = nn.Linear(d_embed, action_dim * n_bins)
        self.register_buffer(
            "control_offsets",
            torch.arange(action_dim) * n_bins,
            persistent=False,
        )
        nn.init.normal_(self.action_embeddings.weight, std=0.02)
        nn.init.normal_(self.action_decoder.weight, std=0.02)
        nn.init.zeros_(self.action_decoder.bias)

    @property
    def serial_steps(self) -> int:
        """Dependent decode steps at inference: one per timestep.

        Read this rather than ``grid``. ``grid`` counts target cells
        (``H * D``) and stopped being the suffix length when the head moved
        to timestep tokens.
        """
        return self.horizon

    def _grid_targets(self, target_bins: torch.Tensor) -> torch.Tensor:
        """Normalize ``[B, H, D]`` or flattened ``[B, H*D]`` targets."""
        if target_bins.ndim == 2:
            if target_bins.shape[1] != self.grid:
                raise ValueError(
                    f"target_bins must contain {self.grid} cells"
                )
            target_bins = target_bins.view(
                -1, self.horizon, self.action_dim
            )
        if target_bins.ndim != 3:
            raise ValueError(
                "target_bins must have shape [B, H, D] or [B, G]"
            )
        if target_bins.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"target_bins must contain {self.grid} cells"
            )
        if target_bins.dtype != torch.long:
            raise TypeError("target_bins must have dtype torch.long")
        if bool(((target_bins < 0) | (target_bins >= self.n_bins)).any()):
            raise ValueError("target bins must lie in [0, n_bins)")
        return target_bins

    def _embed_timesteps(self, bins: torch.Tensor) -> torch.Tensor:
        """Sum a ``[B, T, D]`` bin grid into ``[B, T, d_embed]`` tokens."""
        return self.action_embeddings(bins + self.control_offsets).sum(-2)

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Read ``[..., d_embed]`` hidden states into ``[..., D, bins]``."""
        logits = self.action_decoder(
            hidden.to(self.action_decoder.weight.dtype)
        )
        return logits.unflatten(-1, (self.action_dim, self.n_bins)).float()

    def teacher_forced_logits(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        text_attention_mask: torch.Tensor,
        target_bins: torch.Tensor,
    ) -> torch.Tensor:
        """Predict every timestep in one causal teacher-forced pass."""
        target_bins = self._grid_targets(target_bins)
        prefix, prefix_valid, prefix_positions = (
            self.backbone.embed_inputs(
                images,
                input_ids,
                state,
                text_attention_mask,
            )
        )
        action_inputs = self._embed_timesteps(target_bins[:, :-1])
        sequence = torch.cat(
            [prefix, action_inputs.to(prefix.dtype)], dim=1
        )
        attention_mask = torch.cat(
            [
                prefix_valid,
                torch.ones(
                    target_bins.shape[0],
                    self.horizon - 1,
                    device=prefix_valid.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        hidden = self.backbone.contextualize(
            sequence,
            attention_mask,
            extend_position_ids(prefix_positions, self.horizon - 1),
        )
        # The last prefix state predicts timestep 0; each supplied timestep
        # embedding predicts the next one.
        action_hidden = hidden[:, prefix.shape[1] - 1 :]
        return self._decode(action_hidden)

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
        target_grid = self._grid_targets(target_bins)
        if pad_mask is not None:
            if pad_mask.ndim == 2 and pad_mask.shape[1] == self.horizon:
                pad_mask = pad_mask.unsqueeze(-1).expand_as(target_grid)
            elif pad_mask.ndim == 2 and pad_mask.shape[1] == self.grid:
                pad_mask = pad_mask.view_as(target_grid)
        logits = self.teacher_forced_logits(
            images,
            input_ids,
            state,
            text_attention_mask,
            target_grid,
        )
        return masked_token_cross_entropy(
            logits,
            target_grid,
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
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Generate one ``[B, H, D]`` grid using SmolLM2's KV cache.

        Costs ``H`` serial steps, one per timestep, because each timestep's
        controls are drawn together from a single hidden state.
        """
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        from ch04.decoding import evaluation_mode, sample_logits

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

            for timestep in range(self.horizon):
                next_bins = sample_logits(
                    self._decode(hidden),
                    temperature=temperature,
                    top_p=top_p,
                    greedy=temperature == 0.0,
                )
                generated.append(next_bins)
                if timestep + 1 == self.horizon:
                    break

                next_embedding = self._embed_timesteps(
                    next_bins[:, None]
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            next_bins.shape[0],
                            1,
                            device=prefix.device,
                            dtype=torch.bool,
                        ),
                    ],
                    dim=1,
                )
                cache_position = torch.tensor(
                    [prefix_length + timestep],
                    device=prefix.device,
                )
                next_position = prefix_positions[:, -1:] + timestep + 1
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

        return torch.stack(generated, dim=1)
