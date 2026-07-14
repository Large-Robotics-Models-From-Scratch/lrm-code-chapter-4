"""Parallel (one-shot categorical) action head -- the chapter's foil.

The manuscript's SS4.5.2 introduces this head *before* the shipped
autoregressive one (``autoregressive_action_head.py``, PR 4) to make
the autoregressive design's payoff legible: this head predicts every
``(timestep, joint)`` bin in a single forward pass, each position
scored independently by its own softmax over ``n_bins``. That
independence is also its failure mode -- nothing here lets the model
condition one joint's prediction on another, so it cannot express
correlations within a chunk (e.g. "if the wrist rotates this far, the
gripper should already be closing"). We build it anyway because it is
the natural first thing to try, and because the chapter needs a
concrete baseline to measure the autoregressive head against.

Architecture (prose spec, SS4.5.2): a two-layer MLP maps the pooled
backbone state to every bin's logit at once --
``Linear(d_embed, hidden) -> GELU -> Linear(hidden, H * D * n_bins)``
-- reshaped to ``[B, H, D, n_bins]``. The final layer's bias is
zero-initialized so a fresh head starts at (approximately) uniform
softmax: the manuscript's sanity check is that an untrained head
reports a loss near ``log(n_bins)`` (== log(256) ~= 5.545 nats), the
entropy of a uniform categorical distribution. If a fresh head instead
reported something far from that, it would signal a broken
initialization, not a model that "already knows something."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch04 import ACTION_DIM, CHUNK_H, N_BINS


def adjacent_bin_targets(
    target_bins: torch.Tensor, n_bins: int, eps: float
) -> torch.Tensor:
    """Soft targets: ``(1-eps)`` on the target bin, ``eps/2`` on each
    of its neighbors ``[B, H, D]`` -> ``[B, H, D, n_bins]``.

    Bin ids are ordinal (adjacent ids are adjacent physical values),
    but plain cross-entropy treats them as unrelated classes -- being
    off by one bin costs exactly as much as being off by 128. Spreading
    a little probability mass onto the immediate neighbors buys back
    some of that ordinal prior (manuscript Table 4.3, eps=0.05) without
    the cost of a full ordinal-regression loss.

    Edge bins (id 0 or ``n_bins - 1``) have only one neighbor, so that
    neighbor receives the *full* ``eps`` rather than ``eps/2`` --
    every row still sums to 1.
    """
    device = target_bins.device
    soft = torch.zeros(
        *target_bins.shape, n_bins, dtype=torch.float32, device=device
    )
    soft.scatter_(-1, target_bins.unsqueeze(-1), 1.0 - eps)

    has_left = target_bins > 0
    has_right = target_bins < n_bins - 1
    both = has_left & has_right
    # Interior bins split eps evenly; edge bins put all of eps on
    # their single neighbor.
    left_mass = torch.where(both, eps / 2, eps).float()
    right_mass = torch.where(both, eps / 2, eps).float()

    left = (target_bins - 1).clamp(min=0)
    right = (target_bins + 1).clamp(max=n_bins - 1)
    soft.scatter_add_(
        -1,
        left.unsqueeze(-1),
        (left_mass * has_left).unsqueeze(-1),
    )
    soft.scatter_add_(
        -1,
        right.unsqueeze(-1),
        (right_mass * has_right).unsqueeze(-1),
    )
    return soft


class ParallelActionHead(nn.Module):
    """One-shot categorical head: every bin predicted independently.

    Consumes the backbone's pooled final-state embedding (callers do
    the pooling; this head only ever sees ``[B, d_embed]``) and emits
    logits over ``n_bins`` for each of the ``chunk_h * action_dim``
    positions in a single forward pass -- no autoregressive
    conditioning between positions, by design (see module docstring).
    """

    def __init__(
        self,
        d_embed: int = 576,
        chunk_h: int = CHUNK_H,
        action_dim: int = ACTION_DIM,
        n_bins: int = N_BINS,
        hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.chunk_h = chunk_h
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, hidden),
            nn.GELU(),
            nn.Linear(hidden, chunk_h * action_dim * n_bins),
        )
        # Zero bias => logits start near 0 => softmax starts near
        # uniform (up to first-layer noise), so a fresh head's loss
        # lands near log(n_bins) -- see module docstring.
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """``[B, d_embed]`` -> logits ``[B, H, D, n_bins]``."""
        batch = pooled.shape[0]
        flat = self.mlp(pooled)
        return flat.view(
            batch, self.chunk_h, self.action_dim, self.n_bins
        )

    def loss(
        self,
        pooled: torch.Tensor,
        target_bins: torch.Tensor,
        label_smoothing_eps: float = 0.0,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy on ``[B, H, D]`` int64 targets.

        ``label_smoothing_eps == 0`` (default) is plain per-position
        categorical cross-entropy, computed with ``F.cross_entropy``
        for numerical stability. ``label_smoothing_eps > 0`` swaps in
        the adjacent-bin soft targets from ``adjacent_bin_targets`` and
        computes ``-(soft * log_softmax(logits)).sum(-1).mean()`` by
        hand, since ``F.cross_entropy`` only accepts hard class
        indices.

        ``pad_mask`` (optional, ``[B, H]`` bool, ``True`` == ignore)
        handles lerobot's episode-boundary padding: chunks that run
        past an episode's end repeat the last action and flag those
        steps in ``action_is_pad`` (SS4.6). Training on the repeated
        rows teaches the policy to freeze at episode end, so a padded
        timestep is dropped from the mean (its full ``[B, H, D]`` row
        of positions is zeroed, then divided by the unmasked count).
        Default ``None`` keeps the original unmasked behavior.
        """
        logits = self(pooled)  # [B, H, D, n_bins]
        if label_smoothing_eps == 0.0:
            per = F.cross_entropy(
                logits.reshape(-1, self.n_bins),
                target_bins.reshape(-1),
                reduction="none",
            ).view(*target_bins.shape)  # [B, H, D]
        else:
            soft = adjacent_bin_targets(
                target_bins, self.n_bins, label_smoothing_eps
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            per = -(soft * log_probs).sum(dim=-1)  # [B, H, D]
        if pad_mask is None:
            return per.mean()
        keep = (~pad_mask).unsqueeze(-1).to(per.dtype)  # [B, H, 1]
        return (per * keep).sum() / keep.expand_as(
            per
        ).sum().clamp(min=1.0)

    def sample(
        self, pooled: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Independent per-position categorical sampling.

        ``[B, d_embed]`` -> bin ids ``[B, H, D]``. Each of the
        ``H * D`` positions is drawn independently from its own
        softmax -- this is the head's namesake limitation: sampling
        one joint's bin has no way to see what any other joint just
        sampled, so a sampled chunk can be internally incoherent even
        when every individual marginal is well-calibrated.
        """
        logits = self(pooled)  # [B, H, D, n_bins]
        batch = logits.shape[0]
        probs = torch.softmax(logits / temperature, dim=-1)
        flat_probs = probs.reshape(-1, self.n_bins)
        flat_bins = torch.multinomial(flat_probs, num_samples=1)
        return flat_bins.view(
            batch, self.chunk_h, self.action_dim
        ).to(torch.int64)
