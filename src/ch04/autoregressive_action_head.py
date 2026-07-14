"""Autoregressive action head -- the head this chapter ships.

The same transformer that reads the scene decodes the action: we
append the action-chunk tokens to the fused prefix and let the causal
backbone predict each bin from everything before it, so the model can
express the within-chunk joint correlations the parallel head
(SS4.5.2) cannot. Actions live on *reserved* vocabulary ids
(``act_token_base + bin`` -- the last ``n_bins`` SmolLM2 slots), which
keeps ch3's pretrained text and image embeddings untouched
(SS4.3.1 / SS4.5.4). This module trains with teacher forcing (the
whole target chunk is fed at once, shifted right); the constrained,
step-by-step decode ships with the policy in PR 7.

Mirrors Listing 4.10: map bins to reserved ids, shift-right with BOS,
embed, concat after the prefix, run the causal forward, read out the
last ``T`` positions, cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveActionHead(nn.Module):
    """Teacher-forced autoregressive decode over reserved action ids.

    ``forward`` returns the scalar cross-entropy training loss;
    ``logits`` exposes the same path's per-position logits for
    diagnostics. The prefix is the fused observation embedding from
    ``FusionAdapter.encode_prefix`` (``[B, P, d_embed]``); targets are
    integer bin ids ``[B, T]`` with ``T = H * D``.
    """

    def __init__(
        self,
        fusion,
        d_embed: int = 576,
        n_bins: int = 256,
        act_token_base: int = 48896,
        bos_id: int = 1,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.n_bins = n_bins
        self.act_token_base = act_token_base
        self.bos_id = bos_id
        self.readout = nn.Linear(d_embed, n_bins)
        # Zero bias => fresh logits sit near 0 => softmax near uniform,
        # so an untrained head reports loss ~ log(n_bins) -- the same
        # sanity check the parallel head uses (SS4.5.2).
        nn.init.zeros_(self.readout.bias)

    def logits(self, prefix, target_bins):
        """Teacher-forced logits ``[B, T, n_bins]``.

        Map bins to reserved ids, shift right with BOS, embed, concat
        after the prefix, run the causal backbone, read out the final
        ``T`` positions. Shared by ``forward`` (which adds the CE).
        """
        self._validate(target_bins)
        batch, horizon = target_bins.shape
        # Reserved-id mapping keeps ch3's embeddings untouched; the
        # shift-right (BOS first, drop the last target) makes position
        # t predict from tokens strictly before it -- teacher forcing.
        tok = self.act_token_base + target_bins
        start = torch.full(
            (batch, 1), self.bos_id,
            dtype=tok.dtype, device=tok.device,
        )
        tok_in = torch.cat([start, tok[:, :-1]], dim=1)
        action_embeds = self.fusion.embed(tok_in).to(prefix.dtype)
        seq = torch.cat([prefix, action_embeds], dim=1)
        hidden, _ = self.fusion.forward(seq)
        # Read out only the action positions (the last T of the fused
        # sequence). readout runs in the hidden dtype (bf16 on the real
        # backbone); the loss upcasts (see forward).
        return self.readout(hidden[:, -horizon:, :])

    def forward(self, prefix, target_bins, pad_mask=None):
        """Teacher-forced cross-entropy loss (scalar).

        ``pad_mask`` (optional, ``[B, T]`` bool, ``True`` == ignore)
        handles lerobot's episode-boundary padding: chunks that run
        past an episode's end repeat the last action and flag those
        steps in ``action_is_pad`` (SS4.6). Training on the repeated
        rows teaches the policy to freeze at episode end, so the caller
        expands the ``[B, H]`` frame mask to token level
        (``repeat_interleave`` by ``action_dim`` in ``train.py``) and
        passes it here; masked positions are dropped from the mean (CE
        with ``reduction="none"``, zeroed, divided by the unmasked
        count). Default ``None`` keeps the original unmasked behavior.
        """
        logits = self.logits(prefix, target_bins)
        # bf16 gotcha: cast logits to float32 before cross_entropy. On
        # the real backbone hidden (hence logits) is bf16, whose
        # softmax is too coarse at the ~log(256) scale; float32 CE is
        # standard practice and keeps the loss stable and comparable.
        flat_logits = logits.reshape(-1, self.n_bins).float()
        flat_targets = target_bins.reshape(-1)
        if pad_mask is None:
            return F.cross_entropy(flat_logits, flat_targets)
        ce = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        )
        keep = (~pad_mask.reshape(-1)).to(ce.dtype)
        return (ce * keep).sum() / keep.sum().clamp(min=1.0)

    def _validate(self, target_bins) -> None:
        if torch.is_floating_point(target_bins) or (
            target_bins.dtype == torch.bool
        ):
            raise ValueError(
                "target_bins must be an integer tensor, got dtype "
                f"{target_bins.dtype}"
            )
        if target_bins.numel() and (
            int(target_bins.min()) < 0
            or int(target_bins.max()) >= self.n_bins
        ):
            raise ValueError(
                "target_bins values must be in [0, "
                f"{self.n_bins}); got range ["
                f"{int(target_bins.min())}, "
                f"{int(target_bins.max())}]"
            )
