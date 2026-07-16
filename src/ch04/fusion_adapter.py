"""Fusion adapter: the Chapter 3 -> Chapter 4 bridge.

The manuscript's action-head listings assume a small surface --
``fusion.encode_prefix``, ``fusion.embed``, ``fusion.forward`` -- that
Chapter 3's ``UnifiedEmbeddingBackbone`` does not expose. This adapter
provides that surface by *composing* the backbone, never touching it:
no assignment into ``ch03`` objects, no monkeypatching. It replicates
the backbone's own embedding construction (vision projection, text
lookup, two ``masked_scatter`` splices) up to -- but not including --
the ``language_backbone`` call, so the fused prefix it hands the action
head is byte-for-byte what ch3's ``forward`` would have fed SmolLM2.

Why split the backbone's forward here? The action head needs two things
ch3's monolithic ``forward`` never exposes: the *prefix embeddings*
before the language model runs (so it can append action-token
embeddings), and a *cache-aware* language-model call (so it can decode
one action token at a time). ``encode_prefix`` gives the first;
``forward`` gives the second.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ch03.preprocess import preprocess_image


def stack_cameras(batch: dict) -> torch.Tensor:
    """Stack up + side cameras to ``[B, 2, 3, 224, 224]``, ``[0, 1]``.

    Each camera arrives as ``[B, 3, H, W]`` in ``[0, 1]``;
    ``preprocess_image`` resizes to SigLIP's ``224 x 224`` without
    touching the value range, then the two views stack on a new camera
    axis (up first, then side -- ch3's fixed order).

    Lives here (torch + ch3 only, no lerobot) so both ``encode_prefix``
    and ``chunk_data.prepare_images`` share one implementation without
    dragging lerobot into the adapter's dependency surface.
    """
    up = preprocess_image(batch["observation.images.up"])
    side = preprocess_image(batch["observation.images.side"])
    return torch.stack([up, side], dim=1)  # [B, 2, 3, 224, 224]


class FusionAdapter(nn.Module):
    """Compose ch3's backbone into the action head's fusion surface."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        # Registered as a submodule so ``.to()``/``.eval()``/state_dict
        # flow through, but we only ever *read* its attributes.
        self.backbone = backbone
        # This dataset carries a single shared instruction, so the fused
        # sequence_ids are constant across the run. Cache them per
        # instruction string to skip re-tokenizing every batch.
        self._sequence_id_cache: dict[str, list[int]] = {}

    def _sequence_ids(self, instruction: str) -> list[int]:
        cached = self._sequence_id_cache.get(instruction)
        if cached is None:
            text_ids = self.backbone.tokenize_instruction(instruction)
            cached = self.backbone.build_sequence_ids(text_ids)
            self._sequence_id_cache[instruction] = cached
        return cached

    def encode_prefix(self, batch: dict) -> torch.Tensor:
        """Fuse image, text, and state into ``[B, 392 + L + 1, 576]``.

        Replicates ch3's ``forward`` body up to (not including) the
        ``language_backbone`` call: stack the two cameras (up then side,
        ch3's order), run frozen SigLIP + ``img_proj``, look up the text
        ids, then splice the projected image and state tokens into their
        reserved placeholder rows with ``masked_scatter``.

        ``batch`` keys: ``observation.images.up`` / ``...side``
        (``[B, 3, H, W]`` in ``[0, 1]``), ``observation.state``
        (``[B, 6]``), ``task`` (``list[str]``). All ``task`` strings
        must be identical -- variable-length instructions need padded
        ``sequence_ids``, which ch3 leaves to the caller and this
        single-task chapter does not implement.
        """
        bb = self.backbone
        tasks = batch["task"]
        if any(t != tasks[0] for t in tasks):
            raise ValueError(
                "encode_prefix needs identical instructions across the "
                "batch (this single-task dataset shares one string); "
                "variable-length instructions require padded "
                "sequence_ids, which is out of scope for this chapter."
            )

        cameras = stack_cameras(batch)  # [B, 2, 3, 224, 224]
        batch_size = cameras.shape[0]

        siglip = bb.vision_encoder(cameras.flatten(0, 1))  # [B*2, P, 768]
        width = bb.img_proj.out_features
        img = bb.img_proj(siglip).reshape(batch_size, -1, width)

        ids = self._sequence_ids(tasks[0])
        weight = bb.embed_tokens.weight
        sequence_ids = torch.tensor(
            [ids] * batch_size, dtype=torch.long, device=weight.device
        )
        emb = bb.embed_tokens(sequence_ids)  # [B, N, 576]

        img = img.to(emb.dtype)
        state_tok = bb.state_proj(batch["observation.state"]).to(emb.dtype)
        img_mask = (sequence_ids == bb.image_id).unsqueeze(-1)
        st_mask = (sequence_ids == bb.state_id).unsqueeze(-1)
        emb = emb.masked_scatter(img_mask, img)  # fill the image rows
        emb = emb.masked_scatter(st_mask, state_tok)  # fill the state row
        return emb

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup, ``[B, T]`` -> ``[B, T, 576]``.

        The action head appends action-token embeddings to the prefix;
        this is the same table ch3 splices into, so the two streams
        share one dtype and one geometry.
        """
        return self.backbone.embed_tokens(token_ids)

    def forward(
        self,
        seq_embeds: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """Run SmolLM2 on ``inputs_embeds`` (causal by construction).

        SmolLM2 is a causal decoder, so the causal mask is inherent -- no
        mask is passed. Returns ``(last_hidden_state [B, T, 576],
        past_key_values)``; with ``use_cache=True`` the returned cache
        feeds the next incremental single-token step.
        """
        out = self.backbone.language_backbone(
            inputs_embeds=seq_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return out.last_hidden_state, out.past_key_values

    def parameters_backbone(self):
        """Trainable backbone params -- the optimizer's second group.

        Yields ``img_proj``, ``state_proj``, ``embed_tokens``, and
        ``language_backbone`` parameters and excludes the vision encoder
        (frozen SigLIP). We exclude by *module* rather than by the
        ``requires_grad`` flag so the group is stable even if SigLIP were
        ever unfrozen -- membership expresses the design intent, not the
        current flag state.
        """
        for module in (
            self.backbone.img_proj,
            self.backbone.state_proj,
            self.backbone.embed_tokens,
            self.backbone.language_backbone,
        ):
            yield from module.parameters()
