"""Tiny stand-in for ch3's ``UnifiedEmbeddingBackbone``.

The fusion adapter (and, later, the action heads and training loop)
only touch a narrow slice of the Chapter 3 backbone: its projection
modules, its embedding table, its ``tokenize_instruction`` /
``build_sequence_ids`` helpers, its ``image_id`` / ``state_id``
placeholders, and its ``language_backbone``. Loading the real
SmolLM2 + SigLIP stack for a unit test is slow and needs network, so
this fake reproduces exactly that surface at toy dimensions (vocab 64,
width 8) with deterministic, cheap stand-ins. Integration tests still
run against the real backbone; these fakes keep the unit tests fast.

Kept in ``tests/`` (not ``src/``) so it never ships, and minimal on
purpose: later tasks reuse it, so it grows only when a real contract
demands it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FAKE_VOCAB = 64
FAKE_WIDTH = 8
FAKE_SIGLIP_WIDTH = 8
FAKE_PATCHES = 196  # per camera, matching ch3's 196 -> 392 total
FAKE_IMAGE_ID = 62
FAKE_STATE_ID = 63
FAKE_TEXT_IDS = [1, 2, 3]  # a fixed 3-token "instruction"


class _FakeVision(nn.Module):
    """Stub SigLIP: ``[B*2, 3, 224, 224]`` -> ``[B*2, 196, 8]``.

    Frozen like the real one (params ``requires_grad=False``) so the
    adapter's ``parameters_backbone`` exclude-by-module can be tested.
    """

    def __init__(self) -> None:
        super().__init__()
        out_dim = FAKE_PATCHES * FAKE_SIGLIP_WIDTH
        self.proj = nn.Linear(3 * 224 * 224, out_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = images.flatten(1)  # [B*2, 3*224*224]
        out = self.proj(flat)  # [B*2, 196*8]
        return out.reshape(images.shape[0], FAKE_PATCHES, FAKE_SIGLIP_WIDTH)


class _FakeLM(nn.Module):
    """Deterministic stub decoder over ``inputs_embeds``.

    Applies one position-independent linear map to every position, so a
    cached incremental decode is trivially equal to a full-sequence
    forward: the equivalence unit test can run here, and the real
    SmolLM2 causal check lives in the integration test. Returns an
    object exposing ``last_hidden_state`` and ``past_key_values`` to
    mirror a Hugging Face ``AutoModel`` output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Linear(FAKE_WIDTH, FAKE_WIDTH)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        h = self.mix(inputs_embeds)
        seen = 0 if past_key_values is None else int(past_key_values)
        cache = seen + inputs_embeds.shape[1] if use_cache else None

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = h
        out.past_key_values = cache
        return out


class _CausalFakeLM(nn.Module):
    """Genuinely causal stub decoder over ``inputs_embeds``.

    ``_FakeLM`` applies the same map to every position, so its output
    at position ``i`` ignores every other position -- it is "causal"
    only trivially, and a causality test run against it proves
    nothing. This variant mixes a *cumulative* (prefix) sum instead:
    position ``i``'s output depends on inputs at positions ``0..i`` and
    on none after, which is exactly the dependency the autoregressive
    head's causality test exercises. Deterministic and network-free.

    The cache carries the running prefix sum (last column) plus the
    seen count, so an incremental single-token decode reproduces a
    full-sequence forward -- keeping it drop-in with the adapter's
    cache-aware ``forward`` like ``_FakeLM``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Linear(FAKE_WIDTH, FAKE_WIDTH)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        if past_key_values is None:
            base = torch.zeros(
                inputs_embeds.shape[0], 1, FAKE_WIDTH,
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )
            seen = 0
        else:
            base, seen = past_key_values
        prefix = base + torch.cumsum(inputs_embeds, dim=1)
        h = self.mix(prefix)
        cache = None
        if use_cache:
            cache = (prefix[:, -1:, :], seen + inputs_embeds.shape[1])

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = h
        out.past_key_values = cache
        return out


class FakeBackbone(nn.Module):
    """Toy ``UnifiedEmbeddingBackbone`` with the adapter's contract.

    ``causal=True`` swaps the position-wise ``_FakeLM`` for the
    genuinely causal ``_CausalFakeLM`` -- needed by tests (the
    autoregressive head's causality check) that would be vacuous
    against a position-wise map.
    """

    def __init__(self, causal: bool = False) -> None:
        super().__init__()
        self.image_id = FAKE_IMAGE_ID
        self.state_id = FAKE_STATE_ID
        self.vision_encoder = _FakeVision()
        self.img_proj = nn.Linear(FAKE_SIGLIP_WIDTH, FAKE_WIDTH)
        self.state_proj = nn.Linear(6, FAKE_WIDTH)
        self.embed_tokens = nn.Embedding(FAKE_VOCAB, FAKE_WIDTH)
        self.language_backbone = (
            _CausalFakeLM() if causal else _FakeLM()
        )

    def tokenize_instruction(self, instruction: str) -> list[int]:
        return list(FAKE_TEXT_IDS)

    def build_sequence_ids(self, text_ids: list[int]) -> list[int]:
        image_ids = [self.image_id] * (2 * FAKE_PATCHES)
        return image_ids + list(text_ids) + [self.state_id]
