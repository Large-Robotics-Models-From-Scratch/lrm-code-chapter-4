"""Tests for the ch3 -> ch4 fusion adapter.

Unit tests run against ``tests/fakes.FakeBackbone`` (toy dims, no
network). The integration test builds the real
``UnifiedEmbeddingBackbone`` and proves the adapter reconstructs ch3's
own fused forward exactly.
"""

import pytest
import torch
from fakes import FAKE_PATCHES, FAKE_TEXT_IDS, FAKE_WIDTH, FakeBackbone

from ch04.fusion_adapter import FusionAdapter

IMG_TOKENS = 2 * FAKE_PATCHES  # 392
L = len(FAKE_TEXT_IDS)
PREFIX_LEN = IMG_TOKENS + L + 1


def _fake_batch(batch_size: int = 2):
    torch.manual_seed(0)
    return {
        "observation.images.up": torch.rand(batch_size, 3, 224, 224),
        "observation.images.side": torch.rand(batch_size, 3, 224, 224),
        "observation.state": torch.rand(batch_size, 6),
        "task": ["pick up the cube"] * batch_size,
    }


@pytest.fixture
def adapter():
    return FusionAdapter(FakeBackbone())


def test_encode_prefix_shape(adapter):
    batch = _fake_batch(2)
    prefix = adapter.encode_prefix(batch)
    assert prefix.shape == (2, PREFIX_LEN, FAKE_WIDTH)
    assert prefix.dtype == adapter.backbone.embed_tokens.weight.dtype


def test_encode_prefix_masked_scatter_placement(adapter):
    """Image rows are the projected features; last row is the state."""
    batch = _fake_batch(1)
    bb = adapter.backbone
    cams = torch.stack(
        [batch["observation.images.up"], batch["observation.images.side"]],
        dim=1,
    )  # [1, 2, 3, 224, 224]
    feats = bb.vision_encoder(cams.flatten(0, 1))
    expected_img = bb.img_proj(feats).reshape(1, IMG_TOKENS, FAKE_WIDTH)
    expected_state = bb.state_proj(batch["observation.state"])
    text_emb = bb.embed_tokens(torch.tensor([list(FAKE_TEXT_IDS)]))

    prefix = adapter.encode_prefix(batch)
    assert torch.allclose(prefix[:, :IMG_TOKENS], expected_img, atol=1e-6)
    assert torch.allclose(
        prefix[:, IMG_TOKENS : IMG_TOKENS + L], text_emb, atol=1e-6
    )
    assert torch.allclose(prefix[:, -1], expected_state, atol=1e-6)


def test_encode_prefix_rejects_mixed_instructions(adapter):
    batch = _fake_batch(2)
    batch["task"] = ["pick up the cube", "put down the cube"]
    with pytest.raises(ValueError, match="identical"):
        adapter.encode_prefix(batch)


def test_embed_shape_and_dtype(adapter):
    ids = torch.tensor([[1, 2, 3, 4]])
    emb = adapter.embed(ids)
    assert emb.shape == (1, 4, FAKE_WIDTH)
    assert emb.dtype == adapter.backbone.embed_tokens.weight.dtype


def test_forward_returns_hidden_and_cache(adapter):
    batch = _fake_batch(2)
    prefix = adapter.encode_prefix(batch)
    hidden, past = adapter.forward(prefix)
    assert hidden.shape == (2, PREFIX_LEN, FAKE_WIDTH)
    assert past is None  # use_cache defaults to False


def test_forward_cache_equivalence(adapter):
    """Incremental cached decode == full-sequence forward (fake LM)."""
    batch = _fake_batch(1)
    prefix = adapter.encode_prefix(batch)
    extra = adapter.embed(torch.tensor([[5, 6, 7]]))
    full_in = torch.cat([prefix, extra], dim=1)
    full_hidden, _ = adapter.forward(full_in)

    _, past = adapter.forward(prefix, use_cache=True)
    steps = []
    for t in range(extra.shape[1]):
        step = extra[:, t : t + 1]
        h, past = adapter.forward(
            step, past_key_values=past, use_cache=True
        )
        steps.append(h)
    incremental = torch.cat(steps, dim=1)
    assert torch.allclose(
        incremental, full_hidden[:, prefix.shape[1] :], atol=1e-6
    )


def test_parameters_backbone_excludes_vision(adapter):
    trainable = set(adapter.parameters_backbone())
    vision = set(adapter.backbone.vision_encoder.parameters())
    assert trainable.isdisjoint(vision)
    for mod in (
        adapter.backbone.img_proj,
        adapter.backbone.state_proj,
        adapter.backbone.embed_tokens,
        adapter.backbone.language_backbone,
    ):
        assert set(mod.parameters()) <= trainable


@pytest.mark.integration
@pytest.mark.slow
def test_integration_faithful_to_ch3():
    from ch03 import UnifiedEmbeddingBackbone

    torch.manual_seed(0)
    backbone = UnifiedEmbeddingBackbone().float()
    backbone.eval()
    adapter = FusionAdapter(backbone)

    instruction = "pick up the cube"
    batch = {
        "observation.images.up": torch.rand(1, 3, 224, 224),
        "observation.images.side": torch.rand(1, 3, 224, 224),
        "observation.state": torch.rand(1, 6),
        "task": [instruction],
    }
    text_ids = backbone.tokenize_instruction(instruction)
    seq_len = 392 + len(text_ids) + 1

    with torch.no_grad():
        prefix = adapter.encode_prefix(batch)
        assert prefix.shape == (1, seq_len, 576)

        # (b) adapter forward matches ch3's own fused forward.
        from ch03.preprocess import preprocess_image

        cams = torch.stack(
            [
                batch["observation.images.up"],
                batch["observation.images.side"],
            ],
            dim=1,
        )
        cams = preprocess_image(cams.flatten(0, 1)).reshape(
            1, 2, 3, 224, 224
        )
        sequence_ids = torch.tensor([backbone.build_sequence_ids(text_ids)])
        ref = backbone(cams, sequence_ids, batch["observation.state"])
        hidden, _ = adapter.forward(prefix)
        assert torch.allclose(hidden, ref, rtol=1e-4, atol=1e-4)

        # (c) KV-cache decode equivalence over 3 extra positions.
        extra = adapter.embed(torch.tensor([[10, 11, 12]]))
        full_hidden, _ = adapter.forward(torch.cat([prefix, extra], dim=1))
        _, past = adapter.forward(prefix, use_cache=True)
        steps = []
        for t in range(extra.shape[1]):
            h, past = adapter.forward(
                extra[:, t : t + 1], past_key_values=past, use_cache=True
            )
            steps.append(h)
        incremental = torch.cat(steps, dim=1)
        assert torch.allclose(
            incremental, full_hidden[:, prefix.shape[1] :],
            rtol=1e-3, atol=1e-3,
        )
