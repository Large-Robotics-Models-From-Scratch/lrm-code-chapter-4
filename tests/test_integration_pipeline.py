"""Opt-in tests for the live Chapter 2 and Chapter 3 handoffs."""

import pytest
import torch


@pytest.mark.integration
@pytest.mark.slow
def test_real_backbone_parallel_head_contract():
    from ch03 import VLABackbone

    from ch04 import ParallelDecodeActionHead
    from ch04.constants import MAX_INSTRUCTION_TOKENS
    from ch04.data import _padded_sequence_ids

    backbone = VLABackbone().eval()
    head = ParallelDecodeActionHead(backbone).eval()
    text = backbone.tokenize_instruction("pick up the object")
    ids = torch.tensor([backbone.build_sequence_ids(text)])
    images = torch.rand(1, 2, 3, 224, 224)
    state = torch.rand(1, 6)
    with torch.no_grad():
        logits = head(images, ids, state)
    assert logits.shape == (1, 96, 256)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()

    # A short instruction must produce the same action logits alone or
    # beside a longer instruction. Compact position ids prevent rectangular
    # batch padding from changing the short row's RoPE positions.
    long_text = backbone.tokenize_instruction(
        "pick up the object and place it carefully on the target"
    )
    single_ids = _padded_sequence_ids(
        backbone, [text], max_text_tokens=MAX_INSTRUCTION_TOKENS
    )
    mixed_ids = _padded_sequence_ids(
        backbone, [text, long_text],
        max_text_tokens=MAX_INSTRUCTION_TOKENS,
    )
    with torch.no_grad():
        single = head(images, single_ids, state)
        mixed = head(
            images.expand(2, -1, -1, -1, -1).contiguous(),
            mixed_ids,
            state.expand(2, -1).contiguous(),
        )
    torch.testing.assert_close(single[0], mixed[0], atol=2e-5, rtol=2e-5)


@pytest.mark.integration
def test_real_lerobot_chunk_contract():
    from ch04.data import make_chunked_dataloader

    loader, stats = make_chunked_dataloader(batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["action"].shape == (2, 16, 6)
    assert batch["action"].dtype == torch.float32
    assert batch["action_is_pad"].shape == (2, 16)
    assert batch["action_is_pad"].dtype == torch.bool
    assert stats["action"]["mean"].shape == (6,)


@pytest.mark.integration
def test_episode_split_is_disjoint():
    from ch04.data import make_chunked_dataloaders

    train, validation, _ = make_chunked_dataloaders(
        batch_size=2, validation_fraction=0.2, seed=3
    )
    train_episodes = set(train.dataset.episodes)
    validation_episodes = set(validation.dataset.episodes)
    assert train_episodes
    assert validation_episodes
    assert train_episodes.isdisjoint(validation_episodes)
