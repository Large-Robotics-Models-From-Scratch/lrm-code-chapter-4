"""Opt-in tests for the live Chapter 2 and Chapter 3 handoffs."""

import pytest
import torch


@pytest.mark.integration
@pytest.mark.slow
def test_real_backbone_parallel_head_contract():
    from ch03 import VLABackbone

    from ch04 import ParallelDecodeActionHead

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
