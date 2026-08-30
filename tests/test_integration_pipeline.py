"""Opt-in tests for the live Chapter 2 and Chapter 3 handoffs."""

import pytest
import torch


@pytest.mark.integration
@pytest.mark.slow
def test_real_backbone_parallel_head_contract():
    from ch03 import VLABackbone

    from ch04 import AutoregressiveActionHead, ParallelDecodeActionHead
    from ch04.train import make_optimizer

    backbone = VLABackbone().eval()
    backbone.language_backbone.set_attn_implementation("eager")
    head = ParallelDecodeActionHead(backbone).eval()
    text = backbone.tokenizer(
        ["pick up the object"], padding=True, return_tensors="pt"
    )
    # Exercise the real Chapter 2 -> Chapter 3 handoff. Chapter 4 must
    # not pre-resize frames: VisionEncoder owns the one shared bicubic
    # preprocessing path.
    images = torch.rand(1, 2, 3, 480, 640)
    state = torch.rand(1, 6)
    with torch.no_grad():
        logits = head(
            images,
            text.input_ids,
            state,
            text.attention_mask.bool(),
        )
    assert logits.shape == (1, 16, 6, 256)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()

    assert backbone.language_backbone.config.vocab_size == 49_152
    assert (
        backbone.language_backbone.get_input_embeddings().num_embeddings
        == 49_152
    )

    autoregressive = AutoregressiveActionHead(
        backbone, horizon=1, action_dim=2
    ).eval()
    with torch.no_grad():
        generated = autoregressive.generate(
            images,
            text.input_ids,
            state,
            text.attention_mask.bool(),
        )
    assert generated.shape == (1, 1, 2)
    assert generated.dtype == torch.long
    optimizer = make_optimizer(head, backbone)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(
        id(parameter) not in optimized
        for parameter in backbone.vision_encoder.siglip.parameters()
    )
    assert id(backbone.vision_encoder.project.weight) in optimized


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
