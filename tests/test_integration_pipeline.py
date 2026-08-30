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


@pytest.mark.integration
def test_tokenizer_and_stats_never_see_validation_frames():
    """Fitting on held-out actions would leak the evaluation split."""
    from ch04.data import (
        collect_normalized_actions,
        make_chunked_dataloaders,
    )

    train, validation, stats = make_chunked_dataloaders(
        batch_size=2, validation_fraction=0.2, seed=3
    )
    train_episodes = set(train.dataset.episodes)
    seen = {
        int(value)
        for value in train.dataset.hf_dataset["episode_index"]
    }
    assert seen == train_episodes
    held_out = {
        int(value)
        for value in validation.dataset.hf_dataset["episode_index"]
    }
    assert seen.isdisjoint(held_out)

    actions = collect_normalized_actions(train, stats)
    assert actions.shape[0] == len(train.dataset.hf_dataset)
    assert actions.shape[1] == 6


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "head_name", ["factorized", "autoregressive", "parallel"]
)
def test_every_head_trains_end_to_end_on_the_real_backbone(head_name):
    """Each manuscript head must complete real optimizer steps.

    This uses a synthetic batch rather than the LeRobot loader so it
    checks the head/backbone/optimizer path without a dataset download.
    """
    import numpy as np
    from ch03 import VLABackbone

    from ch04 import ActionTokenizer
    from ch04.cli import build_action_head
    from ch04.train import train_action_head

    torch.manual_seed(0)
    backbone = VLABackbone()
    head = build_action_head(head_name, backbone)
    batch = {
        "observation.images.up": torch.rand(1, 3, 96, 128),
        "observation.images.side": torch.rand(1, 3, 96, 128),
        "observation.state": torch.rand(1, 6),
        "action": torch.rand(1, 16, 6),
        "action_is_pad": torch.zeros(1, 16, dtype=torch.bool),
        "task": ["pick up the object"],
    }
    stats = {
        key: {"mean": torch.zeros(6), "std": torch.ones(6)}
        for key in ("action", "observation.state")
    }
    tokenizer = ActionTokenizer(
        -5 * np.ones(6, dtype=np.float32),
        5 * np.ones(6, dtype=np.float32),
    )
    trunk = backbone.language_backbone.layers[0].mlp.up_proj
    before = trunk.weight.detach().float().clone()

    history = train_action_head(
        head,
        backbone,
        [batch] * 2,
        stats,
        tokenizer,
        "cpu",
        total_steps=2,
        warmup_steps=0,
        log_every=1,
        validation_loader=[batch],
    )
    assert len(history) == 2
    assert all(np.isfinite(record["loss"]) for record in history)
    # log(256) ~ 5.55 at initialization; a broken head diverges instead.
    assert history[0]["loss"] < 8.0

    after = trunk.weight.detach().float()
    moved = float((after != before).float().mean())
    assert moved > 0.9, (
        "the pretrained trunk barely moved; float32 master weights "
        f"are not in effect (only {moved:.1%} of elements changed)"
    )
