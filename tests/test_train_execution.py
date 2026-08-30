import math

import numpy as np
import pytest
import torch

from ch04 import ActionTokenizer
from ch04.decoding import evaluation_mode
from ch04.execution import TemporalEnsembler, execute_chunk
from ch04.train import (
    backbone_parameters,
    head_parameters,
    make_optimizer,
    train_action_head,
    warmup_cosine_multiplier,
)


def _batch():
    return {
        "observation.images.up": torch.rand(1, 3, 20, 30),
        "observation.images.side": torch.rand(1, 3, 20, 30),
        "observation.state": torch.rand(1, 6),
        "action": torch.rand(1, 16, 6),
        "action_is_pad": torch.zeros(1, 16, dtype=torch.bool),
        "task": ["pick"],
    }


def _tokenizer():
    return ActionTokenizer(
        -5 * np.ones(6, dtype=np.float32),
        5 * np.ones(6, dtype=np.float32),
    )


def test_warmup_cosine_schedule():
    assert warmup_cosine_multiplier(0, 10, 100) == 0.1
    assert warmup_cosine_multiplier(10, 10, 100) == 1.0
    assert warmup_cosine_multiplier(100, 10, 100) == pytest.approx(0.0)


def test_optimizer_separates_head_and_backbone_learning_rates(
    fake_backbone,
):
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    head_params = head_parameters(head, fake_backbone)
    backbone_params = backbone_parameters(fake_backbone)
    assert {id(parameter) for parameter in head_params} == {
        id(head.slots),
        id(head.readout.weight),
        id(head.readout.bias),
    }
    optimizer = make_optimizer(head, fake_backbone)
    assert [group["lr"] for group in optimizer.param_groups] == [
        1e-4,
        1e-5,
    ]
    assert backbone_params


def test_evaluation_mode_restores_child_modes(fake_backbone):
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    head.train()
    fake_backbone.eval()
    assert head.training and not fake_backbone.training
    with evaluation_mode(head):
        assert not head.training
        assert not fake_backbone.training
    assert head.training
    assert not fake_backbone.training


def test_training_updates_backbone_and_saves_full_best_checkpoint(
    fake_backbone, fake_stats, tmp_path
):
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    batch = _batch()
    before = fake_backbone.state_encoder.layer.weight.detach().clone()
    train_action_head(
        head,
        fake_backbone,
        [batch],
        fake_stats,
        _tokenizer(),
        "cpu",
        total_steps=1,
        warmup_steps=0,
        log_every=1,
        checkpoint_every=1,
        checkpoint_dir=tmp_path,
        validation_loader=[batch],
        snapshot_steps=(1,),
    )
    assert head.training
    assert not torch.equal(
        fake_backbone.state_encoder.layer.weight, before
    )
    latest = torch.load(tmp_path / "latest.pt")
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "step000001.pt").exists()
    assert "scheduler" in latest
    assert "normalization" in latest
    assert "validation_loss" in latest
    assert set(latest["model"]) == {"head", "backbone"}
    assert not any(
        name.startswith("backbone.")
        for name in latest["model"]["head"]
    )

    saved_slots = latest["model"]["head"]["slots"].clone()
    with torch.no_grad():
        head.slots.add_(1.0)
    train_action_head(
        head,
        fake_backbone,
        [batch],
        fake_stats,
        _tokenizer(),
        "cpu",
        total_steps=1,
        warmup_steps=0,
        log_every=1,
        resume_from=tmp_path / "latest.pt",
    )
    torch.testing.assert_close(head.slots, saved_slots)


def test_resume_restores_optimizer_scheduler_and_step(
    fake_backbone, fake_stats, tmp_path
):
    """Resuming must continue the run, not restart the optimizer state."""
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    batch = _batch()
    common = dict(
        total_steps=4,
        warmup_steps=0,
        log_every=1,
        checkpoint_every=2,
        checkpoint_dir=tmp_path,
        validation_loader=[batch],
    )
    train_action_head(
        head, fake_backbone, [batch] * 2, fake_stats, _tokenizer(),
        "cpu", **common,
    )
    checkpoint = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert checkpoint["step"] == 4
    assert checkpoint["optimizer"]["state"], "optimizer state is empty"
    assert checkpoint["scheduler"]["last_epoch"] == 4

    history = train_action_head(
        head, fake_backbone, [batch] * 2, fake_stats, _tokenizer(),
        "cpu", resume_from=tmp_path / "latest.pt", **common,
    )
    # The run is already complete, so a resume performs no further steps.
    assert history == []


def test_snapshots_are_off_by_default(
    fake_backbone, fake_stats, tmp_path
):
    """A policy snapshot is ~1 GB, so keep only latest.pt and best.pt."""
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    batch = _batch()
    train_action_head(
        head, fake_backbone, [batch], fake_stats, _tokenizer(), "cpu",
        total_steps=1, warmup_steps=0, log_every=1, checkpoint_every=1,
        checkpoint_dir=tmp_path, validation_loader=[batch],
    )
    written = sorted(path.name for path in tmp_path.glob("*.pt"))
    assert written == ["best.pt", "latest.pt"]


def test_training_continues_after_loader_epoch_boundary(
    fake_backbone, fake_stats
):
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    history = train_action_head(
        head,
        fake_backbone,
        [_batch()],
        fake_stats,
        _tokenizer(),
        "cpu",
        total_steps=2,
        warmup_steps=0,
        log_every=1,
    )
    assert [record["step"] for record in history] == [0.0, 1.0]
    assert history[0]["head_lr"] == pytest.approx(1e-4)
    assert history[0]["backbone_lr"] == pytest.approx(1e-5)


def test_held_out_loss_respects_its_batch_bound(
    fake_backbone, fake_stats
):
    """An unbounded validation pass costs one forward per held-out frame."""
    from ch04 import ParallelDecodeActionHead
    from ch04.train import held_out_loss

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    loader = [_batch() for _ in range(5)]
    before = fake_backbone.vision_encoder.calls
    held_out_loss(
        head, fake_backbone, loader, fake_stats, _tokenizer(), "cpu",
        max_batches=2,
    )
    assert fake_backbone.vision_encoder.calls - before == 2

    before = fake_backbone.vision_encoder.calls
    held_out_loss(
        head, fake_backbone, loader, fake_stats, _tokenizer(), "cpu"
    )
    assert fake_backbone.vision_encoder.calls - before == 5


def test_upcast_promotes_low_precision_trainable_weights():
    """bfloat16 rounds a 1e-5 relative AdamW update to nothing."""
    from ch04.train import upcast_trainable_parameters

    layer = torch.nn.Linear(4, 4).to(torch.bfloat16)
    before = layer.weight.detach().float().clone()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-5)
    layer(torch.randn(2, 4, dtype=torch.bfloat16)).sum().backward()
    optimizer.step()
    stuck = (layer.weight.detach().float() == before).float().mean()
    assert stuck > 0.5, "expected bfloat16 to swallow most of the update"

    layer = torch.nn.Linear(4, 4).to(torch.bfloat16)
    promoted = upcast_trainable_parameters(layer)
    assert set(promoted) == {"weight", "bias"}
    assert layer.weight.dtype == torch.float32
    before = layer.weight.detach().clone()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-5)
    layer(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    assert torch.all(layer.weight.detach() != before)


def test_upcast_leaves_frozen_parameters_alone():
    from ch04.train import upcast_trainable_parameters

    layer = torch.nn.Linear(4, 4).to(torch.bfloat16)
    layer.bias.requires_grad_(False)
    assert upcast_trainable_parameters(layer) == ["weight"]
    assert layer.bias.dtype == torch.bfloat16


def test_temporal_ensemble_matches_weighted_average():
    ensemble = TemporalEnsembler(decay=math.log(2))
    first = torch.tensor([[0.0], [2.0]])
    second = torch.tensor([[4.0], [8.0]])
    assert ensemble.add(first).item() == 0.0
    result = ensemble.add(second)
    assert result.item() == pytest.approx((2.0 + 4.0 * 0.5) / 1.5)


def test_execute_chunk_preserves_order():
    chunk = torch.arange(12).reshape(3, 4)
    controls = list(execute_chunk(chunk))
    assert torch.equal(torch.stack(controls), chunk)
