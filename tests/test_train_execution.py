import math

import numpy as np
import pytest
import torch

from ch04 import ActionTokenizer
from ch04.decoding import evaluation_mode
from ch04.execution import TemporalEnsembler, execute_chunk
from ch04.train import (
    action_head_logits,
    action_metrics,
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
    assert latest["best_validation_loss"] == latest["validation_loss"]
    assert [record["step"] for record in latest["history"]] == [1.0]
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
    # The run is already complete, so resume restores the complete curve
    # without performing more optimizer steps.
    assert [record["step"] for record in history] == [1.0, 2.0, 3.0, 4.0]


def test_checkpoints_are_mirrored_to_durable_storage(
    fake_backbone, fake_stats, tmp_path
):
    from ch04 import ParallelDecodeActionHead

    local = tmp_path / "local"
    durable = tmp_path / "drive"
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    train_action_head(
        head,
        fake_backbone,
        [_batch()],
        fake_stats,
        _tokenizer(),
        "cpu",
        total_steps=1,
        warmup_steps=0,
        log_every=1,
        checkpoint_every=1,
        checkpoint_dir=local,
        checkpoint_mirror_dir=durable,
        validation_loader=[_batch()],
        snapshot_steps=(1,),
    )
    expected = {"latest.pt", "best.pt", "step000001.pt"}
    assert {path.name for path in durable.glob("*.pt")} == expected
    assert not list(durable.glob(".*.tmp"))
    for name in expected:
        local_state = torch.load(local / name, weights_only=False)
        durable_state = torch.load(durable / name, weights_only=False)
        assert durable_state["step"] == local_state["step"] == 1


def test_drive_mirror_failure_keeps_local_checkpoint_and_training_alive(
    fake_backbone, fake_stats, monkeypatch, tmp_path, capsys
):
    import ch04.train as train_module
    from ch04 import ParallelDecodeActionHead

    def fail_copy(*_args, **_kwargs):
        raise OSError("Drive disconnected")

    monkeypatch.setattr(train_module.shutil, "copy2", fail_copy)
    local = tmp_path / "local"
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    history = train_action_head(
        head,
        fake_backbone,
        [_batch()],
        fake_stats,
        _tokenizer(),
        "cpu",
        total_steps=1,
        warmup_steps=0,
        log_every=1,
        checkpoint_every=1,
        checkpoint_dir=local,
        checkpoint_mirror_dir=tmp_path / "drive",
    )
    assert len(history) == 1
    assert (local / "latest.pt").exists()
    assert "warning: could not mirror latest.pt" in capsys.readouterr().out


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
    assert [record["step"] for record in history] == [1.0, 2.0]
    assert history[0]["head_lr"] == pytest.approx(1e-4)
    assert history[0]["backbone_lr"] == pytest.approx(1e-5)


def test_action_metrics_report_per_control_accuracy_and_mae(fake_stats):
    batch = _batch()
    tokenizer = _tokenizer()
    from ch04.data import action_targets

    bins, pad = action_targets(batch, fake_stats, tokenizer, "cpu")
    logits = torch.full((*bins.shape, tokenizer.n_bins), -20.0)
    logits.scatter_(-1, bins.unsqueeze(-1), 20.0)
    metrics = action_metrics(
        logits, bins, pad, batch["action"], fake_stats, tokenizer
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["accuracy_by_control"] == pytest.approx([1.0] * 6)
    assert len(metrics["mae_in_std_by_control"]) == 6
    assert len(metrics["mae_raw_by_control"]) == 6


def test_tensorboard_receives_loss_accuracy_and_per_control_metrics(
    fake_backbone, fake_stats, monkeypatch, tmp_path
):
    import ch04.train as train_module
    from ch04 import ParallelDecodeActionHead

    class Writer:
        def __init__(self):
            self.scalars = []
            self.closed = False

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, float(value), step))

        def flush(self):
            pass

        def close(self):
            self.closed = True

    writer = Writer()
    monkeypatch.setattr(
        train_module, "_make_summary_writer", lambda *_: writer
    )
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    train_action_head(
        head,
        fake_backbone,
        [_batch()],
        fake_stats,
        _tokenizer(),
        "cpu",
        total_steps=1,
        warmup_steps=0,
        log_every=1,
        validation_loader=[_batch()],
        validate_every=1,
        tensorboard_log_dir=tmp_path,
    )
    tags = {tag for tag, _, _ in writer.scalars}
    for expected in (
        "loss/train",
        "loss/held_out",
        "accuracy/train",
        "accuracy/held_out",
        "accuracy_by_control/train_0",
        "mae_in_std_by_control/held_out_5",
    ):
        assert expected in tags
    assert writer.closed


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

@pytest.mark.parametrize(
    "head_name",
    ["factorized", "autoregressive", "parallel"],
)
def test_shared_training_loop_supports_every_head(
    head_name, fake_backbone, fake_stats
):
    from ch04 import (
        AutoregressiveActionHead,
        FactorizedActionHead,
        ParallelDecodeActionHead,
    )

    builders = {
        "factorized": lambda: FactorizedActionHead(d_embed=12),
        "autoregressive": lambda: AutoregressiveActionHead(
            fake_backbone, d_embed=12
        ),
        "parallel": lambda: ParallelDecodeActionHead(
            fake_backbone, d_embed=12
        ),
    }
    history = train_action_head(
        builders[head_name](),
        fake_backbone,
        [_batch()],
        fake_stats,
        _tokenizer(),
        "cpu",
        total_steps=1,
        warmup_steps=0,
        log_every=1,
    )
    assert len(history) == 1
    assert math.isfinite(history[0]["loss"])


def test_autoregressive_logits_require_teacher_forcing_targets(
    fake_backbone, model_inputs
):
    from ch04 import AutoregressiveActionHead

    head = AutoregressiveActionHead(fake_backbone, d_embed=12)
    with pytest.raises(ValueError, match="target_bins"):
        action_head_logits(head, fake_backbone, model_inputs)


def test_history_carries_the_held_out_curve(fake_backbone, fake_stats):
    """A training curve is only readable next to a held-out curve."""
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    batch = _batch()
    history = train_action_head(
        head, fake_backbone, [batch] * 2, fake_stats, _tokenizer(),
        "cpu", total_steps=4, warmup_steps=0, log_every=1,
        validation_loader=[batch], validate_every=2,
    )
    assert [record["step"] for record in history] == [1.0, 2.0, 3.0, 4.0]
    measured = [
        record["step"] for record in history
        if not math.isnan(record["validation_loss"])
    ]
    # Validation runs after steps 2 and 4, annotating those records only,
    # so matplotlib skips the rest as NaN instead of drawing a staircase.
    assert measured == [2.0, 4.0]
    assert all(
        record["validation_loss"] > 0
        for record in history
        if not math.isnan(record["validation_loss"])
    )
    assert math.isnan(history[0]["validation_loss"])


def test_training_resumes_train_mode_after_validating(
    fake_backbone, fake_stats
):
    """A validation pass must not leave the policy in eval mode."""
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    batch = _batch()
    train_action_head(
        head, fake_backbone, [batch], fake_stats, _tokenizer(), "cpu",
        total_steps=1, warmup_steps=0, log_every=1,
        validation_loader=[batch], validate_every=1,
    )
    assert head.training
    assert fake_backbone.training


def test_validation_is_skipped_without_a_loader(
    fake_backbone, fake_stats
):
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    history = train_action_head(
        head, fake_backbone, [_batch()], fake_stats, _tokenizer(),
        "cpu", total_steps=1, warmup_steps=0, log_every=1,
    )
    assert math.isnan(history[0]["validation_loss"])


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
