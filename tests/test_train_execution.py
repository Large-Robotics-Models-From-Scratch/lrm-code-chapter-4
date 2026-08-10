import math

import numpy as np
import pytest
import torch

from ch04 import ActionTokenizer
from ch04.decoding import evaluation_mode
from ch04.execution import TemporalEnsembler, execute_chunk
from ch04.train import (
    freeze_backbone,
    head_only_state_dict,
    head_parameters,
    train_action_head,
    warmup_cosine_multiplier,
)


def test_warmup_cosine_schedule():
    assert warmup_cosine_multiplier(0, 10, 100) == 0.0
    assert warmup_cosine_multiplier(10, 10, 100) == 1.0
    assert warmup_cosine_multiplier(100, 10, 100) == pytest.approx(0.0)


def test_head_parameters_exclude_backbone(fake_backbone):
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    freeze_backbone(fake_backbone)
    params = head_parameters(head, fake_backbone)
    assert {id(p) for p in params} == {
        id(head.slot),
        id(head.readout.weight),
        id(head.readout.bias),
    }
    assert all(
        not parameter.requires_grad
        for parameter in fake_backbone.parameters()
    )


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


def test_training_keeps_backbone_eval_and_saves_compact_resume_state(
    fake_backbone, fake_stats, tmp_path
):
    from ch04 import ParallelDecodeActionHead

    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    batch = {
        "observation.images.up": torch.rand(1, 3, 20, 30),
        "observation.images.side": torch.rand(1, 3, 20, 30),
        "observation.state": torch.rand(1, 6),
        "action": torch.rand(1, 16, 6),
        "action_is_pad": torch.zeros(1, 16, dtype=torch.bool),
        "task": ["pick"],
    }
    tokenizer = ActionTokenizer(
        -5 * np.ones(6, dtype=np.float32),
        5 * np.ones(6, dtype=np.float32),
    )
    train_action_head(
        head,
        fake_backbone,
        [batch],
        fake_stats,
        tokenizer,
        "cpu",
        total_steps=1,
        warmup_steps=0,
        log_every=1,
        checkpoint_every=1,
        checkpoint_dir=tmp_path,
    )
    assert head.training
    assert not fake_backbone.training
    compact = head_only_state_dict(head)
    assert compact
    assert not any(name.startswith("backbone.") for name in compact)
    checkpoint = torch.load(tmp_path / "step000000.pt")
    assert "scheduler" in checkpoint
    assert not any(
        name.startswith("backbone.") for name in checkpoint["head"]
    )
    saved_slot = checkpoint["head"]["slot"].clone()
    with torch.no_grad():
        head.slot.add_(1.0)
    train_action_head(
        head,
        fake_backbone,
        [batch],
        fake_stats,
        tokenizer,
        "cpu",
        total_steps=1,
        warmup_steps=0,
        log_every=1,
        resume_from=tmp_path / "step000000.pt",
    )
    torch.testing.assert_close(head.slot, saved_slot)


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
