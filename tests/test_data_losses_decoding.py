import math

import numpy as np
import pytest
import torch

from ch04 import ActionTokenizer
from ch04.data import (
    _compute_stats_from_frame_columns,
    action_targets,
    denormalize_from_stats,
    normalize_from_stats,
    prepare_batch,
)
from ch04.decoding import (
    mean_absolute_error_by_timestep,
    nucleus_probabilities,
    select_bins,
)
from ch04.losses import expand_timestep_pad_mask, masked_token_cross_entropy


def test_normalization_round_trip(fake_stats):
    values = torch.randn(2, 16, 6)
    normalized = normalize_from_stats(values, fake_stats, "action")
    restored = denormalize_from_stats(normalized, fake_stats, "action")
    torch.testing.assert_close(restored, values)


def test_training_stats_use_non_video_frame_columns():
    class Frames:
        hf_dataset = {
            "observation.state": [torch.zeros(6), torch.full((6,), 2.0)],
            "action": [torch.ones(6), torch.full((6,), 3.0)],
        }

    stats = _compute_stats_from_frame_columns(Frames())
    torch.testing.assert_close(
        stats["observation.state"]["mean"], torch.ones(6)
    )
    torch.testing.assert_close(
        stats["action"]["mean"], torch.full((6,), 2.0)
    )


def test_prepare_batch_shapes(fake_backbone, fake_stats):
    batch = {
        "observation.images.up": torch.rand(2, 3, 20, 30),
        "observation.images.side": torch.rand(2, 3, 20, 30),
        "observation.state": torch.rand(2, 6),
        "task": ["pick", "place"],
    }
    images, ids, state, text_valid = prepare_batch(
        batch, fake_stats, "cpu", fake_backbone
    )
    assert images.shape == (2, 2, 3, 20, 30)
    assert ids.ndim == 2 and ids.dtype == torch.int64
    assert state.shape == (2, 6) and state.dtype == torch.float32
    assert text_valid.shape == ids.shape and text_valid.dtype == torch.bool


def test_prepare_batch_tokenizes_and_masks_mixed_instructions(
    fake_backbone, fake_stats
):
    batch = {
        "observation.images.up": torch.rand(2, 3, 20, 30),
        "observation.images.side": torch.rand(2, 3, 20, 30),
        "observation.state": torch.rand(2, 6),
        "task": ["pick", "pick the object"],
    }
    _, ids, _, valid = prepare_batch(
        batch, fake_stats, "cpu", fake_backbone
    )
    assert ids.shape == valid.shape == (2, 3)
    assert valid[0].tolist() == [True, False, False]
    assert valid[1].tolist() == [True, True, True]


def test_prepare_batch_leaves_resize_to_chapter3(
    fake_backbone, fake_stats
):
    up = torch.rand(1, 3, 480, 640)
    side = torch.rand(1, 3, 480, 640)
    batch = {
        "observation.images.up": up,
        "observation.images.side": side,
        "observation.state": torch.rand(1, 6),
        "task": ["pick"],
    }
    images, _, _, _ = prepare_batch(
        batch, fake_stats, "cpu", fake_backbone
    )
    torch.testing.assert_close(images[:, 0], up)
    torch.testing.assert_close(images[:, 1], side)


def test_action_target_grid_order(fake_stats):
    tokenizer = ActionTokenizer(-np.ones(6), np.ones(6))
    actions = torch.zeros(2, 3, 6)
    actions[0, 0] = torch.arange(6) * 0.1
    pad = torch.tensor([[False, True, False], [False, False, False]])
    batch = {"action": actions, "action_is_pad": pad}
    bins, token_pad = action_targets(batch, fake_stats, tokenizer, "cpu")
    assert bins.shape == (2, 18)
    assert token_pad.shape == (2, 18)
    assert token_pad[0].tolist() == [False] * 6 + [True] * 6 + [False] * 6


def test_masked_cross_entropy_and_all_pad_error():
    logits = torch.zeros(1, 3, 256)
    targets = torch.tensor([[0, 1, 2]])
    pad = torch.tensor([[False, True, True]])
    loss = masked_token_cross_entropy(logits, targets, pad, 0.05)
    assert float(loss) == pytest.approx(math.log(256), rel=1e-5)
    with pytest.raises(ValueError):
        masked_token_cross_entropy(logits, targets, torch.ones_like(pad))


def test_pad_mask_expansion():
    pad = torch.tensor([[False, True]])
    expanded = expand_timestep_pad_mask(pad, 3)
    assert expanded.tolist() == [[False, False, False, True, True, True]]


def test_top_p_keeps_threshold_crossing_token():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    filtered = nucleus_probabilities(logits, top_p=0.6)
    assert filtered[0, 0] > 0
    assert filtered[0, 1] > 0
    assert filtered[0, 2] == 0
    assert filtered.sum().item() == pytest.approx(1.0)


def test_select_bins_defaults_to_argmax():
    logits = torch.tensor([[[0.0, 2.0, 1.0]]])
    assert select_bins(logits).item() == 1
    with pytest.raises(ValueError, match="unknown strategy"):
        select_bins(logits, strategy="median")


def test_timestep_mae_excludes_padding():
    expert = torch.zeros(2, 2, 1)
    predicted = torch.tensor([[[1.0], [100.0]], [[3.0], [5.0]]])
    pad = torch.tensor([[False, True], [False, False]])
    mae = mean_absolute_error_by_timestep(predicted, expert, pad)
    torch.testing.assert_close(mae, torch.tensor([2.0, 5.0]))


def test_diagnostics_cover_pairs_and_temporal_jitter():
    from ch04.diagnostics import sample_cell_pairs, temporal_jitter

    logits = torch.zeros(1, 96, 256)
    first, second = sample_cell_pairs(logits, 4, 5, n_samples=20)
    assert first.shape == second.shape == (20,)
    grid = np.arange(16 * 6).reshape(16, 6)
    assert temporal_jitter(grid) == pytest.approx(6.0)
