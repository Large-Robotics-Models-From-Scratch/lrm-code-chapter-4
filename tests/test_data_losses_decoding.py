import math

import numpy as np
import pytest
import torch

from ch04 import ActionTokenizer
from ch04.data import (
    _padded_sequence_ids,
    action_targets,
    denormalize_from_stats,
    normalize_from_stats,
    prepare_batch,
)
from ch04.decoding import (
    mean_absolute_error_by_timestep,
    nucleus_probabilities,
)
from ch04.losses import expand_timestep_pad_mask, masked_token_cross_entropy


def test_normalization_round_trip(fake_stats):
    values = torch.randn(2, 16, 6)
    normalized = normalize_from_stats(values, fake_stats, "action")
    restored = denormalize_from_stats(normalized, fake_stats, "action")
    torch.testing.assert_close(restored, values)


def test_prepare_batch_shapes(fake_backbone, fake_stats):
    batch = {
        "observation.images.up": torch.rand(2, 3, 20, 30),
        "observation.images.side": torch.rand(2, 3, 20, 30),
        "observation.state": torch.rand(2, 6),
        "task": ["pick", "place"],
    }
    images, ids, state = prepare_batch(
        batch, fake_stats, fake_backbone, "cpu"
    )
    assert images.shape == (2, 2, 3, 224, 224)
    assert ids.ndim == 2 and ids.dtype == torch.int64
    assert state.shape == (2, 6) and state.dtype == torch.float32


def test_sequence_padding_follows_state_and_is_masked(fake_backbone):
    from ch04.backbone_adapter import prefix_valid_mask

    ids = _padded_sequence_ids(
        fake_backbone, [[10], [11, 12, 13]], max_text_tokens=3
    )
    assert ids[0].tolist() == [300, 300, 300, 300, 10, 301, 0, 0]
    assert ids[1].tolist() == [300, 300, 300, 300, 11, 12, 13, 301]
    valid = prefix_valid_mask(fake_backbone, ids)
    assert valid[0].tolist() == [True] * 6 + [False, False]
    assert valid[1].tolist() == [True] * 8


def test_prepare_batch_reuses_chapter3_antialiased_resize(
    fake_backbone, fake_stats
):
    from ch03 import preprocess_image

    up = torch.rand(1, 3, 480, 640)
    side = torch.rand(1, 3, 480, 640)
    batch = {
        "observation.images.up": up,
        "observation.images.side": side,
        "observation.state": torch.rand(1, 6),
        "task": ["pick"],
    }
    images, _, _ = prepare_batch(batch, fake_stats, fake_backbone, "cpu")
    torch.testing.assert_close(images[:, 0], preprocess_image(up))
    torch.testing.assert_close(images[:, 1], preprocess_image(side))


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
