import inspect
import math

import torch

from ch04 import (
    AutoregressiveActionHead,
    FactorizedActionHead,
    ParallelDecodeActionHead,
)


def test_factorized_head_shape_and_float32():
    head = FactorizedActionHead(d_embed=12)
    logits = head(torch.rand(3, 12))
    assert logits.shape == (3, 96, 256)
    assert logits.dtype == torch.float32
    assert torch.count_nonzero(head.action_decoder.bias) == 0


def test_parallel_head_mask_and_shape(fake_backbone, model_inputs):
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    mask = head._mask(7, torch.device("cpu"))
    assert mask.shape == (1, 1, 103, 103)
    assert torch.isneginf(mask[0, 0, 0, 1])
    assert torch.all(mask[0, 0, 7:, 7:] == 0)
    logits = head(*model_inputs)
    assert logits.shape == (2, 96, 256)
    assert logits.dtype == torch.float32
    logits.mean().backward()
    assert head.slot.grad is not None


def test_parallel_mask_excludes_padded_prefix_keys(fake_backbone):
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    valid = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    mask = head._mask(4, torch.device("cpu"), prefix_valid=valid)
    action_query = 4
    assert torch.isneginf(mask[0, 0, action_query, 3])
    assert mask[1, 0, action_query, 3] == 0


def test_parallel_positions_ignore_batch_padding(fake_backbone):
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    images = torch.rand(2, 2, 3, 8, 8)
    ids = torch.tensor(
        [
            [300, 300, 300, 300, 10, 301, 0, 0],
            [300, 300, 300, 300, 10, 11, 12, 301],
        ]
    )
    head(images, ids, torch.rand(2, 6))
    positions = fake_backbone.language_backbone.last_position_ids
    assert positions[0, :8].tolist() == [0, 1, 2, 3, 4, 5, 5, 5]
    assert positions[1, :8].tolist() == list(range(8))
    assert positions[0, 8] == 6
    assert positions[1, 8] == 8


def test_ar_teacher_forcing_shift(fake_backbone, model_inputs):
    head = AutoregressiveActionHead(fake_backbone, d_embed=12)
    targets = torch.arange(10).repeat(2, 1)
    captured = []
    handle = fake_backbone.embed_tokens.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        logits = head.teacher_forced_logits(*model_inputs, targets)
    finally:
        handle.remove()
    seen = captured[-1]
    prefix = model_inputs[1].shape[1]
    assert logits.shape == (2, 10, 256)
    assert seen.shape[1] == prefix + 9
    assert torch.equal(seen[:, prefix:], 48896 + targets[:, :-1])


def test_ar_masked_loss_ignores_padded_targets(fake_backbone, model_inputs):
    torch.manual_seed(2)
    head = AutoregressiveActionHead(fake_backbone, d_embed=12)
    targets = torch.randint(0, 256, (2, 8))
    pad = torch.zeros_like(targets, dtype=torch.bool)
    pad[:, -2:] = True
    loss = head.loss(*model_inputs, targets, pad)
    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert 0 < loss < 2 * math.log(256)


def test_ar_loss_default_matches_manuscript():
    default = inspect.signature(
        AutoregressiveActionHead.loss
    ).parameters["label_smoothing"].default
    assert default == 0.05


def test_ar_decode_conditions_on_previous_bins(fake_backbone, model_inputs):
    head = AutoregressiveActionHead(fake_backbone, d_embed=12)
    captured = []
    handle = fake_backbone.embed_tokens.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        bins = head.decode(*model_inputs, grid_size=4, greedy=True)
    finally:
        handle.remove()
    assert bins.shape == (2, 4)
    assert fake_backbone.vision_encoder.calls == 1
    lengths = [ids.shape[1] for ids in captured]
    start = model_inputs[1].shape[1]
    assert lengths == [start, 1, 2, 3]


def test_ar_custom_bin_count_uses_matching_native_tail(fake_backbone):
    head = AutoregressiveActionHead(fake_backbone, d_embed=12, n_bins=128)
    assert head.token_base == 49_024
