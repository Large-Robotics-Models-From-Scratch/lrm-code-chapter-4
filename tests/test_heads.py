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
    assert logits.shape == (3, 16, 6, 256)
    assert logits.dtype == torch.float32
    assert torch.count_nonzero(head.action_decoder.bias) == 0


def test_parallel_head_mask_and_shape(fake_backbone, model_inputs):
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    valid = torch.ones(1, 7, dtype=torch.bool)
    mask = head._mask(valid, torch.float32)
    assert mask.shape == (1, 1, 23, 23)
    assert torch.isneginf(mask[0, 0, 0, 1])
    assert torch.all(mask[0, 0, 7:, 7:] == 0)
    logits = head(*model_inputs)
    assert logits.shape == (2, 16, 6, 256)
    assert logits.dtype == torch.float32
    logits.mean().backward()
    assert head.slots.grad is not None


def test_parallel_mask_excludes_padded_prefix_keys(fake_backbone):
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    valid = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    mask = head._mask(valid, torch.float32)
    action_query = 4
    assert torch.isneginf(mask[0, 0, action_query, 3])
    assert mask[1, 0, action_query, 3] == 0


def test_parallel_positions_ignore_text_padding(fake_backbone):
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    images = torch.rand(2, 2, 3, 8, 8)
    ids = torch.tensor([[10, 0, 0], [10, 11, 12]])
    text_valid = torch.tensor(
        [[True, False, False], [True, True, True]]
    )
    head(images, ids, torch.rand(2, 6), text_valid)
    positions = fake_backbone.language_backbone.last_position_ids
    assert positions[0, :8].tolist() == [0, 1, 2, 3, 4, 0, 0, 5]
    assert positions[1, :8].tolist() == list(range(8))
    assert positions[0, 8] == 6
    assert positions[1, 8] == 8


def test_ar_teacher_forcing_shift(fake_backbone, model_inputs):
    head = AutoregressiveActionHead(
        fake_backbone,
        d_embed=12,
        horizon=2,
        action_dim=5,
    )
    targets = torch.arange(10).repeat(2, 1)
    captured = []
    handle = head.action_embeddings.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        logits = head.teacher_forced_logits(*model_inputs, targets)
    finally:
        handle.remove()
    assert logits.shape == (2, 2, 5, 256)
    assert torch.equal(captured[-1], targets[:, :-1])


def test_ar_masked_loss_ignores_padded_targets(fake_backbone, model_inputs):
    torch.manual_seed(2)
    head = AutoregressiveActionHead(
        fake_backbone,
        d_embed=12,
        horizon=2,
        action_dim=4,
    )
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


def test_ar_generate_uses_cache_and_encodes_vision_once(
    fake_backbone, model_inputs
):
    head = AutoregressiveActionHead(
        fake_backbone,
        d_embed=12,
        horizon=1,
        action_dim=4,
    )
    captured = []
    handle = head.action_embeddings.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        bins = head.generate(*model_inputs)
    finally:
        handle.remove()
    assert bins.shape == (2, 1, 4)
    assert fake_backbone.vision_encoder.calls == 1
    assert [ids.shape for ids in captured] == [(2, 1)] * 3


def test_ar_uses_separate_action_embedding_table(fake_backbone):
    head = AutoregressiveActionHead(
        fake_backbone, d_embed=12, n_bins=128
    )
    assert head.action_embeddings.num_embeddings == 128
    assert (
        head.action_embeddings
        is not fake_backbone.language_backbone.get_input_embeddings()
    )
