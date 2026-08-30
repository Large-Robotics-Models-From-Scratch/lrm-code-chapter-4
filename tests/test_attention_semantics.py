"""Behavioral checks on masking, positions, and the AR KV cache.

Shape assertions cannot tell a correct attention mask from a broken one.
These tests perturb inputs the mask is supposed to hide and assert that
the action logits do or do not move accordingly.
"""

import pytest
import torch

from ch04 import AutoregressiveActionHead, ParallelDecodeActionHead


def _images():
    return torch.rand(2, 2, 3, 8, 8)


def test_padded_instruction_tokens_cannot_reach_action_logits(
    fake_backbone,
):
    torch.manual_seed(0)
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12).eval()
    images = _images()
    state = torch.rand(2, 6)
    text_valid = torch.tensor(
        [[True, False, False], [True, True, True]]
    )
    first = torch.tensor([[10, 0, 0], [10, 11, 12]])
    second = torch.tensor([[10, 77, 91], [10, 11, 12]])

    with torch.no_grad():
        left = head(images, first, state, text_valid)
        right = head(images, second, state, text_valid)

    # Row 0 hides positions 1 and 2, so its action logits must not move.
    torch.testing.assert_close(left[0], right[0])
    torch.testing.assert_close(left[1], right[1])


def test_action_slots_attend_bidirectionally(fake_backbone):
    """A later slot must be able to change an earlier slot's logits."""
    torch.manual_seed(0)
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12).eval()
    inputs = (
        _images(),
        torch.tensor([[10, 11], [10, 11]]),
        torch.rand(2, 6),
        torch.ones(2, 2, dtype=torch.bool),
    )
    with torch.no_grad():
        before = head(*inputs)
        head.slots[-1].add_(5.0)
        after = head(*inputs)
    assert not torch.allclose(before[:, 0], after[:, 0]), (
        "the first action slot did not see the last slot, so the "
        "action block is not bidirectional"
    )


def test_prefix_stays_causal_under_the_parallel_mask(fake_backbone):
    """No prefix position may attend to a later prefix position."""
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    mask = head._mask(torch.ones(1, 5, dtype=torch.bool), torch.float32)
    prefix = mask[0, 0, :5, :5]
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.isneginf(prefix[future]).all()
    assert torch.all(prefix[~future] == 0)
    # Action queries see the whole valid prefix and every other slot.
    assert torch.all(mask[0, 0, 5:, :] == 0)


def test_ar_cached_generation_matches_teacher_forcing(fake_backbone):
    """Greedy cached decoding must agree with a full teacher-forced pass.

    Teacher forcing at position ``i`` conditions on exactly the bins the
    greedy loop had already emitted, so re-scoring the generated grid has
    to reproduce it. A cache that misplaces positions or keys breaks this.
    """
    torch.manual_seed(0)
    head = AutoregressiveActionHead(
        fake_backbone, d_embed=12, horizon=3, action_dim=2
    ).eval()
    inputs = (
        _images(),
        torch.tensor([[10, 11, 0], [10, 11, 12]]),
        torch.rand(2, 6),
        torch.tensor([[True, True, False], [True, True, True]]),
    )
    with torch.no_grad():
        generated = head.generate(*inputs, temperature=0.0)
        rescored = head.teacher_forced_logits(*inputs, generated)
    torch.testing.assert_close(rescored.argmax(-1), generated)


def test_ar_generation_ignores_padded_instruction_tokens(fake_backbone):
    torch.manual_seed(0)
    head = AutoregressiveActionHead(
        fake_backbone, d_embed=12, horizon=2, action_dim=2
    ).eval()
    images, state = _images(), torch.rand(2, 6)
    valid = torch.tensor([[True, False], [True, True]])
    with torch.no_grad():
        left = head.generate(
            images, torch.tensor([[10, 0], [10, 11]]), state, valid
        )
        right = head.generate(
            images, torch.tensor([[10, 63], [10, 11]]), state, valid
        )
    torch.testing.assert_close(left[0], right[0])


def test_parallel_head_rejects_a_malformed_prefix_mask(fake_backbone):
    head = ParallelDecodeActionHead(fake_backbone, d_embed=12)
    with pytest.raises(ValueError, match=r"\[B, N\]"):
        head._mask(torch.ones(3, dtype=torch.bool), torch.float32)
