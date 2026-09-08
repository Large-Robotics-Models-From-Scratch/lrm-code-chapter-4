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


# --- section 4.4.3: SmolVLA-style temporal tokenization -------------------


def _prefix_length(head, model_inputs):
    return head.backbone.embed_inputs(*model_inputs)[0].shape[1]


def test_ar_appends_one_position_per_timestep(fake_backbone, model_inputs):
    """The AR suffix is H positions long, not H * D.

    The parallel head already appends one slot per future timestep. An AR
    head that appended one token per (timestep, control) cell would carry a
    six-fold longer suffix, so every latency and FLOP comparison in section
    4.6 would confound decoding order with sequence length.
    """
    head = AutoregressiveActionHead(
        fake_backbone, d_embed=12, horizon=4, action_dim=3
    )
    targets = torch.randint(0, 256, (2, 4, 3))
    logits = head.teacher_forced_logits(*model_inputs, targets)

    assert logits.shape == (2, 4, 3, 256)
    positions = fake_backbone.language_backbone.last_position_ids
    prefix = _prefix_length(head, model_inputs)
    assert positions.shape[1] == prefix + head.horizon - 1


def test_ar_generation_costs_one_serial_step_per_timestep(
    fake_backbone, model_inputs
):
    """Serial depth is H. Section 4.6 prints this number."""
    head = AutoregressiveActionHead(
        fake_backbone, d_embed=12, horizon=4, action_dim=3
    ).eval()
    calls = []
    handle = fake_backbone.language_backbone.register_forward_pre_hook(
        lambda *_: calls.append(1)
    )
    try:
        with torch.no_grad():
            generated = head.generate(*model_inputs, temperature=0.0)
    finally:
        handle.remove()

    assert generated.shape == (2, 4, 3)
    assert int(generated.min()) >= 0 and int(generated.max()) < 256
    # One cached prefill plus H-1 incremental passes: H serial steps.
    assert len(calls) == head.horizon


def test_ar_conditions_on_earlier_timesteps_only(
    fake_backbone, model_inputs
):
    """Temporal causality: timestep t sees earlier ones, nothing later."""
    torch.manual_seed(0)
    head = AutoregressiveActionHead(
        fake_backbone, d_embed=12, horizon=4, action_dim=3
    ).eval()
    base = torch.randint(0, 256, (2, 4, 3))
    later = base.clone()
    later[:, 2:] = (later[:, 2:] + 7) % 256

    with torch.no_grad():
        left = head.teacher_forced_logits(*model_inputs, base)
        right = head.teacher_forced_logits(*model_inputs, later)

    # Timesteps 0..2 are decided before timestep 2's bins are ever read.
    torch.testing.assert_close(left[:, :3], right[:, :3])
    # Timestep 3 consumes timestep 2, so it must move.
    assert not torch.allclose(left[:, 3], right[:, 3])


def test_ar_controls_within_a_timestep_are_independent(
    fake_backbone, model_inputs
):
    """The accepted cost of SmolVLA granularity, pinned deliberately.

    All D controls of a timestep are read from one hidden state, so no
    control conditions on another chosen at the same timestep. This is a
    real modeling loss relative to the 96-token head -- section 4.4.3 and
    figure 4.9 must argue on the temporal axis instead -- and it is
    asserted here so nobody later mistakes it for a bug.
    """
    torch.manual_seed(0)
    head = AutoregressiveActionHead(
        fake_backbone, d_embed=12, horizon=4, action_dim=3
    ).eval()
    base = torch.randint(0, 256, (2, 4, 3))
    nudged = base.clone()
    nudged[:, 1, 0] = (nudged[:, 1, 0] + 11) % 256

    with torch.no_grad():
        left = head.teacher_forced_logits(*model_inputs, base)
        right = head.teacher_forced_logits(*model_inputs, nudged)

    # Control 0's own bin does not reach its siblings at timestep 1.
    torch.testing.assert_close(left[:, 1], right[:, 1])
    # It does reach the next timestep, where AR still earns its keep.
    assert not torch.allclose(left[:, 2], right[:, 2])
