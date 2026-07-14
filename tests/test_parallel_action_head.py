"""Tests for the parallel (one-shot categorical) action head.

This head is the manuscript's pedagogical foil (SS4.5.2): a two-layer
MLP that predicts every ``(timestep, joint)`` logit in a single
forward pass, independently. The tests below pin the shapes, the
zero-init sanity check (a fresh head must start at uniform entropy,
``log(n_bins)``), the adjacent-bin smoothing distribution, and the
sampling contract.
"""

import math

import torch

from ch04 import ACTION_DIM, CHUNK_H, N_BINS
from ch04.parallel_action_head import (
    ParallelActionHead,
    adjacent_bin_targets,
)

D_EMBED = 576
BATCH = 2


def test_forward_shape():
    head = ParallelActionHead()
    pooled = torch.randn(BATCH, D_EMBED)
    logits = head(pooled)
    assert logits.shape == (BATCH, CHUNK_H, ACTION_DIM, N_BINS)


def test_fresh_init_loss_is_near_uniform_entropy():
    """Zero-init final bias => softmax ~ uniform => CE ~ log(n_bins).

    This is the manuscript's SS4.5.2 sanity check: an untrained
    categorical head reports loss close to the entropy of a uniform
    distribution over ``n_bins`` bins, not some arbitrary number.
    """
    torch.manual_seed(0)
    head = ParallelActionHead()
    pooled = torch.randn(BATCH, D_EMBED)
    targets = torch.randint(0, N_BINS, (BATCH, CHUNK_H, ACTION_DIM))

    loss = head.loss(pooled, targets)

    expected = math.log(N_BINS)
    assert abs(loss.item() - expected) < 0.05


def test_adjacent_bin_targets_rows_sum_to_one():
    torch.manual_seed(0)
    target_bins = torch.randint(0, N_BINS, (BATCH, CHUNK_H, ACTION_DIM))
    eps = 0.05

    soft = adjacent_bin_targets(target_bins, N_BINS, eps)

    assert soft.shape == (BATCH, CHUNK_H, ACTION_DIM, N_BINS)
    sums = soft.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_adjacent_bin_targets_interior_mass_placement():
    """Interior target bin: (1-eps) on target, eps/2 on each neighbor."""
    eps = 0.05
    target_bins = torch.tensor([[[5]]])  # [B=1, H=1, D=1]

    soft = adjacent_bin_targets(target_bins, N_BINS, eps)
    row = soft[0, 0, 0]

    expected = torch.zeros(N_BINS)
    expected[5] = 1 - eps
    expected[4] = eps / 2
    expected[6] = eps / 2
    assert torch.allclose(row, expected, atol=1e-6)


def test_adjacent_bin_targets_edge_bin_mass_placement():
    """Edge bin (no left neighbor) puts the full eps on the one
    neighbor it has."""
    eps = 0.05
    target_bins = torch.tensor([[[0]]])  # [B=1, H=1, D=1]

    soft = adjacent_bin_targets(target_bins, N_BINS, eps)
    row = soft[0, 0, 0]

    expected = torch.zeros(N_BINS)
    expected[0] = 1 - eps
    expected[1] = eps
    assert torch.allclose(row, expected, atol=1e-6)

    # Last bin: no right neighbor, full eps on the left one.
    target_bins_last = torch.tensor([[[N_BINS - 1]]])
    soft_last = adjacent_bin_targets(target_bins_last, N_BINS, eps)
    row_last = soft_last[0, 0, 0]

    expected_last = torch.zeros(N_BINS)
    expected_last[N_BINS - 1] = 1 - eps
    expected_last[N_BINS - 2] = eps
    assert torch.allclose(row_last, expected_last, atol=1e-6)


def test_smoothed_loss_with_zero_eps_matches_plain_ce():
    """eps=0 is a consistency check: the soft-target formula, fed
    ``adjacent_bin_targets(..., eps=0)`` (all mass on the target bin),
    must numerically match ``head.loss``'s plain cross-entropy path."""
    torch.manual_seed(1)
    head = ParallelActionHead()
    pooled = torch.randn(BATCH, D_EMBED)
    targets = torch.randint(0, N_BINS, (BATCH, CHUNK_H, ACTION_DIM))

    plain = head.loss(pooled, targets, label_smoothing_eps=0.0)

    logits = head(pooled)
    soft = adjacent_bin_targets(targets, N_BINS, 0.0)
    log_probs = torch.log_softmax(logits, dim=-1)
    manual = -(soft * log_probs).sum(dim=-1).mean()

    assert torch.allclose(plain, manual, atol=1e-5)


def test_pad_mask_ignores_masked_timestep():
    """A masked (padded) timestep must not affect the loss.

    lerobot repeats the last action past an episode boundary and flags
    those steps in ``action_is_pad``; the head drops the whole
    ``[B, H, D]`` row for a padded timestep. Because this head scores
    every position independently (no teacher forcing), corrupting a
    masked timestep's targets must leave the loss unchanged, and differ
    from an unmasked run.
    """
    torch.manual_seed(0)
    head = ParallelActionHead()
    pooled = torch.randn(BATCH, D_EMBED)
    targets = torch.randint(0, N_BINS, (BATCH, CHUNK_H, ACTION_DIM))

    mask = torch.zeros(BATCH, CHUNK_H, dtype=torch.bool)
    mask[:, -1] = True  # last timestep is padding

    corrupted = targets.clone()
    corrupted[:, -1] = (targets[:, -1] + 1) % N_BINS

    masked_clean = head.loss(pooled, targets, pad_mask=mask)
    masked_corrupt = head.loss(pooled, corrupted, pad_mask=mask)
    assert torch.allclose(masked_clean, masked_corrupt, atol=1e-6)

    unmasked = head.loss(pooled, targets)
    assert not torch.allclose(masked_clean, unmasked)


def test_sample_shape_dtype_and_range():
    torch.manual_seed(0)
    head = ParallelActionHead()
    pooled = torch.randn(BATCH, D_EMBED)

    bins = head.sample(pooled)

    assert bins.shape == (BATCH, CHUNK_H, ACTION_DIM)
    assert bins.dtype == torch.int64
    assert bins.min() >= 0
    assert bins.max() < N_BINS
