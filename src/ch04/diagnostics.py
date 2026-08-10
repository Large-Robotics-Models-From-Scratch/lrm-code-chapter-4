"""Small numerical and plotting diagnostics for section 4.6."""

from __future__ import annotations

import numpy as np


def joint_mismatch_rate(
    first: np.ndarray,
    second: np.ndarray,
    split_first: int,
    split_second: int,
) -> float:
    """Fraction in off-diagonal high/low quadrants."""
    x = np.asarray(first)
    y = np.asarray(second)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("joint samples must be non-empty and matching")
    x_high = x >= split_first
    y_high = y >= split_second
    return float(np.mean(x_high != y_high))


def sample_cell_pairs(
    logits,
    first_cell: int,
    second_cell: int,
    n_samples: int = 500,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a pair of action-grid marginals for coordination plots."""
    import torch

    values = torch.as_tensor(logits).float()
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2:
        raise ValueError("logits must have shape [G, bins] or [B, G, bins]")
    if (
        not 0 <= first_cell < values.shape[0]
        or not 0 <= second_cell < values.shape[0]
    ):
        raise IndexError("cell index is outside the action grid")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    probs = values[[first_cell, second_cell]].softmax(dim=-1)
    samples = torch.multinomial(
        probs,
        n_samples,
        replacement=True,
        generator=generator,
    )
    return samples[0].cpu().numpy(), samples[1].cpu().numpy()


def temporal_jitter(action_grid: np.ndarray) -> float:
    """Mean absolute first difference over a sampled ``[H, D]`` chunk."""
    values = np.asarray(action_grid)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("action_grid must have shape [H >= 2, D]")
    return float(np.abs(np.diff(values, axis=0)).mean())


def plot_joint_pairs(first, second, ax=None, title: str | None = None):
    """Plot sampled action-bin pairs and return the figure."""
    import matplotlib.pyplot as plt

    x = np.asarray(first)
    y = np.asarray(second)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("joint samples must be matching 1-D arrays")
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.hist2d(x, y, bins=32)
    ax.set(xlabel="first joint bin", ylabel="second joint bin")
    if title is not None:
        ax.set_title(title)
    return ax.figure


def plot_action_distribution(probabilities, ax=None):
    """Plot probability over action bins and return the figure."""
    import matplotlib.pyplot as plt

    probs = np.asarray(probabilities)
    if probs.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(probs.size), probs)
    ax.set(xlabel="bin id", ylabel="probability")
    return ax.figure


def plot_chunk_comparison(predicted, expert, joint_names=None):
    """Plot one predicted and expert ``[H, D]`` action chunk."""
    import matplotlib.pyplot as plt

    pred = np.asarray(predicted)
    target = np.asarray(expert)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("chunks must match and have shape [H, D]")
    names = joint_names or [f"joint {i}" for i in range(pred.shape[1])]
    fig, axes = plt.subplots(
        pred.shape[1], 1, figsize=(8, 2 * pred.shape[1]), sharex=True
    )
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(target[:, index], label="expert")
        axis.plot(pred[:, index], label="policy")
        axis.set_ylabel(names[index])
    axes[0].legend()
    axes[-1].set_xlabel("chunk timestep")
    fig.tight_layout()
    return fig
