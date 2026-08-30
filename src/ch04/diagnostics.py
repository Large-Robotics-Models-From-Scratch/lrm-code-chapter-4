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
    if values.ndim == 4:
        values = values[0]
    if values.ndim == 3:
        values = values.flatten(0, 1)
    if values.ndim != 2:
        raise ValueError(
            "logits must be [H, D, bins], [B, H, D, bins], or [G, bins]"
        )
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


def within_expert_support(
    draws: np.ndarray,
    expert_pairs: np.ndarray,
    support_radius: float,
) -> np.ndarray:
    """Mark sampled pairs that lie near the held-out expert support."""
    samples = np.asarray(draws, dtype=np.float32)
    expert = np.asarray(expert_pairs, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("draws must have shape [N, 2]")
    if expert.ndim != 2 or expert.shape[1] != 2 or expert.shape[0] == 0:
        raise ValueError("expert_pairs must be non-empty with shape [M, 2]")
    if support_radius <= 0:
        raise ValueError("support_radius must be positive")
    squared_distance = np.square(samples[:, None] - expert[None]).sum(
        axis=-1
    )
    return squared_distance.min(axis=1) <= support_radius**2


def plot_joint_support(
    expert_pairs: np.ndarray,
    draws: np.ndarray,
    supported: np.ndarray,
    support_radius: float,
    ax=None,
):
    """Overlay expert support and color draws by support membership."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    expert = np.asarray(expert_pairs)
    samples = np.asarray(draws)
    mask = np.asarray(supported, dtype=bool)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("draws must have shape [N, 2]")
    if expert.ndim != 2 or expert.shape[1] != 2 or expert.shape[0] == 0:
        raise ValueError("expert_pairs must be non-empty with shape [M, 2]")
    if mask.shape != (samples.shape[0],):
        raise ValueError("supported must contain one flag per draw")
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    for point in expert:
        ax.add_patch(
            Circle(
                point,
                support_radius,
                color="tab:green",
                alpha=0.08,
                lw=0,
            )
        )
    ax.scatter(
        expert[:, 0], expert[:, 1], color="tab:green", s=12, alpha=0.65,
        label="expert support", zorder=3,
    )
    ax.scatter(
        samples[mask, 0], samples[mask, 1], color="tab:blue", s=7,
        alpha=0.3, label="supported samples",
    )
    ax.scatter(
        samples[~mask, 0], samples[~mask, 1], color="tab:red", s=7,
        alpha=0.3, label="unsupported samples",
    )
    ax.set(xlabel="first control bin", ylabel="second control bin")
    ax.legend()
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


def plot_action_diagnostics(
    logits,
    expert_pairs: np.ndarray,
    example: int,
    timestep: int,
    dims: tuple[int, int],
    support_radius: float,
    n_samples: int = 4000,
    seed: int = 7,
):
    """Plot one marginal and compare draws with expert support.

    This is the runnable entry point used by manuscript listing 4.9.
    ``dims`` selects two control dimensions at the same future timestep, and
    ``expert_pairs`` supplies encoded pairs from the held-out slice.
    """
    import torch

    values = torch.as_tensor(logits).float()
    if values.ndim != 4:
        raise ValueError("logits must have shape [B, H, D, bins]")
    if not 0 <= example < values.shape[0]:
        raise IndexError("example is outside the batch")
    if not 0 <= timestep < values.shape[1]:
        raise IndexError("timestep is outside the action horizon")
    first_dim, second_dim = dims
    if (
        not 0 <= first_dim < values.shape[2]
        or not 0 <= second_dim < values.shape[2]
    ):
        raise IndexError("control dimension is outside the action grid")

    probabilities = values[example, timestep].softmax(dim=-1)
    marginal_figure = plot_action_distribution(
        probabilities[first_dim].detach().cpu().numpy()
    )

    generator = torch.Generator(device=values.device).manual_seed(seed)
    draws = torch.multinomial(
        probabilities[[first_dim, second_dim]],
        n_samples,
        replacement=True,
        generator=generator,
    ).T.cpu().numpy()
    supported = within_expert_support(draws, expert_pairs, support_radius)
    joint_figure = plot_joint_support(
        expert_pairs,
        draws,
        supported,
        support_radius,
    )
    return marginal_figure, joint_figure


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
