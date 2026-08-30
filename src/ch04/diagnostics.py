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


def plot_bimodal_comparison(
    actions,
    mse_prediction: float,
    mixture=None,
    grid_limit: float = 2.0,
    n_grid: int = 401,
    ax=None,
):
    """Figure 4.4: an MSE point estimate against a fitted mixture.

    ``actions`` are the demonstrated one-dimensional targets, and
    ``mixture`` is an optional
    :class:`~ch04.exercises.MixtureDensityNetwork` whose density is
    overlaid on the same axis.
    """
    import matplotlib.pyplot as plt
    import torch

    values = np.asarray(actions).reshape(-1)
    if values.size == 0:
        raise ValueError("actions must be non-empty")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    ax.hist(
        values,
        bins=60,
        density=True,
        color="tab:gray",
        alpha=0.45,
        label="expert actions",
    )
    ax.axvline(
        mse_prediction,
        color="crimson",
        lw=2,
        label=f"MSE prediction ({mse_prediction:+.3f})",
    )
    if mixture is not None:
        grid = torch.linspace(-grid_limit, grid_limit, n_grid)
        density = mixture.density(torch.zeros(1, 1), grid)
        ax.plot(
            grid.numpy(),
            density.numpy(),
            color="tab:blue",
            lw=2,
            label="Gaussian mixture",
        )
    ax.set(xlabel="action", ylabel="density")
    ax.legend()
    return ax.figure


def nearest_state_neighbors(
    states: np.ndarray,
    anchor_index: int,
    n_neighbors: int = 32,
) -> np.ndarray:
    """Return the ``n_neighbors`` rows closest to one normalized state.

    The anchor itself is the nearest neighbour and is always first, so a
    caller asking for 32 neighbours receives the anchor plus 31 others.
    """
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("states must have shape [N, D]")
    if not 0 <= anchor_index < values.shape[0]:
        raise IndexError("anchor_index is outside the state array")
    if not 1 <= n_neighbors <= values.shape[0]:
        raise ValueError("n_neighbors must lie in [1, len(states)]")
    distance = np.square(values - values[anchor_index]).sum(axis=1)
    order = np.argsort(distance, kind="stable")
    return order[:n_neighbors]


def plot_neighbor_softmaxes(
    probabilities: np.ndarray,
    target_bins: np.ndarray,
    n_curves: int = 6,
    n_bins: int | None = None,
    caption: str | None = None,
):
    """Figure 4.8: held-out softmaxes around one proprioceptive anchor.

    The upper panel histograms the demonstrated bins of the neighbourhood.
    The lower panel overlays ``n_curves`` individual softmaxes with the
    cluster mean, so a bimodal individual curve can be told apart from a
    bimodal average over slightly different inputs.
    """
    import matplotlib.pyplot as plt

    probs = np.asarray(probabilities, dtype=np.float32)
    targets = np.asarray(target_bins).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape [N, bins]")
    if targets.shape[0] != probs.shape[0]:
        raise ValueError("one target bin is required per neighbour")
    if n_curves < 1:
        raise ValueError("n_curves must be positive")
    bin_count = n_bins or probs.shape[1]

    figure, axes = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    axes[0].hist(
        targets,
        bins=min(64, bin_count),
        range=(0, bin_count),
        color="tab:gray",
    )
    axes[0].set_ylabel("expert frames")
    # Provenance can be long; wrap it rather than stretching the figure.
    axes[0].set_title(
        caption or "held-out neighbourhood", fontsize=8, wrap=True
    )

    axis = np.arange(probs.shape[1])
    for row in probs[: min(n_curves, probs.shape[0])]:
        axes[1].plot(axis, row, color="tab:blue", alpha=0.45, lw=1)
    axes[1].plot(
        axis,
        probs.mean(axis=0),
        color="crimson",
        lw=2,
        label="cluster mean",
    )
    axes[1].set(xlabel="bin id", ylabel="probability")
    axes[1].legend()
    figure.tight_layout()
    return figure


def plot_joint_mismatch_panels(
    samples_by_head: dict,
    expert_pairs: np.ndarray,
    n_bins: int = 32,
    bin_range: tuple[int, int] | None = None,
):
    """Figure 4.9: one sampled ``(control, control)`` panel per head.

    ``samples_by_head`` maps a head name to an ``[N, 2]`` array of sampled
    bin pairs. Each panel overlays the demonstrated expert pairs so the
    off-diagonal mass that marks joint mismatch is visible directly.
    """
    import matplotlib.pyplot as plt

    if not samples_by_head:
        raise ValueError("samples_by_head must contain at least one head")
    expert = np.asarray(expert_pairs)
    if expert.ndim != 2 or expert.shape[1] != 2 or expert.shape[0] == 0:
        raise ValueError("expert_pairs must be non-empty with shape [M, 2]")
    limits = bin_range or (0, 256)

    names = list(samples_by_head)
    figure, axes = plt.subplots(
        1, len(names), figsize=(4 * len(names), 4), squeeze=False
    )
    for axis, name in zip(axes[0], names, strict=True):
        draws = np.asarray(samples_by_head[name])
        if draws.ndim != 2 or draws.shape[1] != 2:
            raise ValueError(f"{name} samples must have shape [N, 2]")
        axis.hist2d(
            draws[:, 0],
            draws[:, 1],
            bins=n_bins,
            range=[limits, limits],
            cmap="Blues",
        )
        axis.scatter(
            expert[:, 0],
            expert[:, 1],
            color="tab:green",
            s=18,
            label="expert pairs",
        )
        axis.set(xlabel="first control bin", title=name)
    axes[0][0].set_ylabel("second control bin")
    axes[0][-1].legend(loc="upper right")
    figure.tight_layout()
    return figure


def plot_temporal_traces(samples_by_head: dict, control: int = 0):
    """Section 4.6.2's temporal check: sampled bins across a chunk.

    ``samples_by_head`` maps a head name to ``[N, H, D]`` sampled grids.
    A factorized head draws each timestep independently and shows visible
    jitter; the conditioned heads produce smoother traces.
    """
    import matplotlib.pyplot as plt

    if not samples_by_head:
        raise ValueError("samples_by_head must contain at least one head")
    names = list(samples_by_head)
    figure, axes = plt.subplots(
        1, len(names), figsize=(4 * len(names), 3),
        squeeze=False, sharey=True,
    )
    for axis, name in zip(axes[0], names, strict=True):
        grids = np.asarray(samples_by_head[name])
        if grids.ndim != 3:
            raise ValueError(f"{name} samples must have shape [N, H, D]")
        if not 0 <= control < grids.shape[2]:
            raise IndexError("control is outside the action grid")
        for grid in grids[:12]:
            axis.plot(grid[:, control], alpha=0.4, lw=1)
        axis.set(xlabel="chunk timestep", title=name)
    axes[0][0].set_ylabel(f"sampled bin (control {control})")
    figure.tight_layout()
    return figure


def plot_execution_schedules(
    traces: dict,
    expert=None,
    control: int = 0,
):
    """Figure 4.10: the section 4.7.2 schedules against the clock.

    ``traces`` maps a schedule name to a ``[T, D]`` control trace, as
    returned by :func:`ch04.execution.execution_schedules`.
    """
    import matplotlib.pyplot as plt

    if not traces:
        raise ValueError("traces must contain at least one schedule")
    _, ax = plt.subplots(figsize=(8, 3))
    if expert is not None:
        reference = np.asarray(expert)
        if reference.ndim != 2:
            raise ValueError("expert must have shape [T, D]")
        ax.plot(
            reference[:, control],
            color="black",
            lw=2,
            alpha=0.6,
            label="expert",
        )
    for name, trace in traces.items():
        values = np.asarray(trace)
        if values.ndim != 2:
            raise ValueError(f"{name} trace must have shape [T, D]")
        if not 0 <= control < values.shape[1]:
            raise IndexError("control is outside the action vector")
        ax.plot(values[:, control], lw=1.5, label=name)
    ax.set(xlabel="control timestep", ylabel=f"control {control}")
    ax.legend()
    return ax.figure


def plot_open_loop_episode(
    predicted,
    expert,
    valid=None,
    joint_names=None,
):
    """Figure 4.11: a held-out episode trace, one panel per control.

    ``predicted`` and ``expert`` are ``[T, D]`` trajectories in the
    dataset's raw action units. ``valid`` optionally masks padded frames.
    """
    import matplotlib.pyplot as plt

    pred = np.asarray(predicted)
    target = np.asarray(expert)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("trajectories must match and have shape [T, D]")
    steps = np.arange(pred.shape[0])
    if valid is not None:
        keep = np.asarray(valid, dtype=bool).reshape(-1)
        if keep.shape[0] != pred.shape[0]:
            raise ValueError("valid must contain one flag per timestep")
        steps, pred, target = steps[keep], pred[keep], target[keep]
    if pred.shape[0] == 0:
        raise ValueError("no valid timesteps remain after masking")

    names = joint_names or [f"joint {i}" for i in range(pred.shape[1])]
    figure, axes = plt.subplots(
        pred.shape[1], 1,
        figsize=(9, 1.6 * pred.shape[1]),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(steps, target[:, index], label="expert", lw=1.5)
        axis.plot(
            steps, pred[:, index], label="policy", lw=1.2, alpha=0.85
        )
        axis.set_ylabel(names[index])
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("episode timestep")
    figure.tight_layout()
    return figure
