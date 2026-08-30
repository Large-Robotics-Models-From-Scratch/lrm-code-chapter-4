"""Small numerical and plotting diagnostics for section 4.6."""

from __future__ import annotations

import numpy as np

from ch04.style import (
    EXPERT_COLOR,
    NEUTRAL_COLOR,
    POLICY_COLOR,
    SUPPORTED_COLOR,
    UNSUPPORTED_COLOR,
    annotate_source,
    head_color,
    head_label,
    head_style,
)


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
        expert[:, 0], expert[:, 1], color=EXPERT_COLOR, s=26,
        alpha=0.85, label="expert support", zorder=3,
    )
    ax.scatter(
        samples[mask, 0], samples[mask, 1], color=SUPPORTED_COLOR, s=9,
        alpha=0.35, lw=0, label=f"supported ({int(mask.sum())})",
    )
    ax.scatter(
        samples[~mask, 0], samples[~mask, 1], color=UNSUPPORTED_COLOR,
        s=9, alpha=0.35, lw=0,
        label=f"unsupported ({int((~mask).sum())})",
    )
    ax.set(xlabel="first control bin", ylabel="second control bin")
    ax.legend(loc="upper right", markerscale=1.6)
    return ax.figure


def plot_action_distribution(probabilities, ax=None):
    """Plot probability over action bins and return the figure."""
    import matplotlib.pyplot as plt

    probs = np.asarray(probabilities)
    if probs.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    axis = np.arange(probs.size)
    ax.fill_between(axis, probs, color=SUPPORTED_COLOR, alpha=0.25, lw=0)
    ax.plot(axis, probs, color=SUPPORTED_COLOR, lw=1.4)
    peak = int(np.argmax(probs))
    ax.annotate(
        f"mode: bin {peak}",
        xy=(peak, probs[peak]),
        xytext=(6, -2),
        textcoords="offset points",
        fontsize=8,
        color="#6E6E76",
    )
    ax.set(
        xlabel=f"action bin (0-{probs.size - 1})",
        ylabel="predicted probability",
        xlim=(0, probs.size - 1),
        ylim=(0, None),
    )
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
        pred.shape[1], 1, figsize=(7.5, 1.25 * pred.shape[1]),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(
            target[:, index], label="expert", color=EXPERT_COLOR,
            lw=1.9, alpha=0.75, marker="o", ms=3,
        )
        axis.plot(
            pred[:, index], label="policy", color=POLICY_COLOR,
            lw=1.4, marker="s", ms=3,
        )
        axis.set_ylabel(names[index], fontsize=9)
    axes[0].legend(loc="upper left", ncols=2)
    axes[-1].set_xlabel("chunk timestep (0 = now)")
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
        color=NEUTRAL_COLOR,
        alpha=0.40,
        label="expert demonstrations",
    )
    ax.axvline(
        mse_prediction,
        color=POLICY_COLOR,
        lw=2.2,
        label=f"MSE prediction ({mse_prediction:+.3f})",
    )
    if mixture is not None:
        grid = torch.linspace(-grid_limit, grid_limit, n_grid)
        density = mixture.density(torch.zeros(1, 1), grid)
        ax.plot(
            grid.numpy(),
            density.numpy(),
            color=SUPPORTED_COLOR,
            lw=2.2,
            label="two-component Gaussian mixture",
        )
    ax.set(
        xlabel="action value",
        ylabel="probability density",
        title="Regression collapses between modes; a mixture does not",
    )
    ax.legend(loc="upper center")
    ax.figure.tight_layout()
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
        2, 1, figsize=(8, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.1]},
    )
    axes[0].hist(
        targets,
        bins=min(64, bin_count),
        range=(0, bin_count),
        color=EXPERT_COLOR,
        alpha=0.85,
    )
    axes[0].set_ylabel("expert frames")
    axes[0].set_title(
        f"Held-out action distribution near one state "
        f"({probs.shape[0]} nearest frames)"
    )

    axis = np.arange(probs.shape[1])
    shown = min(n_curves, probs.shape[0])
    for index, row in enumerate(probs[:shown]):
        axes[1].plot(
            axis,
            row,
            color=SUPPORTED_COLOR,
            alpha=0.45,
            lw=1.0,
            label="individual softmax" if index == 0 else None,
        )
    axes[1].plot(
        axis,
        probs.mean(axis=0),
        color=POLICY_COLOR,
        lw=2.2,
        label="cluster mean",
    )
    axes[1].set(
        xlabel=f"action bin (0-{bin_count - 1})",
        ylabel="predicted probability",
        xlim=(0, bin_count - 1),
    )
    axes[1].set_title(
        f"Policy marginals for the same cell ({shown} of "
        f"{probs.shape[0]} shown)"
    )
    axes[1].legend(loc="upper right")
    figure.tight_layout()
    if caption:
        annotate_source(figure, caption)
    return figure


def plot_joint_mismatch_panels(
    samples_by_head: dict,
    expert_pairs: np.ndarray,
    n_bins: int = 32,
    bin_range: tuple[int, int] | None = None,
    dim_labels: tuple[str, str] = ("A", "B"),
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
    edges = np.linspace(limits[0], limits[1], n_bins + 1)
    counts = {}
    for name in names:
        draws = np.asarray(samples_by_head[name])
        if draws.ndim != 2 or draws.shape[1] != 2:
            raise ValueError(f"{name} samples must have shape [N, 2]")
        counts[name] = np.histogram2d(
            draws[:, 0], draws[:, 1], bins=[edges, edges]
        )[0]
    # One scale across panels: independently normalized panels would make
    # a diffuse head look as concentrated as a sharp one.
    vmax = max(float(grid.max()) for grid in counts.values()) or 1.0

    figure, axes = plt.subplots(
        1, len(names), figsize=(4.1 * len(names) + 0.9, 4.3),
        squeeze=False, sharex=True, sharey=True,
    )
    mesh = None
    for axis, name in zip(axes[0], names, strict=True):
        mesh = axis.pcolormesh(
            edges, edges, counts[name].T,
            cmap="Blues", vmin=0.0, vmax=vmax, shading="flat",
        )
        axis.scatter(
            expert[:, 0], expert[:, 1],
            facecolor="none", edgecolor="#E8590C", linewidth=1.7,
            s=70, label="demonstrated pairs", zorder=4,
        )
        axis.set(
            xlabel=f"control {dim_labels[0]} bin",
            xlim=limits, ylim=limits,
        )
        axis.set_title(head_label(name), color=head_color(name))
        axis.set_aspect("equal")
        axis.grid(False)
    axes[0][0].set_ylabel(f"control {dim_labels[1]} bin")
    axes[0][-1].legend(loc="upper right", framealpha=0.95)
    figure.suptitle(
        "Joint mismatch: sampled control pairs against expert support",
        y=1.0,
    )
    figure.tight_layout()
    bar = figure.colorbar(
        mesh, ax=axes[0].tolist(), fraction=0.024, pad=0.015
    )
    bar.set_label("sampled draws per cell", fontsize=9)
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
        1, len(names), figsize=(4.1 * len(names), 3.2),
        squeeze=False, sharey=True,
    )
    for axis, name in zip(axes[0], names, strict=True):
        grids = np.asarray(samples_by_head[name])
        if grids.ndim != 3:
            raise ValueError(f"{name} samples must have shape [N, H, D]")
        if not 0 <= control < grids.shape[2]:
            raise IndexError("control is outside the action grid")
        colour = head_color(name)
        for grid in grids[:12]:
            axis.plot(grid[:, control], color=colour, alpha=0.45, lw=1.1)
        jitter = float(
            np.abs(np.diff(grids[:, :, control], axis=1)).mean()
        ) if grids.shape[1] > 1 else float("nan")
        axis.set(xlabel="chunk timestep")
        axis.set_title(
            f"{head_label(name)}\nmean step-to-step change: {jitter:.1f} "
            "bins",
            color=colour,
        )
    axes[0][0].set_ylabel(f"sampled bin, control {control}")
    figure.suptitle(
        "Independent per-cell sampling shows as temporal jitter", y=1.03
    )
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
    _, ax = plt.subplots(figsize=(8.5, 3.4))
    if expert is not None:
        reference = np.asarray(expert)
        if reference.ndim != 2:
            raise ValueError("expert must have shape [T, D]")
        ax.plot(
            reference[:, control],
            color=EXPERT_COLOR,
            lw=2.4,
            alpha=0.55,
            label="expert",
            zorder=1,
        )
    dashes = ["--", "-.", "-", ":"]
    for index, (name, trace) in enumerate(traces.items()):
        values = np.asarray(trace)
        if values.ndim != 2:
            raise ValueError(f"{name} trace must have shape [T, D]")
        if not 0 <= control < values.shape[1]:
            raise IndexError("control is outside the action vector")
        ax.plot(
            values[:, control],
            lw=1.6,
            ls=dashes[index % len(dashes)],
            label=name,
        )
    ax.set(
        xlabel="control timestep (30 Hz)",
        ylabel=f"control {control} command",
        title="Executing one chunk stream three ways",
    )
    ax.legend(ncols=2, loc="best")
    ax.figure.tight_layout()
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
        figsize=(9, 1.45 * pred.shape[1]),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(
            steps, target[:, index], label="expert", lw=1.9,
            color=EXPERT_COLOR, alpha=0.75,
        )
        axis.plot(
            steps, pred[:, index], label="policy", lw=1.4,
            color=POLICY_COLOR,
        )
        axis.fill_between(
            steps, target[:, index], pred[:, index],
            color=POLICY_COLOR, alpha=0.10, lw=0,
        )
        axis.set_ylabel(names[index], fontsize=9)
        error = float(np.abs(pred[:, index] - target[:, index]).mean())
        axis.annotate(
            f"MAE {error:.2f}",
            xy=(0.995, 0.86), xycoords="axes fraction",
            ha="right", fontsize=8, color="#6E6E76",
        )
    axes[0].legend(loc="upper left", ncols=2)
    axes[0].set_title("Open-loop chunk prediction on a held-out episode")
    axes[-1].set_xlabel("episode timestep (30 Hz)")
    figure.tight_layout()
    return figure


def plot_training_curves(histories: dict, log_scale: bool = False):
    """Training and held-out cross-entropy for one or more heads.

    ``histories`` maps a head name to the list :func:`train_action_head`
    returns. Held-out points are sparse by construction, so they are drawn
    as markers over the continuous training curve. The ``ln(B)`` reference
    line is the loss of a uniform policy: a curve that never leaves it has
    not started learning.
    """
    import matplotlib.pyplot as plt

    if not histories:
        raise ValueError("histories must contain at least one head")
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    uniform = None
    for name, history in histories.items():
        if not history:
            raise ValueError(f"{name} has an empty history")
        style = head_style(name)
        steps = [record["step"] for record in history]
        axes[0].plot(
            steps,
            [record["loss"] for record in history],
            color=style["color"],
            ls=style["linestyle"],
            label=f"{style['label']} (train)",
        )
        held_out = [
            (record["step"], record["validation_loss"])
            for record in history
            if not np.isnan(record.get("validation_loss", np.nan))
        ]
        if held_out:
            axes[0].plot(
                [point[0] for point in held_out],
                [point[1] for point in held_out],
                color=style["color"],
                marker=style["marker"],
                ls="none",
                ms=6,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=f"{style['label']} (held out)",
            )
        axes[1].plot(
            steps,
            [record["entropy"] for record in history],
            color=style["color"],
            ls=style["linestyle"],
            label=style["label"],
        )
        entropies = [record["entropy"] for record in history]
        uniform = max(uniform or 0.0, max(entropies))

    for axis, title, ylabel in (
        (axes[0], "Cross-entropy per action token", "nats / token"),
        (axes[1], "Predictive entropy per action token", "nats"),
    ):
        axis.set(xlabel="training step", ylabel=ylabel, title=title)
        if uniform:
            axis.axhline(
                np.log(256), color=NEUTRAL_COLOR, ls=":", lw=1.2,
                label="ln(256), uniform" if axis is axes[0] else None,
            )
    if log_scale:
        axes[0].set_yscale("log")
    # One legend under both panels: an in-axes legend with three heads and
    # their held-out markers covers the curves it is meant to explain.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncols=min(4, len(labels)),
        fontsize=8.5,
        frameon=False,
    )
    figure.tight_layout()
    return figure


def plot_head_comparison(summary: dict, metrics: dict | None = None):
    """A grouped bar chart of the final metric per head.

    ``summary`` maps a head name to a metric dictionary. ``metrics`` maps
    a metric key to the axis title; it defaults to the three the chapter
    argues about: fit, open-loop error, and sampled coherence.
    """
    import matplotlib.pyplot as plt

    if not summary:
        raise ValueError("summary must contain at least one head")
    metrics = metrics or {
        "validation_ce": "Held-out CE (nats / token)",
        "mae_std": "Open-loop MAE (training std)",
        "jitter": "Sampled jitter (bins / step)",
    }
    names = list(summary)
    figure, axes = plt.subplots(
        1, len(metrics), figsize=(4.0 * len(metrics), 3.4)
    )
    axes = np.atleast_1d(axes)
    for axis, (key, title) in zip(axes, metrics.items(), strict=True):
        missing = [name for name in names if key not in summary[name]]
        if missing:
            raise KeyError(f"{missing} have no {key!r} entry")
        values = [summary[name][key] for name in names]
        bars = axis.bar(
            [head_label(name).split(" (")[0] for name in names],
            values,
            color=[head_color(name) for name in names],
            width=0.62,
        )
        # Significant figures, not fixed decimals: a jitter of 41.2 bins
        # and a loss of 5.02 nats do not want the same precision.
        axis.bar_label(bars, fmt="%.3g", fontsize=8.5, padding=2)
        axis.set_title(title)
        axis.margins(y=0.18)
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="x", visible=False)
    figure.tight_layout()
    return figure
