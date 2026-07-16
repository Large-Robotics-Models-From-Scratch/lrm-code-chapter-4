"""Diagnostics dashboard and figure generators (SS4.6.5-4.6.7).

The numeric diagnostics are the cheap logs SS4.6.7 says to keep on every
run: per-dimension softmax entropy (falls from ``log(256)`` toward 2-3
nats as the head sharpens), the training-set bin-frequency histogram
(a modal bin over 20% of tokens is the focal-loss trigger), and full
softmax snapshots on canary states (the raw material for Figure 4.8).

``plot_action_traces`` renders the open-loop qualitative eval:
predicted vs ground-truth per-joint action traces on a held-out episode
window (solid expert, dashed policy), the honest sim-free companion to
``rollout.evaluate_open_loop``. ``open_loop_episode_predictions`` builds
its ``(pred, true)`` arrays by decoding one chunk from a held-out
observation.

The three plotting functions render the chapter's evidence figures:

* ``plot_convergence_ridges`` -- Figure 4.8: a categorical head learning
  to be bimodal, one softmax ridge per saved checkpoint, uniform at step
  0, two clean peaks by convergence.
* ``plot_bimodal_comparison`` -- Figure 4.9: the MSE baseline's single
  collapsed prediction against the categorical head's two-peaked
  distribution on the same base-rotation joint.
* ``plot_joint_coordination`` -- Figure 4.10: 2D histograms of sampled
  gripper x wrist bins, parallel (off-diagonal leakage) vs autoregressive
  (samples on the demonstrated diagonal).

Every figure follows FIGURE_STYLE_GUIDE: grayscale-safe (gray fills,
hatching, and line styles carry all distinctions -- never color alone),
300 dpi, within the 5.6 x 7 inch page box, and each returns its output
path for testability. The Agg backend is forced so the figures render
headless (no display needed).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_EPS = 1e-12
_DPI = 300


# -- numeric diagnostics ---------------------------------------------


def softmax_entropy(logits):
    """Base-e softmax entropy over the last (bin) axis.

    ``logits`` is ``[.., n_bins]`` (torch tensor or NumPy array); returns
    ``[..]`` with the bin axis reduced. Entropy is
    ``-sum_b p_b log p_b`` in nats, so a uniform softmax returns
    ``log(n_bins)`` (maximum entropy, step 0) and a one-hot returns ~0
    (fully committed). SS4.6.7 tracks this per dimension: broad means
    underfitting, narrow on a known-multimodal state means mode collapse.
    Returns the same array type it was given (torch in, torch out).
    """
    if isinstance(logits, torch.Tensor):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return -(probs * np.log(probs + _EPS)).sum(axis=-1)


def bin_frequency_histogram(loader, tokenizer, n_batches=None):
    """Count tokenized action bins across a data loader -> ``[n_bins]``.

    Iterates the loader, tokenizes each batch's ``action`` chunk
    (``[B, H, D]``) with ``tokenizer.encode``, and accumulates a global
    histogram over the ``n_bins`` shared action bins (SS4.6.7's
    modal-bin check: a bin over 20% of all tokens says the data is
    imbalanced enough to warrant focal loss). ``n_batches`` (optional)
    caps how many batches are read, for a quick estimate on a large set.

    Bins are counted across *all* dimensions together, matching the
    chapter's shared-vocabulary convention (one set of 256 ids reused by
    every joint). Returns an ``int64`` NumPy array summing to the number
    of action tokens seen.
    """
    counts = np.zeros(tokenizer.n_bins, dtype=np.int64)
    for i, batch in enumerate(loader):
        if n_batches is not None and i >= n_batches:
            break
        actions = np.asarray(batch["action"])
        bins = tokenizer.encode(actions)  # [B, H, D]
        flat = bins.reshape(-1)
        counts += np.bincount(flat, minlength=tokenizer.n_bins)
    return counts


def canary_snapshot(head, fusion, batch, target_bins):
    """Full softmax on a fixed canary state -> ``[T, n_bins]``.

    Runs the autoregressive head's teacher-forced ``logits`` on one
    canary observation (``batch`` -> ``fusion.encode_prefix``) and
    softmaxes over bins, returning the per-position distributions for a
    single-env batch (``B = 1``). SS4.6.7 saves these each checkpoint to
    watch a chosen ``(timestep, joint)`` position evolve from uniform to
    bimodal -- stack that column across checkpoints to feed
    ``plot_convergence_ridges`` (Figure 4.8).

    ``target_bins`` is ``[1, T]`` (``T = H * D``); its values only drive
    teacher forcing (they shift the decoder input), so any valid bin ids
    give a well-formed snapshot on an untrained head.
    """
    with torch.no_grad():
        prefix = fusion.encode_prefix(batch)
        logits = head.logits(prefix, target_bins)  # [1, T, n_bins]
        probs = torch.softmax(logits.float(), dim=-1)
    return probs[0].cpu().numpy()  # [T, n_bins]


# -- shared plotting helpers -----------------------------------------


def _finish(fig, out_path):
    """Save at 300 dpi, close the figure, return the path."""
    fig.savefig(str(out_path), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -- figure generators -----------------------------------------------


def plot_convergence_ridges(
    snapshots, out_path, steps=None, centers=None
):
    """Figure 4.8: softmax ridges of a head learning to be bimodal.

    ``snapshots`` is ``[S, n_bins]`` -- one probability vector per saved
    checkpoint for a single canary ``(timestep, joint)`` position (the
    column ``canary_snapshot`` produces, stacked over checkpoints). Each
    row is drawn as a filled ridge offset vertically by checkpoint, so
    the reader watches a flat uniform band at step 0 resolve into two
    separated peaks. ``steps`` labels each ridge (training step);
    ``centers`` sets the x axis to physical action values (else bin
    index). Grayscale-safe: ridges use graded gray fills and a black
    outline, distinguished by vertical position and label, never color.
    """
    snaps = np.asarray(snapshots, dtype=np.float64)
    if snaps.ndim != 2:
        raise ValueError(
            f"snapshots must be [S, n_bins]; got {snaps.shape}"
        )
    n_snaps, n_bins = snaps.shape
    x = np.asarray(centers) if centers is not None else np.arange(
        n_bins
    )
    if steps is None:
        steps = list(range(n_snaps))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # Offset each ridge so later checkpoints sit higher; scale peaks to
    # a fixed fraction of the offset so ridges read cleanly.
    offset = 1.0
    peak = float(snaps.max()) + _EPS
    for i in range(n_snaps):
        base = i * offset
        y = base + snaps[i] / peak * (offset * 0.9)
        gray = str(0.75 - 0.6 * i / max(n_snaps - 1, 1))
        ax.fill_between(
            x, base, y, facecolor=gray, edgecolor="black",
            linewidth=0.8, zorder=i,
        )
        ax.plot(x, y, color="black", linewidth=0.8, zorder=i)
        ax.text(
            x[0], base + 0.05, f"step {steps[i]}",
            fontsize=8, va="bottom",
        )
    ax.set_yticks([])
    ax.set_xlabel("action value (bin center)" if centers is not None
                  else "bin index")
    ax.set_ylabel("training progress")
    ax.set_title("Softmax convergence to a bimodal action")
    return _finish(fig, out_path)


def plot_bimodal_comparison(mse_pred, cat_probs, centers, out_path):
    """Figure 4.9: MSE point prediction vs categorical distribution.

    On one held-out pre-grasp state, the MSE baseline collapses to a
    single number (``mse_pred``, the conditional mean that aims straight
    at the cube), drawn as a vertical dashed marker. The categorical
    head's full distribution (``cat_probs`` over the ``centers`` of the
    base-rotation joint) is drawn as a bar histogram with two peaks and
    an empty valley -- exactly where ``mse_pred`` lands. Grayscale-safe:
    gray bars for the distribution, a black dashed line for the MSE
    prediction, both labeled.
    """
    centers = np.asarray(centers, dtype=np.float64)
    cat_probs = np.asarray(cat_probs, dtype=np.float64)
    width = (centers[1] - centers[0]) if len(centers) > 1 else 0.1

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(
        centers, cat_probs, width=width, color="0.6",
        edgecolor="black", linewidth=0.4,
        label="categorical policy p(a)",
    )
    ax.axvline(
        float(mse_pred), color="black", linestyle="--",
        linewidth=1.8, label="MSE prediction (mean)",
    )
    ax.annotate(
        "MSE lands in the empty valley",
        xy=(float(mse_pred), cat_probs.max() * 0.5),
        xytext=(float(mse_pred), cat_probs.max() * 0.95),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.set_xlabel("base-rotation action value")
    ax.set_ylabel("probability")
    ax.set_title("Bimodal stress test: collapse vs both modes")
    # Legend in the empty valley (lower center) so it never collides
    # with the annotation or either peak.
    ax.legend(fontsize=8, loc="lower center")
    return _finish(fig, out_path)


def open_loop_episode_predictions(policy, batch, index=0, horizon=None):
    """Decode one chunk open-loop -> ``(pred [H, D], true [H, D])``.

    The honest, sim-free "does the policy track the expert?" data prep
    for ``plot_action_traces``. Take one held-out example (``index``)
    from a chunked ``batch``, feed *its* ground-truth observation to the
    policy, and let the policy decode a full action chunk; the ground
    truth is that example's recorded ``action`` chunk. Both come back as
    ``[H, D]`` (predicted vs expert), aligned step-for-step.

    This is open-loop: the observation is the dataset's, not one the
    policy's own actions produced, so it never compounds errors -- the
    same optimism as ``rollout.evaluate_open_loop`` (which see). The
    policy replays its decoded chunk from its buffer, so calling
    ``select_action`` ``H`` times after a ``reset`` returns exactly the
    one decoded chunk.

    ``policy`` is any object with ``reset()`` / ``select_action(obs)``
    (a ``DiscretePolicy``); ``horizon`` defaults to the chunk length
    ``H`` from ``batch["action"]``.
    """
    action = np.asarray(batch["action"][index])  # [H, D]
    if horizon is None:
        horizon = action.shape[0]
    obs = _single_env_obs(batch, index)
    policy.reset()
    pred = np.stack(
        [np.asarray(policy.select_action(obs)) for _ in range(horizon)]
    )
    return pred, action[:horizon]


def _single_env_obs(batch, index):
    """Slice a chunked ``batch`` to a single-env ``encode_prefix`` obs.

    Pull example ``index`` and add the leading ``B = 1`` axis the fusion
    adapter expects: images ``[1, 3, H, W]``, state ``[1, 6]``, a
    length-1 ``task`` list.
    """
    up = batch["observation.images.up"][index]  # [3, H, W]
    side = batch["observation.images.side"][index]
    state = batch["observation.state"][index]  # [1, 6] from [B, 1, 6]
    if state.dim() == 1:  # tolerate an already-squeezed [6]
        state = state.unsqueeze(0)
    return {
        "observation.images.up": up.unsqueeze(0),
        "observation.images.side": side.unsqueeze(0),
        "observation.state": state,
        "task": [batch["task"][index]],
    }


def plot_action_traces(pred_chunks, true_chunks, out_path,
                       joint_names=None):
    """Open-loop trace figure: predicted vs ground-truth per joint.

    ``pred_chunks`` / ``true_chunks`` are ``[T, D]`` arrays -- the
    policy's decoded chunk actions and the dataset's actions over the
    same held-out episode window. One subplot per joint in a grid draws
    the expert (solid) against the policy (dashed) so the reader sees,
    joint by joint, whether the head tracks the demonstrations on data it
    never trained on. This is Chapter 4's honest qualitative eval:
    open-loop, no simulator, no domain-mismatched success number.

    Grayscale-safe by construction: both lines are black and are
    distinguished by *style* (solid ground truth vs dashed prediction),
    never by color. 300 dpi, Agg backend, returns ``out_path``.

    ``joint_names`` (optional) labels each subplot; defaults to
    ``joint 0 .. joint D-1``.
    """
    pred = np.asarray(pred_chunks, dtype=np.float64)
    true = np.asarray(true_chunks, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != true.shape:
        raise ValueError(
            "pred_chunks and true_chunks must share shape [T, D]; got "
            f"{pred.shape} and {true.shape}"
        )
    n_steps, dim = pred.shape
    if joint_names is None:
        joint_names = [f"joint {d}" for d in range(dim)]

    n_cols = 2 if dim <= 4 else 3
    n_rows = (dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.6, 1.7 * n_rows + 0.6),
        squeeze=False, sharex=True,
    )
    flat = axes.reshape(-1)
    steps = np.arange(n_steps)
    for d in range(dim):
        ax = flat[d]
        ax.plot(
            steps, true[:, d], color="black", linestyle="-",
            linewidth=1.3, label="ground truth",
        )
        ax.plot(
            steps, pred[:, d], color="black", linestyle="--",
            linewidth=1.3, label="predicted",
        )
        ax.set_title(joint_names[d], fontsize=9)
        ax.tick_params(labelsize=7)
    for k in range(dim, len(flat)):
        flat[k].axis("off")
    # One shared legend (styles carry the distinction, not color).
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, fontsize=8,
    )
    fig.suptitle(
        "Open-loop action tracking on held-out data", fontsize=10
    )
    fig.supxlabel("chunk step", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _finish(fig, out_path)


def plot_joint_coordination(
    par_samples, ar_samples, out_path, n_bins=None
):
    """Figure 4.10: gripper x wrist 2D histograms, parallel vs AR.

    Each of ``par_samples`` / ``ar_samples`` is ``[N, 2]`` sampled
    ``(gripper_bin, wrist_bin)`` pairs. Two side-by-side 2D histograms
    (shared bins, shared grayscale intensity scale) show the parallel
    head (left) leaking mass into off-diagonal quadrants -- incoherent
    joint combinations -- while the autoregressive head (right) keeps its
    samples on the demonstrated diagonal. Grayscale-safe: a single
    sequential gray colormap encodes density; the two panels are
    distinguished by position and title, and a reference diagonal is
    drawn for orientation.
    """
    par = np.asarray(par_samples)
    ar = np.asarray(ar_samples)
    if par.ndim != 2 or par.shape[1] != 2:
        raise ValueError(
            f"par_samples must be [N, 2]; got {par.shape}"
        )
    if n_bins is None:
        n_bins = int(max(par.max(), ar.max())) + 1
    edges = np.arange(n_bins + 1) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0), sharey=True)
    h_par, _, _ = np.histogram2d(
        par[:, 0], par[:, 1], bins=[edges, edges]
    )
    h_ar, _, _ = np.histogram2d(
        ar[:, 0], ar[:, 1], bins=[edges, edges]
    )
    vmax = float(max(h_par.max(), h_ar.max()))
    for ax, hist, title in (
        (axes[0], h_par, "Parallel head"),
        (axes[1], h_ar, "Autoregressive head"),
    ):
        ax.imshow(
            hist.T, origin="lower", cmap="Greys", vmin=0,
            vmax=vmax, interpolation="nearest",
        )
        ax.plot(
            [0, n_bins - 1], [0, n_bins - 1], color="black",
            linestyle=":", linewidth=0.8,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("gripper bin")
    axes[0].set_ylabel("wrist bin")
    fig.suptitle("Joint coordination: marginals vs joint samples")
    return _finish(fig, out_path)
