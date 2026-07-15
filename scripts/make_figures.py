"""Regenerate Chapter 4's evidence figures (Figures 4.8-4.10).

Figures 4.8 (softmax convergence ridges), 4.9 (bimodal stress test:
MSE vs categorical), and 4.10 (joint coordination: parallel vs
autoregressive) are the payoff of SS4.2-4.6. The real figures come from
a trained checkpoint (Task 11); this CLI is the machinery that turns a
checkpoint -- or, absent one, deterministic synthetic distributions --
into the exact figures, so the plotting path is correct and reproducible
before the checkpoint exists.

Run ``python scripts/make_figures.py`` to write all three figures from
synthetic inputs into ``figures/``. Pass ``--checkpoint PATH`` to drive
them from a trained head instead; if the path is absent the script says
so and falls back to synthetic inputs (it never silently produces an
empty figure). Individual figures can be selected with ``--only 4.8``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ch04.diagnostics import (
    plot_bimodal_comparison,
    plot_convergence_ridges,
    plot_joint_coordination,
)

_FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

_FILES = {
    "4.8": "figure_4_8_convergence_ridges.png",
    "4.9": "figure_4_9_bimodal_comparison.png",
    "4.10": "figure_4_10_joint_coordination.png",
}


def _synthetic_ridges(n_bins: int = 64):
    """Uniform -> emerging bimodal -> sharp bimodal softmax snapshots."""
    centers = np.linspace(-1.0, 1.0, n_bins)
    snaps = []
    for sharp in (0.0, 3.0, 12.0):
        left = np.exp(-sharp * (centers + 0.5) ** 2)
        right = np.exp(-sharp * (centers - 0.5) ** 2)
        p = left + right + 1e-3
        snaps.append(p / p.sum())
    return np.array(snaps), centers


def _synthetic_bimodal(n_bins: int = 64):
    """A two-peaked categorical distribution and the MSE mean (0.0)."""
    centers = np.linspace(-1.0, 1.0, n_bins)
    left = np.exp(-8.0 * (centers + 0.5) ** 2)
    right = np.exp(-8.0 * (centers - 0.5) ** 2)
    probs = left + right
    probs = probs / probs.sum()
    return 0.0, probs, centers


def _synthetic_coordination(n: int = 400, n_bins: int = 32):
    """AR samples on the diagonal; parallel leaks off-diagonal."""
    rng = np.random.default_rng(0)
    diag = rng.integers(0, n_bins, size=n)
    jitter = rng.integers(-1, 2, size=n)
    ar = np.stack(
        [diag, np.clip(diag + jitter, 0, n_bins - 1)], axis=1
    )
    # Parallel: correct marginals (same diagonal draw per axis) but
    # independent, so pairs scatter off the diagonal.
    par = np.stack(
        [
            rng.permutation(diag),
            rng.permutation(diag),
        ],
        axis=1,
    )
    return par, ar, n_bins


def make_figure_4_8(out_dir: Path, checkpoint=None) -> Path:
    if checkpoint is not None:
        print(
            "figure 4.8: checkpoint-driven snapshots not wired here "
            "(Task 11 supplies the trained head); using synthetic."
        )
    snaps, centers = _synthetic_ridges()
    return plot_convergence_ridges(
        snaps, out_dir / _FILES["4.8"],
        steps=[0, 5000, 20000], centers=centers,
    )


def make_figure_4_9(out_dir: Path, checkpoint=None) -> Path:
    if checkpoint is not None:
        print(
            "figure 4.9: checkpoint-driven distribution not wired "
            "here (Task 11); using synthetic."
        )
    mse_pred, cat_probs, centers = _synthetic_bimodal()
    return plot_bimodal_comparison(
        mse_pred, cat_probs, centers, out_dir / _FILES["4.9"]
    )


def make_figure_4_10(out_dir: Path, checkpoint=None) -> Path:
    if checkpoint is not None:
        print(
            "figure 4.10: checkpoint-driven samples not wired here "
            "(Task 11); using synthetic."
        )
    par, ar, n_bins = _synthetic_coordination()
    return plot_joint_coordination(
        par, ar, out_dir / _FILES["4.10"], n_bins=n_bins
    )


_MAKERS = {
    "4.8": make_figure_4_8,
    "4.9": make_figure_4_9,
    "4.10": make_figure_4_10,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="trained head checkpoint (Task 11); falls back to "
             "synthetic inputs with a message if missing.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_FIG_DIR,
    )
    parser.add_argument(
        "--only", choices=sorted(_MAKERS), default=None,
        help="render just one figure (default: all three).",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is not None and not checkpoint.exists():
        print(
            f"checkpoint {checkpoint} not found; falling back to "
            "synthetic inputs (real figures come from Task 11's run)."
        )
        checkpoint = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = [args.only] if args.only else sorted(_MAKERS)
    for key in selected:
        path = _MAKERS[key](args.out_dir, checkpoint=checkpoint)
        print(f"figure {key}: wrote {path}")


if __name__ == "__main__":
    main()
