"""Listing 4.1: the multimodal trap, on a one-dimensional toy.

A two-layer MLP trained with mean squared error on a bimodal target
(half the mass at -1, half at +1) predicts ~0.0 -- the one value that
appears in *none* of the training data. MSE's optimum is the
conditional mean, and the mean of two modes is the empty valley between
them (SS4.2.1). On the SO-100 pick-and-place task the same failure is a
policy that, trained on teleoperators who approach the cube from either
side, predicts the average heading and drives straight into the cube.

Run as ``python scripts/toy_bimodal.py`` to reproduce the near-zero
prediction and save a figure (the two data modes as a histogram with
the MSE prediction landing in the valley) to ``figures/``. The core is
factored into ``run_toy_bimodal`` so the test can import it and run
fewer steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

_FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def run_toy_bimodal(
    seed: int = 0, steps: int = 5000, n: int = 2000
) -> float:
    """Train the MSE toy and return the final prediction at obs = 0.

    Verbatim semantics of manuscript Listing 4.1: a 1-D bimodal target
    (modes at -1 / +1 with narrow Gaussian noise), all sharing one
    observation so ``p(a | o)`` is the full mixture; a two-layer MLP
    trained with ``nn.MSELoss`` for ``steps`` Adam steps. Returns the
    scalar prediction, which collapses toward 0.0 (the mean of the two
    modes). ``steps`` is overridable so tests run fewer.
    """
    torch.manual_seed(seed)

    modes = torch.randint(0, 2, (n, 1)) * 2 - 1
    actions = modes + 0.05 * torch.randn(n, 1)
    obs = torch.zeros(n, 1)

    model = nn.Sequential(
        nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(steps):
        pred = model(obs)
        loss = loss_fn(pred, actions)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        return float(model(torch.zeros(1, 1)).item())


def _save_figure(actions, prediction: float, out_path: Path):
    """Histogram of the two data modes with the MSE prediction marked.

    Grayscale-safe (FIGURE_STYLE_GUIDE): gray bars for the demonstrated
    action distribution, a black dashed line for the collapsed MSE
    prediction sitting in the empty valley between the modes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = actions.reshape(-1).numpy()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(
        data, bins=60, color="0.6", edgecolor="black",
        linewidth=0.4, label="expert actions (two modes)",
    )
    ax.axvline(
        prediction, color="black", linestyle="--", linewidth=1.8,
        label=f"MSE prediction = {prediction:.3f}",
    )
    ax.annotate(
        "MSE lands in the empty valley",
        xy=(prediction, ax.get_ylim()[1] * 0.45),
        xytext=(prediction, ax.get_ylim()[1] * 0.85),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.set_xlabel("action value")
    ax.set_ylabel("count")
    ax.set_title("Listing 4.1: MSE collapses a bimodal target")
    ax.legend(fontsize=8)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--out", type=Path,
        default=_FIG_DIR / "figure_4_1_toy_bimodal_collapse.png",
    )
    args = parser.parse_args()

    # Reproduce the data for the figure with the same seed the training
    # used, so the histogram matches the run that produced the number.
    torch.manual_seed(args.seed)
    modes = torch.randint(0, 2, (2000, 1)) * 2 - 1
    actions = modes + 0.05 * torch.randn(2000, 1)

    prediction = run_toy_bimodal(seed=args.seed, steps=args.steps)
    print(f"MSE prediction at obs=0: {prediction:.4f}")
    print("(near 0.0 -- the empty valley between the two modes)")

    path = _save_figure(actions, prediction, args.out)
    print(f"saved figure to {path}")


if __name__ == "__main__":
    main()
