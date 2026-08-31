"""One figure style shared by the Colab and ``ch04-figures``.

Every plot in the chapter should be legible in a printed book: no
reliance on colour alone, no default matplotlib blue-on-white, and axis
labels that name the quantity and its units. Calling
:func:`use_manuscript_style` once per session is enough; the plotting
helpers in :mod:`ch04.diagnostics` read the palette from here.
"""

from __future__ import annotations

# One stable identity per action head, used in every comparison figure so
# a reader can carry the colour across plots. Distinct dash patterns keep
# the three separable in grayscale print.
HEAD_STYLES: dict[str, dict[str, object]] = {
    "factorized": {
        "color": "#4C72B0",
        "linestyle": "--",
        "marker": "o",
        "label": "factorized (one-shot)",
    },
    "autoregressive": {
        "color": "#DD8452",
        "linestyle": "-.",
        "marker": "s",
        "label": "autoregressive",
    },
    "parallel": {
        "color": "#55A868",
        "linestyle": "-",
        "marker": "^",
        "label": "parallel (bidirectional)",
    },
}

EXPERT_COLOR = "#33333A"
POLICY_COLOR = "#C44E52"
SUPPORTED_COLOR = "#4C72B0"
UNSUPPORTED_COLOR = "#C44E52"
NEUTRAL_COLOR = "#8C8C94"
GRID_COLOR = "#D9D9DE"


def head_style(name: str) -> dict[str, object]:
    """Return the plotting keywords reserved for one head."""
    return dict(
        HEAD_STYLES.get(
            name,
            {
                "color": NEUTRAL_COLOR,
                "linestyle": "-",
                "marker": ".",
                "label": name,
            },
        )
    )


def head_color(name: str) -> str:
    """Return the colour reserved for one head."""
    return str(head_style(name)["color"])


def head_label(name: str) -> str:
    """Return the display label reserved for one head."""
    return str(head_style(name)["label"])


def use_manuscript_style() -> None:
    """Apply the chapter's matplotlib defaults to the current session."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 8,
            "axes.labelsize": 10,
            "axes.labelpad": 4,
            "axes.edgecolor": "#5A5A62",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.65,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#5A5A62",
            "ytick.color": "#5A5A62",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.88,
            "legend.edgecolor": GRID_COLOR,
            "legend.borderpad": 0.5,
            "lines.linewidth": 1.25,
            "lines.solid_capstyle": "round",
            "image.cmap": "magma",
            "figure.constrained_layout.use": True,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[style["color"] for style in HEAD_STYLES.values()]
        + [POLICY_COLOR, NEUTRAL_COLOR]
    )


def annotate_source(figure, text: str) -> None:
    """Stamp a small provenance line under a figure.

    Section 4.6.1 asks that a reported result carry the checkpoint,
    anchor, neighbour count, and seed. Keeping it in the figure means the
    number and its provenance cannot be separated later.
    """
    figure.text(
        0.5,
        -0.015,
        text,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#6E6E76",
    )
