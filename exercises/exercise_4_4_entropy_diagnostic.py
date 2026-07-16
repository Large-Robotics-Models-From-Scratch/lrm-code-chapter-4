"""Exercise 4.4: per-dimension softmax-entropy diagnostic.

§4.6.7 tracks the head's per-dimension softmax entropy as a training
canary. It starts at ``log(256) ≈ 5.54`` nats (uniform) and should fall
to the ``2-3`` band as the head commits to bins. Too broad means
underfitting; too narrow on a known-multimodal joint means mode
collapse. This exercise builds the diagnostic plot: entropy per joint,
so you can spot one dimension collapsing while the others are healthy.

Your task: fill in ``per_dimension_entropy`` and plot the six values.

Hint: ``ch04.diagnostics.softmax_entropy`` already reduces the last
(bin) axis. Given logits ``[B, H, D, n_bins]``, average the entropy
over the batch and horizon to get one number per dimension ``D``. Then
a simple ``matplotlib`` bar chart over the six joints is the figure.
"""

from __future__ import annotations

import numpy as np

from ch04.diagnostics import softmax_entropy


def per_dimension_entropy(logits) -> np.ndarray:
    """Mean softmax entropy per action dimension.

    ``logits``: ``[B, H, D, n_bins]`` (torch or NumPy). Returns a
    length-``D`` array of mean entropies (nats).
    """
    ent = softmax_entropy(logits)  # [B, H, D]
    # TODO: average over the batch and horizon axes (0 and 1), leaving
    #   one entropy per dimension:
    #   arr = ent.numpy() if hasattr(ent, "numpy") else np.asarray(ent)
    #   return arr.mean(axis=(0, 1))
    raise NotImplementedError("reduce to one entropy per dimension")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    # Five broad joints + one collapsed joint (dim 5), to see the plot
    # flag the collapse.
    logits = rng.normal(0, 0.1, size=(8, 16, 6, 256))
    logits[:, :, 5, 100] += 12.0  # spike -> low entropy on dim 5
    ent = per_dimension_entropy(logits)
    joints = ["shoulder", "upper", "elbow", "wrist", "roll", "gripper"]
    print("per-dim entropy (nats):", np.round(ent, 3))
    plt.bar(joints, ent, color="0.5")
    plt.axhline(np.log(256), ls="--", c="black", label="uniform (5.54)")
    plt.ylabel("entropy (nats)")
    plt.legend()
    plt.title("Exercise 4.4: per-dimension entropy diagnostic")
    plt.tight_layout()
    plt.savefig("figures/exercise_4_4_entropy.png", dpi=150)
    print("saved figures/exercise_4_4_entropy.png")
