"""Exercise 4.6: k-means action binning (K = 256).

The shipped tokenizer bins each dimension *uniformly* over its q01/q99
range. Uniform bins spend resolution evenly even where the data is
sparse. An alternative (§4.3's design-space discussion) is to *learn*
the bin centers with k-means so bins cluster where the actions actually
concentrate. This exercise builds a per-dimension k-means tokenizer and
compares its round-trip error against the uniform one.

Your task: fill in ``fit_kmeans_centers`` and ``encode_kmeans``.

Hint: fit one 1-D k-means per dimension (``K = 256``) on the dataset's
action column -- ``sklearn.cluster.KMeans`` or a short Lloyd's loop.
``encode`` becomes nearest-center assignment; ``decode`` looks the
center up. Compare mean absolute round-trip error against
``ch04.action_tokenizer.ActionTokenizer`` (uniform); k-means should win
on the skewed joints. Mind the tokenizer's edge-CPU contract: the
shipped one is pure NumPy, so keep your inference path NumPy too.
"""

from __future__ import annotations

import numpy as np


def fit_kmeans_centers(
    actions: np.ndarray, k: int = 256, iters: int = 25
) -> np.ndarray:
    """Per-dimension 1-D k-means centers.

    ``actions``: ``[N, D]``. Returns ``[D, k]`` sorted centers (sorting
    keeps ``decode`` monotone, matching the uniform tokenizer's layout).
    """
    n, d = actions.shape
    centers = np.zeros((d, k))
    for dim in range(d):
        col = np.sort(actions[:, dim])
        # Initialize on quantiles, then run Lloyd's iterations.
        cen = np.quantile(col, np.linspace(0, 1, k))
        # TODO: repeat `iters` times:
        #   assign each value to its nearest center (np.searchsorted on
        #   the midpoints, or np.abs outer diff + argmin), then move each
        #   center to the mean of its assigned values; keep `cen` sorted.
        raise NotImplementedError("run the 1-D k-means updates")
        centers[dim] = cen
    return centers


def encode_kmeans(actions: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-center bin id per dimension. ``[..., D]`` -> ``[..., D]``."""
    # TODO: for each dim, assign the nearest center index.
    #   out[..., dim] = np.argmin(
    #       np.abs(actions[..., dim, None] - centers[dim]), axis=-1)
    raise NotImplementedError("nearest-center assignment")


if __name__ == "__main__":
    from ch04.chunk_data import make_chunk_dataset

    ds = make_chunk_dataset(episodes=[0])
    col = np.stack(
        [np.asarray(ds[i]["action"][0]) for i in range(ds.num_frames)]
    )
    centers = fit_kmeans_centers(col)
    bins = encode_kmeans(col, centers)
    recon = np.take_along_axis(centers.T, bins, axis=0)
    print("k-means mean abs error:", np.abs(recon - col).mean())
    print("(compare against ActionTokenizer's uniform round-trip)")
