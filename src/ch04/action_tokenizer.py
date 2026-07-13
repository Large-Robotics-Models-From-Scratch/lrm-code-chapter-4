"""Uniform per-dimension action tokenizer for Chapter 4.

Converts continuous SO-101 action vectors into discrete bin ids and
back (the RT-1 recipe), and maps those bins onto reserved token ids
inside a language-model vocabulary (the RT-2 / OpenVLA recipe) so the
autoregressive head can decode actions through the same transformer
that carries text.

Pure NumPy, no torch dependency: the same tokenizer runs at training
time on GPU-side data and at deployment time on an edge CPU.

Vocabulary convention (matches the chapter's listing 4.7): the last
``n_bins`` ids of the language model vocabulary are reserved as action
tokens, *shared across all action dimensions*. Position in the decoded
sequence (which timestep, which joint) disambiguates the dimension,
the same way OpenVLA reuses 256 ids for all seven action dimensions.
"""

from __future__ import annotations

import numpy as np

from ch04 import N_BINS, SMOLLM_VOCAB

_EPS = 1e-8


class ActionTokenizer:
    """Affine uniform quantizer over a fixed per-dimension range.

    The tokenizer holds no learned parameters: bin edges are fixed at
    construction from ``lo``/``hi`` and a bin count. ``encode`` maps a
    ``[..., D]`` array of continuous actions to ``[..., D]`` integer
    bin ids; ``decode`` maps bin ids back to bin-center action values.
    """

    def __init__(self, lo, hi, n_bins: int = N_BINS):
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        if lo.shape != hi.shape:
            raise ValueError(
                f"lo and hi must share a shape; "
                f"got {lo.shape} vs {hi.shape}"
            )
        if lo.ndim != 1:
            raise ValueError(f"lo/hi must be 1-D [D]; got ndim {lo.ndim}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2; got {n_bins}")
        if np.any(hi < lo):
            raise ValueError("every hi must be >= its lo")

        self.lo = lo
        self.hi = hi
        self.n_bins = int(n_bins)
        # Per-dimension edges [D, n_bins + 1] and centers [D, n_bins].
        # n_bins + 1 edges yield exactly n_bins centers.
        steps = np.linspace(0.0, 1.0, self.n_bins + 1)  # [n_bins + 1]
        span = (hi - lo)[:, None]  # [D, 1]
        self.edges = lo[:, None] + steps[None, :] * span
        self.centers = (self.edges[:, :-1] + self.edges[:, 1:]) / 2.0

    @property
    def action_dim(self) -> int:
        return int(self.lo.shape[0])

    # -- continuous <-> bin id --------------------------------------

    def encode(self, action) -> np.ndarray:
        """Continuous ``[..., D]`` -> integer bin ids ``[..., D]``.

        clip -> normalize by the per-dim range -> floor into a bin
        -> clip the bin id itself, so a value exactly at ``hi`` lands
        in bin ``n_bins - 1`` rather than the out-of-range
        ``n_bins``.
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"last dim must be {self.action_dim}; "
                f"got {action.shape[-1]}"
            )
        action = np.clip(action, self.lo, self.hi)
        norm = (action - self.lo) / (self.hi - self.lo + _EPS)
        ids = np.floor(norm * self.n_bins).astype(np.int64)
        return np.clip(ids, 0, self.n_bins - 1)

    def decode(self, ids) -> np.ndarray:
        """Bin ids ``[..., D]`` -> bin-center actions ``[..., D]``."""
        ids = np.asarray(ids)
        if ids.shape[-1] != self.action_dim:
            raise ValueError(
                f"last dim must be {self.action_dim}; "
                f"got {ids.shape[-1]}"
            )
        ids = np.clip(ids, 0, self.n_bins - 1)
        # Broadcast centers [D, n_bins] to [..., D, n_bins] and gather.
        target = ids.shape[:-1] + self.centers.shape
        centers = np.broadcast_to(self.centers, target)
        return np.take_along_axis(
            centers, ids[..., None], axis=-1
        )[..., 0]

    # -- bin id <-> reserved language-model token id -----------------

    def to_token_ids(
        self, bin_ids, vocab_size: int = SMOLLM_VOCAB
    ) -> np.ndarray:
        """Bin ids ``[..., D]`` -> reserved LM token ids ``[..., D]``.

        ``token_id = bin_id + vocab_size - n_bins``: the last
        ``n_bins`` ids of the vocabulary are reserved, shared across
        every action dimension.
        """
        bin_ids = np.asarray(bin_ids)
        if bin_ids.min() < 0 or bin_ids.max() > self.n_bins - 1:
            raise ValueError(
                f"bin_ids must lie in [0, {self.n_bins - 1}]; got "
                f"range [{bin_ids.min()}, {bin_ids.max()}]"
            )
        base = int(vocab_size) - self.n_bins
        return bin_ids + base

    def from_token_ids(
        self, token_ids, vocab_size: int = SMOLLM_VOCAB
    ) -> np.ndarray:
        """Reserved LM token ids -> bin ids ``[..., D]`` (inverse)."""
        token_ids = np.asarray(token_ids)
        base = int(vocab_size) - self.n_bins
        bin_ids = token_ids - base
        if bin_ids.min() < 0 or bin_ids.max() > self.n_bins - 1:
            raise ValueError(
                f"token_ids must lie in [{base}, {base + self.n_bins - 1}]"
                f"; got range [{token_ids.min()}, {token_ids.max()}]"
            )
        return bin_ids

    # -- constructors --------------------------------------------------

    @classmethod
    def from_lerobot_stats(
        cls, stats: dict, key: str = "action",
        n_bins: int = N_BINS,
    ) -> "ActionTokenizer":
        """Build from a LeRobot ``meta/stats.json``-style dict.

        Uses the robust Q01/Q99 percentiles as the bin range so a
        handful of teleop outliers do not waste bins on behavior the
        policy never needs to reproduce. The real
        ``lerobot/svla_so101_pickplace`` stats do not carry these
        percentiles (only min/max/mean/std/count) -- use
        ``from_lerobot_dataset`` in that case, which computes them
        directly from the action column.
        """
        feat = stats[key]
        if "q01" not in feat or "q99" not in feat:
            raise ValueError(
                f"stats[{key!r}] is missing 'q01'/'q99' percentiles "
                "(only min/max/mean/std/count are present, e.g. the "
                "real lerobot/svla_so101_pickplace dataset). Use "
                "ActionTokenizer.from_lerobot_dataset(dataset, "
                f"key={key!r}) instead, which computes the 1st/99th "
                "percentiles directly from the action column."
            )
        return cls(
            lo=np.asarray(feat["q01"], dtype=np.float64),
            hi=np.asarray(feat["q99"], dtype=np.float64),
            n_bins=n_bins,
        )

    @classmethod
    def from_lerobot_dataset(
        cls, dataset, key: str = "action",
        n_bins: int = N_BINS,
    ) -> "ActionTokenizer":
        """Build from a LeRobot dataset lacking q01/q99 in its stats.

        Computes the 1st/99th percentiles directly over the action
        column with a single vectorized NumPy pass (~12k frames for
        ``svla_so101_pickplace``), rather than trusting the dataset's
        saturated min/max endpoints, which would waste bin resolution
        on outliers the policy never needs to reproduce.

        Accepts any object exposing the column via
        ``dataset.hf_dataset[key]`` (the lerobot 0.5.1
        ``LeRobotDataset`` API); rows may be lists, NumPy arrays, or
        torch tensors -- each is converted with ``np.asarray`` so this
        module never imports torch.
        """
        column = dataset.hf_dataset[key]
        rows = [np.asarray(row, dtype=np.float64) for row in column]
        actions = np.stack(rows, axis=0)  # [N, D]
        lo = np.percentile(actions, 1, axis=0)
        hi = np.percentile(actions, 99, axis=0)
        return cls(lo=lo, hi=hi, n_bins=n_bins)

    def __repr__(self) -> str:
        return (
            f"ActionTokenizer(action_dim={self.action_dim}, "
            f"n_bins={self.n_bins})"
        )
