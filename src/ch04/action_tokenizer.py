"""Uniform per-dimension action tokenizer for Chapter 4.

Converts continuous SO-101 action vectors into discrete bin ids and back
(the RT-1 recipe), and maps those bins onto reserved token ids inside a
language-model vocabulary (the RT-2 / OpenVLA recipe) so the autoregressive
head can decode actions through the same transformer that carries text.

Pure NumPy, no torch dependency: the same tokenizer runs at training time
on GPU-side data and at deployment time on an edge CPU.

Vocabulary convention (matches the chapter's listing 4.7): the last
``n_bins`` ids of the language model vocabulary are reserved as action
tokens, *shared across all action dimensions*. Position in the decoded
sequence (which timestep, which joint) disambiguates the dimension, the
same way OpenVLA reuses 256 ids for all seven action dimensions.
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_BINS = 256  # RT-1 / OpenVLA choice: one byte per bin
_EPS = 1e-8


class ActionTokenizer:
    """Affine uniform quantizer over a fixed per-dimension range.

    The tokenizer holds no learned parameters: bin edges are fixed at
    construction from ``lo``/``hi`` and a bin count. ``encode`` maps a
    ``[..., D]`` array of continuous actions to ``[..., D]`` integer bin
    ids; ``decode`` maps bin ids back to bin-center action values.
    """

    def __init__(self, lo, hi, n_bins: int = DEFAULT_N_BINS):
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
        self.edges = lo[:, None] + steps[None, :] * span  # [D, n_bins+1]
        self.centers = (self.edges[:, :-1] + self.edges[:, 1:]) / 2.0

    @property
    def action_dim(self) -> int:
        return int(self.lo.shape[0])

    # -- continuous <-> bin id ------------------------------------------

    def encode(self, action) -> np.ndarray:
        """Continuous ``[..., D]`` -> integer bin ids ``[..., D]``."""
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

    # -- bin id <-> reserved language-model token id --------------------

    def base_token_id(self, vocab_size: int) -> int:
        """First reserved action-token id: ``vocab_size - n_bins``."""
        if vocab_size < self.n_bins:
            raise ValueError(
                f"vocab_size {vocab_size} < n_bins {self.n_bins}"
            )
        return int(vocab_size) - self.n_bins

    def reserved_range(self, vocab_size: int) -> tuple[int, int]:
        """Half-open ``[start, end)`` of reserved action-token ids."""
        start = self.base_token_id(vocab_size)
        return start, int(vocab_size)

    def to_token_ids(self, action, vocab_size: int) -> np.ndarray:
        """Continuous actions -> reserved LM token ids ``[..., D]``."""
        return self.encode(action) + self.base_token_id(vocab_size)

    def from_token_ids(self, token_ids, vocab_size: int) -> np.ndarray:
        """Reserved LM token ids -> bin-center actions ``[..., D]``."""
        token_ids = np.asarray(token_ids)
        bins = token_ids - self.base_token_id(vocab_size)
        return self.decode(bins)

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_lerobot_stats(
        cls, stats: dict, key: str = "action",
        n_bins: int = DEFAULT_N_BINS,
    ) -> "ActionTokenizer":
        """Build from a LeRobot ``meta/stats.json``-style dict.

        Uses the robust Q01/Q99 percentiles as the bin range so a handful
        of teleop outliers do not waste bins on behavior the policy never
        needs to reproduce.
        """
        feat = stats[key]
        return cls(
            lo=np.asarray(feat["q01"], dtype=np.float64),
            hi=np.asarray(feat["q99"], dtype=np.float64),
            n_bins=n_bins,
        )

    def __repr__(self) -> str:
        return (
            f"ActionTokenizer(action_dim={self.action_dim}, "
            f"n_bins={self.n_bins})"
        )
