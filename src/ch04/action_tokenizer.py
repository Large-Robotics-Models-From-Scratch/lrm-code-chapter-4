"""Uniform per-dimension action tokenizer from listing 4.1."""

from __future__ import annotations

import numpy as np

from ch04.constants import ACTION_BINS, SMOLLM_VOCAB


class ActionTokenizer:
    """Quantize continuous actions using per-dimension uniform bins.

    The tokenizer is intentionally NumPy-only. It operates in the
    Chapter 2 z-score-normalized action space and maps bins onto the
    final ``n_bins`` ids of SmolLM's native vocabulary.
    """

    def __init__(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        n_bins: int = ACTION_BINS,
        vocab_size: int = SMOLLM_VOCAB,
    ) -> None:
        self.lo = np.asarray(lo, dtype=np.float32)
        self.hi = np.asarray(hi, dtype=np.float32)
        if self.lo.ndim != 1 or self.hi.shape != self.lo.shape:
            raise ValueError("lo and hi must be matching 1-D arrays")
        if not np.isfinite(self.lo).all() or not np.isfinite(self.hi).all():
            raise ValueError("lo and hi must contain only finite values")
        reversed_dims = np.flatnonzero(self.hi < self.lo)
        if reversed_dims.size:
            raise ValueError(
                "hi must be greater than lo in every dimension; "
                f"reversed bounds at dimensions {reversed_dims.tolist()}"
            )
        stationary_dims = np.flatnonzero(self.hi == self.lo)
        if stationary_dims.size:
            raise ValueError(
                "hi must be greater than lo in every dimension; "
                f"dimensions {stationary_dims.tolist()} have zero range "
                "(the corresponding joint may never have moved)"
            )
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer greater than one")
        if vocab_size < n_bins:
            raise ValueError("vocab_size must be at least n_bins")

        self.n_bins = n_bins
        self.vocab_size = vocab_size
        self.token_base = vocab_size - n_bins
        width = (self.hi - self.lo) / n_bins
        offsets = np.arange(n_bins, dtype=np.float32) + 0.5
        self.centers = self.lo[:, None] + width[:, None] * offsets

    @property
    def action_dim(self) -> int:
        """Number of independently quantized control dimensions."""
        return int(self.lo.size)

    def encode(self, action: np.ndarray) -> np.ndarray:
        """Map actions shaped ``[..., D]`` to bin ids as ``int64``."""
        values = np.asarray(action)
        self._check_last_dim(values, "action")
        if not np.isfinite(values).all():
            raise ValueError("action must contain only finite values")
        clipped = np.clip(values, self.lo, self.hi)
        norm = (clipped - self.lo) / (self.hi - self.lo)
        ids = np.floor(norm * self.n_bins).astype(np.int64)
        return np.clip(ids, 0, self.n_bins - 1)

    def decode(self, ids: np.ndarray) -> np.ndarray:
        """Map bin ids shaped ``[..., D]`` to bin-center actions."""
        indices = np.asarray(ids)
        self._check_last_dim(indices, "ids")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("ids must have an integer dtype")
        if np.any(indices < 0) or np.any(indices >= self.n_bins):
            raise ValueError("ids must lie in [0, n_bins)")
        flat = indices.reshape(-1, self.action_dim)
        dimensions = np.arange(self.action_dim)[None, :]
        decoded = self.centers[dimensions, flat]
        return decoded.astype(np.float32, copy=False).reshape(indices.shape)

    def bins_to_tokens(self, bins: np.ndarray) -> np.ndarray:
        """Map bin ids to the reserved tail of the native LM vocab."""
        values = np.asarray(bins)
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("bins must have an integer dtype")
        if np.any(values < 0) or np.any(values >= self.n_bins):
            raise ValueError("bins must lie in [0, n_bins)")
        return (values.astype(np.int64) + self.token_base).astype(np.int64)

    def tokens_to_bins(self, tokens: np.ndarray) -> np.ndarray:
        """Invert :meth:`bins_to_tokens`."""
        values = np.asarray(tokens)
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("tokens must have an integer dtype")
        if np.any(values < self.token_base) or np.any(
            values >= self.vocab_size
        ):
            raise ValueError("tokens are outside the reserved action range")
        return (values.astype(np.int64) - self.token_base).astype(np.int64)

    def _check_last_dim(self, values: np.ndarray, name: str) -> None:
        if values.ndim == 0 or values.shape[-1] != self.action_dim:
            raise ValueError(
                f"{name} must have final dimension {self.action_dim}"
            )


def fit_action_tokenizer(
    actions: np.ndarray,
    n_bins: int = ACTION_BINS,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> ActionTokenizer:
    """Fit q01/q99 limits to normalized actions shaped ``[..., D]``."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("actions must be shaped [..., D]")
    if not np.isfinite(values).all():
        raise ValueError("actions must contain only finite values")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError(
            "percentiles must satisfy 0 <= lower < upper <= 100"
        )
    flat = values.reshape(-1, values.shape[-1])
    lo, hi = np.percentile(
        flat, [lower_percentile, upper_percentile], axis=0
    )
    return ActionTokenizer(lo, hi, n_bins=n_bins)
