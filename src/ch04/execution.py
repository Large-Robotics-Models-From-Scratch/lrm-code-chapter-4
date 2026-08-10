"""Action-chunk execution schedules from section 4.7.2."""

from __future__ import annotations

import math

import torch


class TemporalEnsembler:
    """Blend overlapping chunk predictions for the current timestep.

    The earliest overlapping prediction receives weight 1. Later
    predictions receive ``exp(-decay * i)`` in arrival order, matching
    the manuscript's convention. Call :meth:`reset` between episodes.
    """

    def __init__(self, decay: float = 0.01) -> None:
        if decay < 0:
            raise ValueError("decay must be non-negative")
        self.decay = decay
        self._chunks: list[tuple[int, torch.Tensor]] = []
        self._time = 0

    def add(self, chunk: torch.Tensor) -> torch.Tensor:
        """Add ``[H, D]`` and return the ensembled action now due."""
        if chunk.ndim != 2:
            raise ValueError("chunk must have shape [H, D]")
        self._chunks.append((self._time, chunk.detach().clone()))
        predictions = []
        live = []
        for start, candidate in self._chunks:
            offset = self._time - start
            if offset < candidate.shape[0]:
                live.append((start, candidate))
                predictions.append(candidate[offset])
        weights = [
            math.exp(-self.decay * index)
            for index in range(len(predictions))
        ]
        self._chunks = live
        stacked = torch.stack(predictions)
        weight = torch.tensor(
            weights, device=stacked.device, dtype=stacked.dtype
        )
        result = (stacked * weight[:, None]).sum(0) / weight.sum()
        self._time += 1
        return result

    def reset(self) -> None:
        self._chunks.clear()
        self._time = 0


def execute_chunk(chunk: torch.Tensor):
    """Yield the ``H`` controls of one chunk in order."""
    if chunk.ndim != 2:
        raise ValueError("chunk must have shape [H, D]")
    yield from chunk.unbind(dim=0)
