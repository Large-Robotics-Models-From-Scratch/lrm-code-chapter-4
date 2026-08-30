"""Action-chunk execution schedules from section 4.7.2.

Section 4.7.2 lists three ways to turn a stream of decoded ``[H, D]``
chunks into one control per timestep: run a whole chunk open loop,
replan every timestep and keep only its first row, or blend the
overlapping predictions that refer to the same absolute time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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


def _stack_chunks(chunks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Validate a per-timestep chunk stream and stack it to ``[T,H,D]``."""
    if len(chunks) == 0:
        raise ValueError("chunks must contain at least one [H, D] chunk")
    stacked = torch.stack([torch.as_tensor(c) for c in chunks])
    if stacked.ndim != 3:
        raise ValueError("every chunk must have shape [H, D]")
    return stacked


def chunk_by_chunk_trace(
    chunks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Open-loop schedule: run all ``H`` rows before replanning.

    ``chunks[t]`` is the chunk the policy would decode at timestep ``t``.
    This schedule consumes only every ``H``-th chunk, which is what makes
    it the cheapest of the three and the slowest to react.
    """
    stacked = _stack_chunks(chunks)
    horizon = stacked.shape[1]
    steps = stacked.shape[0]
    rows = [
        stacked[step - step % horizon, step % horizon]
        for step in range(steps)
    ]
    return torch.stack(rows)


def receding_horizon_trace(
    chunks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Replan every timestep and execute only the freshest first row."""
    return _stack_chunks(chunks)[:, 0]


def temporal_ensemble_trace(
    chunks: Sequence[torch.Tensor],
    decay: float = 0.01,
) -> torch.Tensor:
    """Replan every timestep and blend the overlapping predictions."""
    stacked = _stack_chunks(chunks)
    ensembler = TemporalEnsembler(decay=decay)
    return torch.stack([ensembler.add(chunk) for chunk in stacked])


def execution_schedules(
    chunks: Sequence[torch.Tensor],
    decay: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Return all three section 4.7.2 traces for one chunk stream."""
    return {
        "chunk-by-chunk": chunk_by_chunk_trace(chunks),
        "receding horizon": receding_horizon_trace(chunks),
        "temporal ensemble": temporal_ensemble_trace(chunks, decay),
    }
