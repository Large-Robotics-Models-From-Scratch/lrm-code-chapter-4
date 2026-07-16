"""Chunked LeRobot data loading.

Wraps ``LeRobotDataset`` so each sample carries a full ``CHUNK_H``-step
action chunk instead of a single frame, using lerobot's own
``delta_timestamps`` mechanism (no custom windowing code). The
manuscript's Listing 4.12 hardcodes ``t / 50.0`` (a 50 Hz assumption);
this module derives the offsets from the dataset's *real* fps (30 for
``svla_so101_pickplace``) via ``chunk_delta_timestamps(fps, chunk_h)``,
so the chunk still spans ``chunk_h`` consecutive frames regardless of
the source fps.

Episode-boundary chunks: when a requested offset falls before an
episode's start or past its end, lerobot clamps the frame index to the
boundary *and* returns a companion ``action_is_pad`` boolean mask
(shape ``[chunk_h]``, per-sample; ``[B, chunk_h]`` once batched) where
``True`` marks a clamped, repeated (padding) step. This comes from
lerobot's internal ``dataset_reader._get_query_indices`` -- it emits a
``f"{key}_is_pad"`` entry for every key that has ``delta_timestamps``
set, so ``observation.state`` and the two camera keys also get
(mostly all-``False``, since their chunk is a single ``[0.0]`` offset)
``*_is_pad`` masks, but only ``action_is_pad`` varies meaningfully.
Task 7's training loop must mask the loss at padded steps using
``batch["action_is_pad"]``.

One more lerobot 0.5.1 shape asymmetry to know: video keys queried
with the single ``[0.0]`` offset come back squeezed
(``[B, 3, 480, 640]``), but tabular keys keep the stacked chunk dim
-- ``observation.state`` arrives as ``[B, 1, 6]``, not ``[B, 6]``.
Squeeze dim 1 before handing state to the Chapter 3 backbone.
"""

from __future__ import annotations

import torch
from ch03.preprocess import preprocess_image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from ch04 import CHUNK_H


def chunk_delta_timestamps(
    fps: float, chunk_h: int = CHUNK_H
) -> dict[str, list[float]]:
    """Per-key relative timestamps for one ``chunk_h``-step sample.

    Observation keys (state, both cameras) only ever need the current
    frame (offset ``0.0``); ``action`` needs the next ``chunk_h`` steps
    so the head can predict a full chunk, spaced at ``1 / fps`` (the
    dataset's real sampling rate, not a hardcoded constant).
    """
    action_offsets = [t / fps for t in range(chunk_h)]
    return {
        "observation.images.up": [0.0],
        "observation.images.side": [0.0],
        "observation.state": [0.0],
        "action": action_offsets,
    }


def make_chunk_dataset(
    repo_id: str = "lerobot/svla_so101_pickplace",
    chunk_h: int = CHUNK_H,
    episodes: list[int] | None = None,
    root: str | None = None,
) -> LeRobotDataset:
    """``LeRobotDataset`` with delta_timestamps derived from its fps.

    lerobot needs ``fps`` to convert the chunk's time offsets into
    frame indices, but that ``fps`` itself lives on the dataset's own
    metadata -- so a first, plain construction (no delta_timestamps)
    reads ``fps``, and a second construction applies the derived
    chunking. lerobot 0.5.1 does ship a metadata-only route
    (``LeRobotDatasetMetadata`` exposes ``fps`` without loading frame
    data); we deliberately skip it: both constructions here share the
    local cache (seconds, once downloaded), and one class plus one
    import keeps the teaching surface smaller.

    ``episodes`` (optional) restricts loading to a subset of episode
    indices via lerobot's own ``episodes`` kwarg, so a demo run
    (``configs/demo.yaml``) touches a handful of episodes instead of
    the full dataset. ``None`` (default) loads everything.

    ``root`` (optional) is forwarded to ``LeRobotDataset(root=...)``:
    ``None`` (default) uses lerobot's Hub cache, while a local path
    loads a dataset that already lives on disk without touching the
    Hub -- the seam the end-to-end tests use to point at a tiny toy
    dataset built by ``tests/toy_dataset.py``.
    """
    probe = LeRobotDataset(repo_id, episodes=episodes, root=root)
    delta_timestamps = chunk_delta_timestamps(probe.fps, chunk_h)
    return LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        episodes=episodes,
        root=root,
    )


def make_chunk_loader(
    ds: LeRobotDataset, batch_size: int = 16, num_workers: int = 4
) -> DataLoader:
    """Shuffled DataLoader; batches carry action ``[B, H, D]``."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def prepare_images(batch: dict) -> torch.Tensor:
    """Stack up + side cameras to ``[B, 2, 3, 224, 224]``, ``[0, 1]``.

    Each camera arrives as ``[B, 3, H, W]`` in ``[0, 1]``;
    ``preprocess_image`` resizes to SigLIP's ``224 x 224`` without
    touching the value range.
    """
    up = preprocess_image(batch["observation.images.up"])
    side = preprocess_image(batch["observation.images.side"])
    return torch.stack([up, side], dim=1)
