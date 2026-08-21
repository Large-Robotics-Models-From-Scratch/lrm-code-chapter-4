"""LeRobot action chunks and Chapter 3 batch preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from ch04.constants import ACTION_HORIZON

DEFAULT_DATASET_ID = "lerobot/svla_so101_pickplace"
CAMERA_KEYS = (
    "observation.images.up",
    "observation.images.side",
)


def make_chunked_dataloader(
    dataset_id: str = DEFAULT_DATASET_ID,
    horizon: int = ACTION_HORIZON,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """Return a LeRobot loader with ``H`` future actions and Ch2 stats.

    Imports are local so tokenization/head-only use does not require the
    optional dataset stack.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    from ch02 import make_pickplace_dataloader
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    _, stats = make_pickplace_dataloader(dataset_id, batch_size=1)
    metadata = LeRobotDatasetMetadata(dataset_id)
    loader = _make_chunk_loader(
        dataset_id,
        metadata.fps,
        horizon,
        batch_size,
        shuffle,
        num_workers,
    )
    return loader, stats


def make_chunked_dataloaders(
    dataset_id: str = DEFAULT_DATASET_ID,
    horizon: int = ACTION_HORIZON,
    batch_size: int = 32,
    validation_fraction: float = 0.1,
    seed: int | None = None,
    num_workers: int = 0,
):
    """Return episode-disjoint loaders and training-episode statistics."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    metadata = LeRobotDatasetMetadata(dataset_id)
    if metadata.total_episodes < 2:
        raise ValueError(
            "episode-level validation needs at least two episodes"
        )
    episodes = list(range(metadata.total_episodes))
    if seed is not None:
        import random

        random.Random(seed).shuffle(episodes)
    n_validation = max(
        1,
        min(
            metadata.total_episodes - 1,
            round(len(episodes) * validation_fraction),
        ),
    )
    validation_episodes = sorted(episodes[-n_validation:])
    train_episodes = sorted(episodes[:-n_validation])
    train_frames = LeRobotDataset(dataset_id, episodes=train_episodes)
    stats = _compute_stats_from_frame_columns(train_frames)
    train_loader = _make_chunk_loader(
        dataset_id,
        metadata.fps,
        horizon,
        batch_size,
        True,
        num_workers,
        episodes=train_episodes,
    )
    validation_loader = _make_chunk_loader(
        dataset_id,
        metadata.fps,
        horizon,
        batch_size,
        False,
        num_workers,
        episodes=validation_episodes,
    )
    return train_loader, validation_loader, stats


def _make_chunk_loader(
    dataset_id: str,
    fps: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    episodes: list[int] | None = None,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        dataset_id,
        episodes=episodes,
        delta_timestamps={
            "action": [step / fps for step in range(horizon)]
        },
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


def collect_normalized_actions(
    loader,
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    max_batches: int | None = None,
) -> np.ndarray:
    """Collect valid normalized actions from a chunk loader.

    This is intended for tokenizer calibration on training episodes only.
    With ``max_batches=None`` and a LeRobot loader, it reads the underlying
    Arrow action column once, avoiding both repeated chunk timesteps and
    video decoding. An integer bound keeps the loader-based smoke path.
    """
    frame_dataset = getattr(loader.dataset, "hf_dataset", None)
    if max_batches is None and frame_dataset is not None:
        actions = _stack_frame_column(frame_dataset, "action")
        normalized = normalize_from_stats(actions, stats, "action")
        return normalized.cpu().numpy()

    collected = []
    for index, batch in enumerate(loader):
        actions = torch.as_tensor(batch["action"]).float()
        normalized = normalize_from_stats(actions, stats, "action")
        pad = torch.as_tensor(
            batch.get(
                "action_is_pad",
                torch.zeros(actions.shape[:2], dtype=torch.bool),
            ),
            dtype=torch.bool,
        )
        collected.append(normalized[~pad].cpu().numpy())
        if max_batches is not None and index + 1 >= max_batches:
            break
    if not collected:
        raise ValueError("loader produced no actions for tokenizer fitting")
    return np.concatenate(collected, axis=0)


def _stack_frame_column(dataset, key: str) -> torch.Tensor:
    """Read an Arrow control column without touching video streams."""
    values = dataset[key]
    if isinstance(values, torch.Tensor):
        return values.float()
    return torch.stack(
        [torch.as_tensor(value).float() for value in values]
    )


def _compute_stats_from_frame_columns(dataset):
    """Compute training-only Ch2 statistics without decoding images."""
    frame_dataset = getattr(dataset, "hf_dataset", None)
    if frame_dataset is None:
        from ch02 import compute_stats

        return compute_stats(dataset)
    result = {}
    for key in ("observation.state", "action"):
        values = _stack_frame_column(frame_dataset, key)
        result[key] = {
            "mean": values.mean(0),
            "std": values.std(0),
            "min": values.min(0).values,
            "max": values.max(0).values,
        }
    return result


def normalize_from_stats(
    values: torch.Tensor,
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    key: str,
) -> torch.Tensor:
    """Chapter 2 z-score normalization while preserving device and dtype.

    Chapter 2's public helper is numerically equivalent, but its stats may
    remain on CPU or use a different dtype.  Chapter 4 moves them alongside
    ``values`` so action-head training stays device-local.
    """
    mean = torch.as_tensor(
        stats[key]["mean"], device=values.device, dtype=values.dtype
    )
    std = torch.as_tensor(
        stats[key]["std"], device=values.device, dtype=values.dtype
    )
    return (values - mean) / (std + 1e-8)


def denormalize_from_stats(
    values: torch.Tensor,
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    key: str,
) -> torch.Tensor:
    """Invert :func:`normalize_from_stats`."""
    mean = torch.as_tensor(
        stats[key]["mean"], device=values.device, dtype=values.dtype
    )
    std = torch.as_tensor(
        stats[key]["std"], device=values.device, dtype=values.dtype
    )
    return values * (std + 1e-8) + mean


def prepare_batch(
    batch: Mapping[str, object],
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device | str,
    backbone,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the four positional inputs accepted by Chapter 4 heads.

    Chapter 3 now owns image resizing and multimodal sequence assembly.
    Chapter 4 therefore supplies raw two-camera frames, padded text ids,
    normalized state, and the text validity mask.
    """
    camera_tensors = []
    for key in CAMERA_KEYS:
        if key not in batch:
            raise KeyError(f"batch is missing camera key {key!r}")
        camera = torch.as_tensor(batch[key])
        if camera.dtype == torch.uint8:
            camera = camera.float() / 255.0
        else:
            camera = camera.float()
        camera_tensors.append(camera.clamp(0.0, 1.0))
    images = torch.stack(camera_tensors, dim=1).to(device)

    raw_state = torch.as_tensor(batch["observation.state"]).float()
    state = normalize_from_stats(raw_state, stats, "observation.state")
    state = state.to(device)

    tasks = _task_strings(batch.get("task"), images.shape[0])
    text = backbone.tokenizer(
        tasks,
        padding=True,
        return_tensors="pt",
    )
    input_ids = text.input_ids.to(device)
    text_attention_mask = text.attention_mask.bool().to(device)
    return images, input_ids, state, text_attention_mask


def action_targets(
    batch: Mapping[str, object],
    stats: Mapping[str, Mapping[str, torch.Tensor]],
    tokenizer,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened action bins and the matching token pad mask."""
    actions = torch.as_tensor(batch["action"]).float()
    if actions.ndim != 3:
        raise ValueError("chunked actions must have shape [B, H, D]")
    normalized = normalize_from_stats(actions, stats, "action")
    bins_np = tokenizer.encode(normalized.cpu().numpy())
    bins = torch.from_numpy(bins_np).reshape(actions.shape[0], -1)

    timestep_pad = batch.get("action_is_pad")
    if timestep_pad is None:
        timestep_pad = torch.zeros(actions.shape[:2], dtype=torch.bool)
    timestep_pad = torch.as_tensor(timestep_pad, dtype=torch.bool)
    pad = timestep_pad.repeat_interleave(actions.shape[-1], dim=1)
    return bins.to(device), pad.to(device)


def _task_strings(tasks: object, batch_size: int) -> list[str]:
    if tasks is None:
        return ["pick up the object"] * batch_size
    if isinstance(tasks, str):
        return [tasks] * batch_size
    if isinstance(tasks, Sequence) and len(tasks) == batch_size:
        return [str(task) for task in tasks]
    raise ValueError("task must be one string per batch row")
