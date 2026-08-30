"""Drivers behind the section 4.6 and 4.7 figures.

:mod:`ch04.diagnostics` holds pure plotting functions that take arrays.
This module runs a trained head over held-out data to produce those
arrays: the neighbourhood softmaxes of listing 4.9, the per-head joint
samples of figure 4.9, the episode trace of figure 4.11, and the chunk
stream the execution schedules of figure 4.10 consume.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch

from ch04.data import action_targets, prepare_batch
from ch04.decoding import (
    decode_action_chunk,
    evaluation_mode,
    sample_action_grids,
)
from ch04.diagnostics import (
    joint_logit_mass,
    joint_logit_mismatch_rate,
    joint_mismatch_rate,
    nearest_state_neighbors,
    plot_neighbor_softmaxes,
)
from ch04.train import action_head_logits


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch so a reported figure is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(loader: Iterable, max_batches: int | None):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            return
        yield batch


@torch.no_grad()
def collect_cell_softmaxes(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    timestep: int = 0,
    control: int = 0,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    """Gather one grid cell's softmax over a held-out loader.

    Returns the normalized proprioception of every frame together with
    the predicted distribution and the demonstrated bin for the chosen
    ``(timestep, control)`` cell. Padded frames are dropped.
    """
    states: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with evaluation_mode(head), evaluation_mode(backbone):
        for batch in _batches(loader, max_batches):
            model_inputs = prepare_batch(batch, stats, device, backbone)
            bins, pad = action_targets(batch, stats, tokenizer, device)
            if not 0 <= timestep < bins.shape[1]:
                raise IndexError("timestep is outside the action horizon")
            if not 0 <= control < bins.shape[2]:
                raise IndexError("control is outside the action grid")
            logits = action_head_logits(
                head, backbone, model_inputs, bins
            )
            keep = ~pad[:, timestep, control]
            if not bool(keep.any()):
                continue
            cell = logits[:, timestep, control][keep]
            states.append(model_inputs[2][keep].float().cpu().numpy())
            probabilities.append(
                cell.softmax(dim=-1).float().cpu().numpy()
            )
            targets.append(bins[:, timestep, control][keep].cpu().numpy())
    if not states:
        raise ValueError("the loader produced no valid frames")
    return {
        "states": np.concatenate(states),
        "probabilities": np.concatenate(probabilities),
        "target_bins": np.concatenate(targets),
    }


def neighborhood_softmax_figure(
    collected: Mapping[str, np.ndarray],
    anchor_index: int,
    n_neighbors: int = 32,
    n_curves: int = 6,
    checkpoint: str = "unspecified",
    seed: int = 0,
):
    """Figure 4.8 from :func:`collect_cell_softmaxes` output.

    The caption records the checkpoint, anchor index, neighbour count, and
    seed, which section 4.6.1 requires alongside any reported result.
    """
    neighbors = nearest_state_neighbors(
        collected["states"], anchor_index, n_neighbors
    )
    # Keep the caption identifying but short: a full path stretches the
    # figure until the panels are unreadable.
    path = Path(checkpoint)
    label = (
        "/".join(path.parts[-2:]) if len(path.parts) > 1 else str(path)
    )
    caption = (
        f"checkpoint={label} anchor={anchor_index} "
        f"neighbors={n_neighbors} seed={seed}"
    )
    return plot_neighbor_softmaxes(
        collected["probabilities"][neighbors],
        collected["target_bins"][neighbors],
        n_curves=n_curves,
        caption=caption,
    )


@torch.no_grad()
def collect_joint_logit_mass(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    dims: tuple[int, int] = (4, 5),
    timestep: int = 0,
    max_batches: int | None = None,
) -> np.ndarray:
    """Average a head's two-cell probability mass over held-out frames.

    This uses softmax probabilities directly rather than estimating them
    with sampled action grids. For autoregressive heads the logits are
    teacher-forced, so the later cell is conditioned on each frame's
    demonstrated preceding cells.
    """
    total = None
    examples = 0
    with evaluation_mode(head), evaluation_mode(backbone):
        for batch in _batches(loader, max_batches):
            model_inputs = prepare_batch(batch, stats, device, backbone)
            bins, pad = action_targets(batch, stats, tokenizer, device)
            logits = action_head_logits(
                head, backbone, model_inputs, bins
            )
            keep = ~pad[:, timestep, dims[0]] & ~pad[:, timestep, dims[1]]
            if not bool(keep.any()):
                continue
            mass = joint_logit_mass(logits[keep], dims, timestep)
            count = int(keep.sum())
            total = mass * count if total is None else total + mass * count
            examples += count
    if total is None or examples == 0:
        raise ValueError("the loader produced no valid joint-logit frames")
    return total / examples


@torch.no_grad()
def collect_expert_pairs(
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    dims: tuple[int, int] = (4, 5),
    timestep: int = 0,
    max_batches: int | None = None,
) -> np.ndarray:
    """Collect valid held-out target pairs for the logit comparison."""
    pairs = []
    for batch in _batches(loader, max_batches):
        bins, pad = action_targets(batch, stats, tokenizer, device)
        if not 0 <= timestep < bins.shape[1]:
            raise IndexError("timestep is outside the action horizon")
        first, second = dims
        controls = bins.shape[2]
        if not 0 <= first < controls or not 0 <= second < controls:
            raise IndexError("control is outside the action grid")
        keep = ~pad[:, timestep, first] & ~pad[:, timestep, second]
        if bool(keep.any()):
            pairs.append(
                bins[keep, timestep][:, [first, second]].cpu().numpy()
            )
    if not pairs:
        raise ValueError("the loader produced no valid expert pairs")
    return np.concatenate(pairs)


@torch.no_grad()
def joint_mismatch_samples(
    heads: Mapping[str, object],
    backbone,
    model_inputs,
    dims: tuple[int, int] = (4, 5),
    timestep: int = 0,
    n_samples: int = 512,
    example: int = 0,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Figure 4.9 draws: one ``[N, 2]`` bin-pair array per head.

    Each head samples through its own inference path, so the
    autoregressive pairs carry the conditioning the parallel heads lack.
    """
    first, second = dims
    samples: dict[str, np.ndarray] = {}
    for name, head in heads.items():
        grids = sample_action_grids(
            head,
            backbone,
            model_inputs,
            n_samples=n_samples,
            example=example,
            temperature=temperature,
        )
        if not 0 <= timestep < grids.shape[1]:
            raise IndexError("timestep is outside the action horizon")
        controls = grids.shape[2]
        if not 0 <= first < controls or not 0 <= second < controls:
            raise IndexError("control is outside the action grid")
        pair = grids[:, timestep][:, [first, second]]
        samples[name] = pair.cpu().numpy()
    return samples


@torch.no_grad()
def sampled_grids_by_head(
    heads: Mapping[str, object],
    backbone,
    model_inputs,
    n_samples: int = 12,
    example: int = 0,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Per-head ``[N, H, D]`` draws for the temporal-trace comparison."""
    return {
        name: sample_action_grids(
            head,
            backbone,
            model_inputs,
            n_samples=n_samples,
            example=example,
            temperature=temperature,
        ).cpu().numpy()
        for name, head in heads.items()
    }


def mismatch_rates(
    samples_by_head: Mapping[str, np.ndarray],
    split_first: int,
    split_second: int,
) -> dict[str, float]:
    """Off-diagonal quadrant rate per head: figure 4.9's headline number."""
    return {
        name: joint_mismatch_rate(
            pairs[:, 0], pairs[:, 1], split_first, split_second
        )
        for name, pairs in samples_by_head.items()
    }


def logit_mismatch_rates(
    mass_by_head: Mapping[str, np.ndarray],
    split_first: int,
    split_second: int,
) -> dict[str, float]:
    """Off-diagonal probability per head, computed without sampling."""
    return {
        name: joint_logit_mismatch_rate(
            mass, split_first, split_second
        )
        for name, mass in mass_by_head.items()
    }


@torch.no_grad()
def open_loop_episode_trace(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    tokenizer,
    stats,
    device: torch.device | str,
    max_batches: int | None = None,
    strategy: str = "argmax",
) -> dict[str, np.ndarray]:
    """Figure 4.11 data: the first decoded command of every held-out frame.

    The loader must be unshuffled so consecutive frames form a trajectory.
    Each frame contributes the first row of its decoded chunk, which is the
    receding-horizon command, next to the expert command for that instant.
    """
    predicted: list[np.ndarray] = []
    expert: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    with evaluation_mode(head), evaluation_mode(backbone):
        for batch in _batches(loader, max_batches):
            model_inputs = prepare_batch(batch, stats, device, backbone)
            chunk = decode_action_chunk(
                head,
                backbone,
                model_inputs,
                tokenizer,
                stats,
                strategy=strategy,
            ).cpu()
            reference = torch.as_tensor(batch["action"]).float().cpu()
            pad = torch.as_tensor(
                batch.get(
                    "action_is_pad",
                    torch.zeros(reference.shape[:2], dtype=torch.bool),
                ),
                dtype=torch.bool,
            ).cpu()
            predicted.append(chunk[:, 0].numpy())
            expert.append(reference[:, 0].numpy())
            valid.append((~pad[:, 0]).numpy())
    if not predicted:
        raise ValueError("the loader produced no batches")
    return {
        "predicted": np.concatenate(predicted),
        "expert": np.concatenate(expert),
        "valid": np.concatenate(valid),
    }


@torch.no_grad()
def decoded_chunk_stream(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    tokenizer,
    stats,
    device: torch.device | str,
    max_batches: int | None = None,
    strategy: str = "argmax",
) -> list[torch.Tensor]:
    """One decoded ``[H, D]`` chunk per held-out frame, in time order.

    This is the input the section 4.7.2 schedules consume: each entry is
    the chunk the policy would have decoded at that control timestep.
    """
    chunks: list[torch.Tensor] = []
    with evaluation_mode(head), evaluation_mode(backbone):
        for batch in _batches(loader, max_batches):
            model_inputs = prepare_batch(batch, stats, device, backbone)
            decoded = decode_action_chunk(
                head,
                backbone,
                model_inputs,
                tokenizer,
                stats,
                strategy=strategy,
            ).cpu()
            chunks.extend(decoded.unbind(dim=0))
    if not chunks:
        raise ValueError("the loader produced no batches")
    return chunks


def expert_pairs_from_batch(
    batch: Mapping[str, object],
    stats,
    tokenizer,
    device: torch.device | str,
    dims: tuple[int, int] = (4, 5),
    timestep: int = 0,
) -> np.ndarray:
    """Encode the demonstrated ``(control, control)`` pairs of one batch."""
    bins, pad = action_targets(batch, stats, tokenizer, device)
    keep = ~pad[:, timestep, dims[0]]
    pairs = bins[:, timestep][:, list(dims)][keep]
    if pairs.shape[0] == 0:
        raise ValueError("the batch contains no valid expert pairs")
    return pairs.cpu().numpy()
