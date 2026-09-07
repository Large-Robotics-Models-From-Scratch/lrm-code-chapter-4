"""Drivers behind the section 4.6 and 4.7 figures.

:mod:`ch04.diagnostics` holds pure plotting functions that take arrays.
This module runs a trained head over held-out data to produce those
arrays: the neighbourhood mode recovery of listing 4.9, the per-head joint
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
    plot_neighborhood_mode_recovery,
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


def select_bimodal_anchor(
    states: np.ndarray,
    target_bins: np.ndarray,
    n_neighbors: int = 32,
    max_candidates: int = 256,
) -> dict[str, object]:
    """Choose a local state neighbourhood with two separated target modes.

    The score favors a large gap between two reasonably balanced groups of
    demonstrated bins.  It is deliberately based on held-out targets rather
    than model confidence, so the diagnostic cannot cherry-pick a flattering
    policy output.  The returned neighbour indices can be passed directly to
    :func:`ch04.diagnostics.plot_neighborhood_mode_recovery`.
    """
    values = np.asarray(states, dtype=np.float32)
    targets = np.asarray(target_bins).reshape(-1)
    if values.ndim != 2 or values.shape[0] != targets.shape[0]:
        raise ValueError("states and target_bins must contain aligned rows")
    if values.shape[0] < 4:
        raise ValueError("at least four held-out rows are required")
    neighbors = min(int(n_neighbors), values.shape[0])
    if neighbors < 4:
        raise ValueError("n_neighbors must be at least four")
    candidates = np.unique(
        np.linspace(
            0,
            values.shape[0] - 1,
            min(max_candidates, values.shape[0]),
            dtype=int,
        )
    )
    minimum_group = max(2, int(np.ceil(0.2 * neighbors)))
    best = None
    for anchor in candidates:
        local = nearest_state_neighbors(values, int(anchor), neighbors)
        ordered = np.sort(targets[local].astype(np.float32))
        gaps = np.diff(ordered)
        eligible = np.arange(minimum_group - 1, neighbors - minimum_group)
        if eligible.size == 0:
            continue
        split = int(eligible[np.argmax(gaps[eligible])])
        left, right = ordered[: split + 1], ordered[split + 1 :]
        separation = float(np.median(right) - np.median(left))
        balance = 2.0 * min(left.size, right.size) / neighbors
        spread = float(np.std(left) + np.std(right))
        score = separation * balance / (1.0 + spread)
        record = {
            "anchor_index": int(anchor),
            "neighbor_indices": local,
            "peak_bins": (float(np.median(left)), float(np.median(right))),
            "separation_bins": separation,
            "balance": balance,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = record
    if best is None:
        raise ValueError("could not score any held-out neighbourhood")
    return best


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


@torch.no_grad()
def collect_action_softmaxes(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    """Gather every action cell in one held-out forward-pass sweep.

    This is the efficient source for a representative Figure 4.8 search:
    callers can scan timestep/control cells without rerunning the backbone
    once per candidate.
    """
    states, probabilities, targets, valid = [], [], [], []
    with evaluation_mode(head), evaluation_mode(backbone):
        for batch in _batches(loader, max_batches):
            model_inputs = prepare_batch(batch, stats, device, backbone)
            bins, pad = action_targets(batch, stats, tokenizer, device)
            logits = action_head_logits(head, backbone, model_inputs, bins)
            states.append(model_inputs[2].float().cpu().numpy())
            probabilities.append(logits.softmax(dim=-1).float().cpu().numpy())
            targets.append(bins.cpu().numpy())
            valid.append((~pad).cpu().numpy())
    if not states:
        raise ValueError("the loader produced no valid frames")
    return {
        "states": np.concatenate(states),
        "probabilities": np.concatenate(probabilities),
        "target_bins": np.concatenate(targets),
        "valid": np.concatenate(valid),
    }


def neighborhood_mode_recovery_figure(
    collected: Mapping[str, np.ndarray],
    anchor_index: int,
    n_neighbors: int = 32,
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
    caption += (
        "; neighborhood=proprioception, policy input=full observation"
    )
    return plot_neighborhood_mode_recovery(
        collected["probabilities"][neighbors],
        collected["target_bins"][neighbors],
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
def collect_generated_pairs(
    head,
    backbone,
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    dims: tuple[int, int] = (4, 5),
    timestep: int = 0,
    max_batches: int | None = None,
    strategy: str = "sample",
    temperature: float = 1.0,
) -> np.ndarray:
    """Collect deployed bin pairs across held-out observations.

    Unlike a teacher-forced outer product, this follows each head's real
    generation path.  In particular, the autoregressive second cell is
    conditioned on the model's own earlier sampled cells.
    """
    from ch04.decoding import action_head_bins

    pairs = []
    first, second = dims
    with evaluation_mode(head), evaluation_mode(backbone):
        for batch in _batches(loader, max_batches):
            model_inputs = prepare_batch(batch, stats, device, backbone)
            bins, pad = action_targets(batch, stats, tokenizer, device)
            predicted = action_head_bins(
                head,
                backbone,
                model_inputs,
                strategy=strategy,
                temperature=temperature,
            )
            if not 0 <= timestep < predicted.shape[1]:
                raise IndexError("timestep is outside the action horizon")
            controls = predicted.shape[2]
            if not 0 <= first < controls or not 0 <= second < controls:
                raise IndexError("control is outside the action grid")
            keep = ~pad[:, timestep, first] & ~pad[:, timestep, second]
            if bool(keep.any()):
                pairs.append(
                    predicted[keep, timestep][:, [first, second]]
                    .cpu()
                    .numpy()
                )
    if not pairs:
        raise ValueError("the loader produced no valid generated pairs")
    return np.concatenate(pairs)


def select_pair_mode_support(
    expert_pairs: np.ndarray,
    minimum_group_fraction: float = 0.15,
    minimum_support_fraction: float = 0.05,
) -> dict[str, object]:
    """Derive two per-control modes and supported quadrants from experts.

    The split for each control is the largest central gap, excluding tiny
    edge groups.  A quadrant is supported when at least
    ``minimum_support_fraction`` of expert pairs occupy it; if that leaves
    fewer than two quadrants, the two most common are retained.  The result
    replaces the unrelated fixed bin-128 threshold used by the first draft.
    """
    pairs = np.asarray(expert_pairs, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] < 8:
        raise ValueError("expert_pairs must have shape [N >= 8, 2]")
    if not 0.0 < minimum_group_fraction < 0.5:
        raise ValueError("minimum_group_fraction must lie in (0, 0.5)")
    if not 0.0 <= minimum_support_fraction < 0.5:
        raise ValueError("minimum_support_fraction must lie in [0, 0.5)")

    splits = []
    peaks = []
    minimum_group = max(
        2, int(np.ceil(minimum_group_fraction * len(pairs)))
    )
    for column in range(2):
        ordered = np.sort(pairs[:, column])
        eligible = np.arange(
            minimum_group - 1, len(ordered) - minimum_group
        )
        split_index = int(eligible[np.argmax(np.diff(ordered)[eligible])])
        left = ordered[: split_index + 1]
        right = ordered[split_index + 1 :]
        splits.append(float((left[-1] + right[0]) / 2.0))
        peaks.append((float(np.median(left)), float(np.median(right))))

    high_x = pairs[:, 0] >= splits[0]
    high_y = pairs[:, 1] >= splits[1]
    quadrant_index = high_x.astype(int) * 2 + high_y.astype(int)
    fractions = np.bincount(quadrant_index, minlength=4) / len(pairs)
    supported = fractions >= minimum_support_fraction
    if int(supported.sum()) < 2:
        supported[np.argsort(fractions)[-2:]] = True
    return {
        "splits": tuple(splits),
        "peaks": tuple(peaks),
        "expert_quadrant_fraction": fractions,
        "supported_quadrants": supported,
        "examples": int(len(pairs)),
    }


@torch.no_grad()
def collect_expert_action_grids(
    loader: Iterable[Mapping[str, object]],
    stats,
    tokenizer,
    device: torch.device | str,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    """Collect held-out action-bin grids and their validity mask."""
    bins, valid = [], []
    for batch in _batches(loader, max_batches):
        batch_bins, pad = action_targets(batch, stats, tokenizer, device)
        bins.append(batch_bins.cpu().numpy())
        valid.append((~pad).cpu().numpy())
    if not bins:
        raise ValueError("the loader produced no held-out action grids")
    return {
        "target_bins": np.concatenate(bins),
        "valid": np.concatenate(valid),
    }


def select_coupled_control_pair(
    target_bins: np.ndarray,
    valid: np.ndarray | None = None,
    timestep: int = 0,
    n_bins: int = 256,
    coarse_bins: int = 16,
) -> dict[str, object]:
    """Select the most mutually informative control pair.

    Coarsening prevents sparse 256-way histograms from making every pair
    look artificially dependent.  Selection uses held-out demonstrations
    only; policy predictions are plotted after the pair has been chosen.
    """
    targets = np.asarray(target_bins)
    if targets.ndim != 3:
        raise ValueError("target_bins must have shape [N, H, D]")
    if not 0 <= timestep < targets.shape[1]:
        raise IndexError("timestep is outside the action horizon")
    mask = (
        np.ones_like(targets, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if mask.shape != targets.shape:
        raise ValueError("valid must match target_bins")
    if not 2 <= coarse_bins <= n_bins:
        raise ValueError("coarse_bins must lie in [2, n_bins]")

    best = None
    for first in range(targets.shape[2]):
        for second in range(first + 1, targets.shape[2]):
            keep = mask[:, timestep, first] & mask[:, timestep, second]
            if int(keep.sum()) < 4:
                continue
            x = np.clip(
                targets[keep, timestep, first] * coarse_bins // n_bins,
                0,
                coarse_bins - 1,
            ).astype(int)
            y = np.clip(
                targets[keep, timestep, second] * coarse_bins // n_bins,
                0,
                coarse_bins - 1,
            ).astype(int)
            joint = np.zeros((coarse_bins, coarse_bins), dtype=np.float64)
            np.add.at(joint, (x, y), 1.0)
            joint /= joint.sum()
            px, py = joint.sum(axis=1), joint.sum(axis=0)
            expected = px[:, None] * py[None, :]
            occupied = joint > 0
            mutual_information = float(
                np.sum(joint[occupied] * np.log(
                    joint[occupied] / expected[occupied]
                ))
            )
            entropy_x = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
            entropy_y = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
            denominator = max(min(entropy_x, entropy_y), 1e-12)
            score = mutual_information / denominator
            record = {
                "dims": (first, second),
                "score": score,
                "examples": int(keep.sum()),
            }
            if best is None or score > best["score"]:
                best = record
    if best is None:
        raise ValueError("no control pair has enough valid held-out rows")
    return best


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
    episode_index: list[np.ndarray] = []
    frame_index: list[np.ndarray] = []
    fallback_frame = 0
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
            batch_size = reference.shape[0]
            episodes = batch.get("episode_index")
            frames = batch.get("frame_index")
            if episodes is None:
                episodes = torch.zeros(batch_size, dtype=torch.long)
            if frames is None:
                frames = torch.arange(
                    fallback_frame, fallback_frame + batch_size
                )
            episode_index.append(
                torch.as_tensor(episodes).reshape(-1).cpu().numpy()
            )
            frame_index.append(
                torch.as_tensor(frames).reshape(-1).cpu().numpy()
            )
            fallback_frame += batch_size
    if not predicted:
        raise ValueError("the loader produced no batches")
    return {
        "predicted": np.concatenate(predicted),
        "expert": np.concatenate(expert),
        "valid": np.concatenate(valid),
        "episode_index": np.concatenate(episode_index),
        "frame_index": np.concatenate(frame_index),
    }


def select_representative_open_loop_window(
    trace: Mapping[str, np.ndarray],
    max_steps: int = 180,
    scale: np.ndarray | None = None,
) -> dict[str, np.ndarray | int | float]:
    """Choose one contiguous, high-motion held-out episode window.

    Plotting the first few batches can splice episodes together or show only
    the stationary lead-in.  This selector searches contiguous runs within
    each episode and returns the ``max_steps`` window with the largest mean
    frame-to-frame expert change.
    """
    predicted = np.asarray(trace["predicted"])
    expert = np.asarray(trace["expert"])
    valid = np.asarray(trace.get("valid", np.ones(expert.shape[0], bool)))
    episodes = np.asarray(
        trace.get("episode_index", np.zeros(expert.shape[0], dtype=int))
    ).reshape(-1)
    frames = np.asarray(
        trace.get("frame_index", np.arange(expert.shape[0]))
    ).reshape(-1)
    if predicted.shape != expert.shape or predicted.ndim != 2:
        raise ValueError("trace trajectories must match with shape [T, D]")
    if not (
        valid.shape == episodes.shape == frames.shape == (expert.shape[0],)
    ):
        raise ValueError(
            "trace metadata must contain one value per timestep"
        )
    if max_steps < 2:
        raise ValueError("max_steps must be at least two")

    if scale is None:
        motion_scale = np.ones(expert.shape[1], dtype=np.float64)
    else:
        motion_scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        if motion_scale.shape != (expert.shape[1],):
            raise ValueError("scale must contain one value per control")
        motion_scale = np.maximum(np.abs(motion_scale), 1e-8)

    best = None
    for episode in np.unique(episodes[valid]):
        indices = np.flatnonzero(valid & (episodes == episode))
        indices = indices[np.argsort(frames[indices], kind="stable")]
        discontinuities = np.flatnonzero(np.diff(frames[indices]) != 1) + 1
        for run in np.split(indices, discontinuities):
            if run.size < 2:
                continue
            motion = np.abs(
                np.diff(expert[run], axis=0) / motion_scale
            ).mean(axis=1)
            window = min(max_steps, run.size)
            if run.size > window:
                # First point has no preceding difference, hence window - 1.
                sums = np.convolve(
                    motion, np.ones(window - 1), mode="valid"
                )
                start = int(np.argmax(sums))
            else:
                start = 0
            chosen = run[start : start + window]
            score = float(
                np.abs(
                    np.diff(expert[chosen], axis=0) / motion_scale
                ).mean()
            )
            record = {
                "indices": chosen,
                "episode_index": int(episode),
                "start_frame": int(frames[chosen[0]]),
                "end_frame": int(frames[chosen[-1]]),
                "motion_score": score,
            }
            if best is None or score > best["motion_score"]:
                best = record
    if best is None:
        raise ValueError("no held-out episode has two valid timesteps")

    chosen = best.pop("indices")
    selected: dict[str, np.ndarray | int | float] = {
        key: np.asarray(value)[chosen]
        for key, value in trace.items()
        if np.asarray(value).ndim >= 1
        and np.asarray(value).shape[0] == expert.shape[0]
    }
    selected.update(best)
    selected["indices"] = chosen
    return selected


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


def measure_inference_flops(
    heads: Mapping[str, object],
    backbone,
    model_inputs,
    example: int = 0,
) -> dict[str, dict[str, float]]:
    """Measured forward-pass FLOPs for each head's deployment path.

    Every head is run through :func:`ch04.decoding.action_head_bins`, so the
    count covers what deployment actually executes: one pass for the two
    parallel heads, and the prefill plus every cached decode step for the
    autoregressive head. Counting one example keeps the number independent
    of the evaluation batch size.

    FLOPs severely understate the autoregressive head, because a FLOP count
    is blind to the dependency graph: one wide pass is a compute-bound GEMM,
    while 96 single-token steps are memory-bound GEMVs that cannot overlap.
    Measured on the real Chapter 3 backbone (SmolLM2-135M, 399-token
    prefix), the autoregressive head costs 1.11x the arithmetic of the
    bidirectional head but 17x its GPU latency; the shared prefill dominates
    both FLOP counts and hides the difference. In small configurations the
    gap can even reverse sign, since cached decoding recomputes no
    projections and skips the masked half of the causal attention matrix.
    Use :func:`measure_inference_latency` for the trade-off axis and treat
    ``flops`` as a lower bound recorded alongside ``serial_steps``.
    """
    from torch.utils.flop_counter import FlopCounterMode

    from ch04.decoding import action_head_bins

    batch_size = model_inputs[0].shape[0]
    if not 0 <= example < batch_size:
        raise IndexError("example is outside the batch")
    selected = tuple(
        value[example : example + 1] for value in model_inputs
    )

    measured: dict[str, dict[str, float]] = {}
    for name, head in heads.items():
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            action_head_bins(head, backbone, selected, strategy="argmax")
        measured[name] = {
            "flops": float(counter.get_total_flops()),
            "serial_steps": int(getattr(head, "grid", 1))
            if hasattr(head, "generate")
            else 1,
        }
    return measured


def measure_inference_latency(
    heads: Mapping[str, object],
    backbone,
    model_inputs,
    repeats: int = 20,
    warmup: int = 3,
    example: int = 0,
) -> dict[str, dict[str, float]]:
    """Wall-clock inference latency for each head's deployment path.

    This is the faithful cost axis for the section 4.6 trade-off. A FLOP
    count is blind to the dependency graph, so cached serial decoding scores
    *lower* than one wide pass while running far slower;
    :func:`measure_inference_flops` documents that inversion. Latency
    measures what the arm actually waits for.

    The number is device- and implementation-specific, which is the price of
    faithfulness: report it with the device recorded here. Timing runs after
    ``warmup`` untimed passes, synchronizes CUDA around each repeat so that
    queued kernels are not mistaken for finished work, and returns the
    median with a 10th/90th-percentile spread rather than a mean, which one
    scheduler hiccup can dominate.
    """
    import time

    from ch04.decoding import action_head_bins

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must not be negative")
    batch_size = model_inputs[0].shape[0]
    if not 0 <= example < batch_size:
        raise IndexError("example is outside the batch")
    selected = tuple(
        value[example : example + 1] for value in model_inputs
    )
    device = torch.device(selected[0].device)
    # Both CUDA and MPS dispatch asynchronously: without a barrier the
    # timer measures how fast work was *queued*, which would make the 96
    # dependent decode steps look nearly free.
    if device.type == "cuda":
        synchronize = torch.cuda.synchronize
    elif device.type == "mps":
        synchronize = torch.mps.synchronize
    else:
        def synchronize():
            return None

    if device.type == "cuda":
        device_label = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        device_label = "Apple Metal Performance Shaders"
    else:
        device_label = str(device)

    measured: dict[str, dict[str, float]] = {}
    for name, head in heads.items():
        def run(head=head):
            action_head_bins(head, backbone, selected, strategy="argmax")

        with torch.no_grad():
            for _ in range(warmup):
                run()
            synchronize()
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                run()
                synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
        ordered = np.sort(np.asarray(samples, dtype=np.float64))
        measured[name] = {
            "latency_ms": float(np.median(ordered)),
            "p10_ms": float(ordered[0])
            if repeats < 10
            else float(np.percentile(ordered, 10)),
            "p90_ms": float(ordered[-1])
            if repeats < 10
            else float(np.percentile(ordered, 90)),
            "serial_steps": int(getattr(head, "grid", 1))
            if hasattr(head, "generate")
            else 1,
            "device": device_label,
        }
    return measured
