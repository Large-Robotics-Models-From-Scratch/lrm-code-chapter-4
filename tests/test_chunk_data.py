"""Tests for ch04.chunk_data.

Unit tests exercise ``chunk_delta_timestamps`` and ``prepare_images``
with fake tensors -- no network, no real dataset. The integration test
loads the real ``lerobot/svla_so101_pickplace`` dataset and asserts
the chunked batch shapes, camera keys, and the episode-boundary
``action_is_pad`` mask lerobot emits (see chunk_data.py's module
docstring for what that mask means).
"""

import pytest
import torch

from ch04 import CHUNK_H
from ch04.chunk_data import (
    chunk_delta_timestamps,
    make_chunk_dataset,
    make_chunk_loader,
    prepare_images,
)

FPS = 30


def test_chunk_delta_timestamps_action_offsets():
    dt = chunk_delta_timestamps(FPS, chunk_h=16)
    expected = [t / FPS for t in range(16)]
    assert dt["action"] == expected


def test_chunk_delta_timestamps_observation_keys():
    dt = chunk_delta_timestamps(FPS, chunk_h=16)
    assert dt["observation.images.up"] == [0.0]
    assert dt["observation.images.side"] == [0.0]
    assert dt["observation.state"] == [0.0]


def test_chunk_delta_timestamps_default_chunk_h():
    dt = chunk_delta_timestamps(FPS)
    assert len(dt["action"]) == CHUNK_H


def test_make_chunk_dataset_passes_episodes(monkeypatch):
    """``episodes`` reaches both LeRobotDataset constructions.

    Unit-level (no network): a recording fake stands in for
    ``LeRobotDataset`` and captures the ``episodes`` kwarg from both the
    fps-probe construction and the chunked one.
    """
    import ch04.chunk_data as chunk_data

    calls = []

    class _FakeDataset:
        def __init__(self, repo_id, delta_timestamps=None, episodes=None):
            calls.append(episodes)
            self.fps = 30

    monkeypatch.setattr(chunk_data, "LeRobotDataset", _FakeDataset)
    chunk_data.make_chunk_dataset(episodes=[0, 1])

    assert calls == [[0, 1], [0, 1]]


def test_make_chunk_dataset_episodes_defaults_to_none(monkeypatch):
    import ch04.chunk_data as chunk_data

    calls = []

    class _FakeDataset:
        def __init__(self, repo_id, delta_timestamps=None, episodes=None):
            calls.append(episodes)
            self.fps = 30

    monkeypatch.setattr(chunk_data, "LeRobotDataset", _FakeDataset)
    chunk_data.make_chunk_dataset()

    assert calls == [None, None]


def test_prepare_images_shape_and_range():
    batch = {
        "observation.images.up": torch.full(
            (2, 3, 480, 640), 0.2
        ),
        "observation.images.side": torch.full(
            (2, 3, 480, 640), 0.8
        ),
    }
    images = prepare_images(batch)
    assert images.shape == (2, 2, 3, 224, 224)
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_prepare_images_camera_order_preserved():
    batch = {
        "observation.images.up": torch.full(
            (2, 3, 480, 640), 0.2
        ),
        "observation.images.side": torch.full(
            (2, 3, 480, 640), 0.8
        ),
    }
    images = prepare_images(batch)
    assert images[:, 0].mean().item() == pytest.approx(
        0.2, abs=1e-5
    )
    assert images[:, 1].mean().item() == pytest.approx(
        0.8, abs=1e-5
    )


@pytest.mark.integration
@pytest.mark.slow
def test_integration_real_dataset_chunked_batch():
    """Real dataset: chunked action, both cameras, pad mask present.

    Also documents what was found for ``action_is_pad``: lerobot's
    ``dataset_reader`` emits a ``f"{key}_is_pad"`` bool mask for every
    key that has ``delta_timestamps`` set (not just ``action``), so
    ``observation.state``/camera keys get a (trivial, length-1)
    ``*_is_pad`` too. ``action_is_pad`` has shape ``[B, CHUNK_H]`` and
    is ``True`` at steps lerobot clamped to an episode boundary.
    """
    ds = make_chunk_dataset(episodes=[0, 1])
    assert ds.fps == 30  # the real dataset rate, not the book's 50
    assert ds.num_episodes == 2  # episode subset passthrough
    assert len(ds.delta_timestamps["action"]) == CHUNK_H
    assert ds.delta_timestamps["action"][-1] == pytest.approx(
        15 / 30
    )

    loader = make_chunk_loader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))

    assert batch["action"].shape == (2, CHUNK_H, 6)
    assert "observation.images.up" in batch
    assert "observation.images.side" in batch
    # lerobot 0.5.1 asymmetry: video keys with a single [0.0] offset
    # come back squeezed ([B, 3, 480, 640]) but tabular keys keep the
    # stacked chunk dim -- state is [B, 1, 6], NOT [B, 6]. Task 7 must
    # squeeze dim 1 before feeding ch3's backbone (expects [B, 6]).
    assert batch["observation.images.up"].shape == (2, 3, 480, 640)
    assert batch["observation.state"].shape == (2, 1, 6)

    assert "action_is_pad" in batch
    assert batch["action_is_pad"].shape == (2, CHUNK_H)
    assert batch["action_is_pad"].dtype == torch.bool
