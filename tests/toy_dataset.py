"""A tiny synthetic LeRobot dataset for the offline end-to-end tests.

The Chapter 4 pipeline (tokenizer -> chunked loader -> fusion adapter ->
action heads -> training -> policy -> eval) is written against the real
``lerobot/svla_so101_pickplace`` dataset -- an 82 MB Hub download the CI
box should not have to fetch. This module writes a *real*
``LeRobotDataset`` to a local directory using lerobot 0.5.1's own
creation API (``LeRobotDataset.create`` + ``add_frame`` + ``save_episode``
+ ``finalize``), at toy dimensions, so the whole pipeline runs unchanged
against it.

Why it mirrors the real schema exactly (so the pipeline code is
untouched):

* The same feature keys the pipeline reads -- two camera views
  ``observation.images.up`` / ``observation.images.side``, a 6-D
  ``observation.state``, and a 6-D ``action`` -- plus the single shared
  ``task`` instruction string every frame carries.
* The cameras are stored as **video** (``use_videos=True``), the way the
  real dataset stores them. That matters for one lerobot 0.5.1 shape
  quirk the pipeline depends on: a video key queried with the single
  ``[0.0]`` chunk offset comes back *squeezed* (``[B, 3, H, W]``), while
  a tabular key keeps the stacked chunk dim (state arrives ``[B, 1, 6]``).
  Storing images as plain image files instead would leave the camera
  keys unsqueezed (``[B, 1, 3, H, W]``) and the pipeline's
  ``preprocess_image`` / ``encode_prefix`` path -- built for the real
  squeezed layout -- would not run unchanged.
* Tiny frames (default ``64 x 64``, far below the real ``480 x 640``):
  the loader hard-resizes to SigLIP's ``224 x 224`` via
  ``prepare_images`` / ``encode_prefix`` regardless, so the small source
  size only keeps the fixture fast to build and decode.

The synthetic content is cheap but *structured*: each joint's action is
a per-episode linear ramp plus small Gaussian noise, so the tokenizer's
Q01/Q99 percentiles are non-degenerate (``lo < hi`` per dimension) and
episode-boundary chunks give ``action_is_pad`` something to mask. Images
are random noise (their content is irrelevant to the pipeline's shape
contract). Everything is seeded, so a build is deterministic and
finishes in a few seconds.

lerobot's parallel video encoder uses a process pool that is unreliable
on macOS (a ``BrokenProcessPool`` on ``save_episode``); we call
``save_episode(parallel_encoding=False)`` to encode each episode inline,
which is plenty fast at this size and robust across platforms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def make_toy_lerobot_dataset(
    root: str | Path,
    repo_id: str = "test/toy_pickplace",
    n_episodes: int = 3,
    frames_per_episode: int = 12,
    fps: int = 30,
    height: int = 64,
    width: int = 64,
    seed: int = 0,
) -> tuple[str, str]:
    """Write a tiny real ``LeRobotDataset`` to ``root``; return ids.

    Builds a dataset whose schema matches ``svla_so101_pickplace`` (two
    video cameras, a 6-D state, a 6-D action, one shared task string) so
    the Chapter 4 pipeline loads it with no code changes. See the module
    docstring for why the cameras are stored as video and the frames are
    tiny.

    ``root`` must not already exist (lerobot's ``create`` makes the
    directory itself and refuses to overwrite). Returns
    ``(repo_id, str(root))`` -- exactly the two arguments
    ``make_chunk_dataset(repo_id=..., root=...)`` needs to reload it
    offline.
    """
    root = str(root)
    cam_feature = {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    features = {
        "observation.images.up": cam_feature,
        "observation.images.side": dict(cam_feature),
        "observation.state": {
            "dtype": "float32", "shape": (6,), "names": None,
        },
        "action": {
            "dtype": "float32", "shape": (6,), "names": None,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        use_videos=True,
    )

    rng = np.random.default_rng(seed)
    for episode in range(n_episodes):
        # Per-joint linear ramp whose slope/offset varies by episode, so
        # the pooled action distribution is non-degenerate and the
        # tokenizer's percentiles are well separated per dimension.
        slope = 0.5 * (episode + 1)
        for t in range(frames_per_episode):
            frac = t / frames_per_episode
            base = slope * frac - 0.5
            action = (
                base + 0.02 * rng.standard_normal(6)
            ).astype(np.float32)
            # State trails the action by a small perturbation.
            state = (
                action + 0.01 * rng.standard_normal(6)
            ).astype(np.float32)
            up = rng.integers(
                0, 256, (height, width, 3), dtype=np.uint8
            )
            side = rng.integers(
                0, 256, (height, width, 3), dtype=np.uint8
            )
            dataset.add_frame(
                {
                    "observation.images.up": up,
                    "observation.images.side": side,
                    "observation.state": state,
                    "action": action,
                    "task": "pick up the cube",
                }
            )
        # Inline (non-parallel) encoding: robust across platforms.
        dataset.save_episode(parallel_encoding=False)

    dataset.finalize()
    return repo_id, root
