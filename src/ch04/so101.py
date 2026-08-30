"""Export and safely replay one Chapter 4 action chunk on an SO-101.

Colab produces the policy chunk, but hardware playback belongs on the
computer physically connected to a calibrated follower arm. The CLI is a
dry run unless ``--execute`` is supplied and requires a second explicit
confirmation before enabling the motors.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ch04.constants import ACTION_DIM, ACTION_HORIZON

SO101_ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def _as_chunk(actions) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    expected = (ACTION_HORIZON, ACTION_DIM)
    if values.shape != expected:
        raise ValueError(f"action chunk must have shape {expected}")
    if not np.isfinite(values).all():
        raise ValueError("action chunk contains NaN or infinite values")
    return values


def export_action_chunk(
    path: str | Path,
    actions,
    *,
    fps: int = 30,
    action_min=None,
    action_max=None,
    source: str = "Chapter 4 parallel policy",
) -> Path:
    """Write one denormalized ``[16, 6]`` chunk for local playback."""
    chunk = _as_chunk(actions)
    if fps < 1:
        raise ValueError("fps must be positive")
    lower = (
        np.full(ACTION_DIM, np.nan, dtype=np.float32)
        if action_min is None
        else np.asarray(action_min, dtype=np.float32).reshape(ACTION_DIM)
    )
    upper = (
        np.full(ACTION_DIM, np.nan, dtype=np.float32)
        if action_max is None
        else np.asarray(action_max, dtype=np.float32).reshape(ACTION_DIM)
    )
    if np.isfinite(lower).all() and np.any(chunk < lower):
        raise ValueError("action chunk falls below the training-data range")
    if np.isfinite(upper).all() and np.any(chunk > upper):
        raise ValueError("action chunk exceeds the training-data range")
    destination = Path(path)
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        actions=chunk,
        action_names=np.asarray(SO101_ACTION_NAMES),
        fps=np.asarray(fps, dtype=np.int64),
        action_min=lower,
        action_max=upper,
        source=np.asarray(source),
    )
    return destination


def load_action_chunk(path: str | Path) -> dict[str, object]:
    """Load and validate a chunk exported by :func:`export_action_chunk`."""
    with np.load(path, allow_pickle=False) as payload:
        required = {"actions", "action_names", "fps"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"chunk file is missing {sorted(missing)}")
        actions = _as_chunk(payload["actions"])
        names = tuple(str(name) for name in payload["action_names"])
        fps = int(payload["fps"])
        lower = (
            payload["action_min"]
            if "action_min" in payload.files
            else None
        )
        upper = (
            payload["action_max"]
            if "action_max" in payload.files
            else None
        )
        source = (
            str(payload["source"])
            if "source" in payload.files
            else "unknown"
        )
    if names != SO101_ACTION_NAMES:
        raise ValueError("chunk action names do not match the SO-101 order")
    if fps < 1:
        raise ValueError("chunk fps must be positive")
    if lower is not None and np.isfinite(lower).all():
        if np.any(actions < lower):
            raise ValueError("chunk falls below its recorded safety range")
    if upper is not None and np.isfinite(upper).all():
        if np.any(actions > upper):
            raise ValueError("chunk exceeds its recorded safety range")
    return {
        "actions": actions,
        "action_names": names,
        "fps": fps,
        "action_min": lower,
        "action_max": upper,
        "source": source,
    }


def _action_dict(row: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(SO101_ACTION_NAMES, row, strict=True)
    }


def replay_action_chunk(robot, actions, fps: int = 30) -> None:
    """Send exactly one chunk to a connected LeRobot follower at ``fps``."""
    chunk = _as_chunk(actions)
    if fps < 1:
        raise ValueError("fps must be positive")
    try:
        from lerobot.utils.robot_utils import precise_sleep
    except ImportError:
        precise_sleep = time.sleep
    for row in chunk:
        started = time.perf_counter()
        robot.send_action(_action_dict(row))
        precise_sleep(max(1.0 / fps - (time.perf_counter() - started), 0.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ch04-so101-replay",
        description="Preview or replay one exported Chapter 4 chunk.",
    )
    parser.add_argument("chunk", help=".npz file exported by the Colab")
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", default="chapter4_so101")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="LeRobot per-command joint-position safety cap",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="connect and move the robot; omission is a dry run",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive EXECUTE confirmation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.max_relative_target <= 0:
        raise SystemExit("--max-relative-target must be positive")
    payload = load_action_chunk(arguments.chunk)
    actions = payload["actions"]
    print(
        f"source={payload['source']} shape={actions.shape} "
        f"fps={payload['fps']} "
        f"duration={len(actions) / payload['fps']:.2f}s"
    )
    print("first command:", _action_dict(actions[0]))
    print("last command: ", _action_dict(actions[-1]))
    if not arguments.execute:
        print("dry run only; add --execute after checking the workspace")
        return 0
    lower = payload["action_min"]
    upper = payload["action_max"]
    if (
        lower is None
        or upper is None
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        raise SystemExit(
            "refusing execution: chunk has no finite training-data range"
        )
    if not arguments.yes:
        answer = input(
            "Clear the workspace, keep an emergency stop ready, then type "
            "EXECUTE: "
        )
        if answer != "EXECUTE":
            print("cancelled")
            return 2

    from lerobot.robots.so_follower import (
        SO101Follower,
        SO101FollowerConfig,
    )

    config = SO101FollowerConfig(
        port=arguments.port,
        id=arguments.robot_id,
        max_relative_target=arguments.max_relative_target,
    )
    robot = SO101Follower(config)
    robot.connect()
    try:
        replay_action_chunk(robot, actions, int(payload["fps"]))
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
