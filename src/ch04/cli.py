"""Command-line entry points for training and figure generation.

``ch04-train`` fits any of the manuscript's three action heads on the
SO-101 demonstrations; ``ch04-figures`` regenerates the section 4.6 and
4.7 figures from a saved checkpoint. Both write their inputs and results
to ``summary.json`` so a reported number can be traced back to the run
that produced it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ch04.action_tokenizer import ActionTokenizer
from ch04.autoregressive_action_head import AutoregressiveActionHead
from ch04.constants import (
    ACTION_BINS,
    ACTION_DIM,
    ACTION_HORIZON,
    SMOLLM_WIDTH,
)
from ch04.factorized_action_head import FactorizedActionHead
from ch04.parallel_action_head import ParallelDecodeActionHead

HEAD_NAMES = ("factorized", "autoregressive", "parallel")


def resolve_device(name: str | None = None) -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU unless one is requested."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def backbone_width(backbone, default: int = SMOLLM_WIDTH) -> int:
    """Read the language backbone's hidden width, falling back to 576."""
    config = getattr(
        getattr(backbone, "language_backbone", None), "config", None
    )
    return int(getattr(config, "hidden_size", default))


def build_action_head(
    name: str,
    backbone,
    horizon: int = ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
    n_bins: int = ACTION_BINS,
    d_embed: int | None = None,
):
    """Construct one of the three manuscript heads by name.

    The factorized head reads a single pooled state and therefore does not
    own the backbone; the other two extend the Chapter 3 sequence and do.
    """
    if name not in HEAD_NAMES:
        raise ValueError(
            f"unknown head {name!r}; choose from {list(HEAD_NAMES)}"
        )
    width = d_embed or backbone_width(backbone)
    if name == "factorized":
        return FactorizedActionHead(
            d_embed=width,
            horizon=horizon,
            action_dim=action_dim,
            n_bins=n_bins,
        )
    head_class = (
        AutoregressiveActionHead
        if name == "autoregressive"
        else ParallelDecodeActionHead
    )
    return head_class(
        backbone,
        d_embed=width,
        horizon=horizon,
        action_dim=action_dim,
        n_bins=n_bins,
    )


def _build_backbone(device: torch.device):
    from ch03 import VLABackbone

    return VLABackbone().to(device)


def _train_one_head(
    name: str,
    arguments: argparse.Namespace,
    loaders,
    device: torch.device,
) -> dict[str, object]:
    """Fit one head on a fresh backbone and return its run summary."""
    from ch04.analysis import set_seed
    from ch04.decoding import evaluate_open_loop
    from ch04.train import held_out_metrics, train_action_head

    train_loader, validation_loader, stats, tokenizer = loaders
    set_seed(arguments.seed)
    backbone = _build_backbone(device)
    head = build_action_head(
        name,
        backbone,
        horizon=arguments.horizon,
        n_bins=arguments.n_bins,
    ).to(device)

    checkpoint_dir = Path(arguments.checkpoint_dir) / name
    tensorboard_dir = (
        Path(arguments.tensorboard_dir) / name
        if arguments.tensorboard_dir is not None
        else None
    )
    started = time.perf_counter()
    interrupted = False
    try:
        history = train_action_head(
            head,
            backbone,
            train_loader,
            stats,
            tokenizer,
            device,
            total_steps=arguments.steps,
            warmup_steps=min(arguments.warmup, arguments.steps - 1),
            learning_rate=arguments.learning_rate,
            backbone_learning_rate=arguments.backbone_learning_rate,
            label_smoothing=arguments.label_smoothing,
            log_every=arguments.log_every,
            checkpoint_every=arguments.checkpoint_every,
            checkpoint_dir=checkpoint_dir,
            validation_loader=validation_loader,
            resume_from=arguments.resume_from,
            validation_batches=arguments.validation_batches,
            snapshot_steps=tuple(arguments.snapshot_steps),
            tensorboard_log_dir=tensorboard_dir,
        )
    except KeyboardInterrupt:
        interrupted = True
        history = []
    elapsed = time.perf_counter() - started

    validation = held_out_metrics(
        head,
        backbone,
        validation_loader,
        stats,
        tokenizer,
        device,
        max_batches=arguments.validation_batches,
    )
    summary: dict[str, object] = {
        "head": name,
        "steps": arguments.steps,
        "seed": arguments.seed,
        "device": str(device),
        "batch_size": arguments.batch_size,
        "horizon": arguments.horizon,
        "n_bins": arguments.n_bins,
        "learning_rate": arguments.learning_rate,
        "backbone_learning_rate": arguments.backbone_learning_rate,
        "label_smoothing": arguments.label_smoothing,
        "validation_fraction": arguments.validation_fraction,
        "split_seed": arguments.split_seed,
        "validation_episodes": sorted(
            getattr(validation_loader.dataset, "episodes", []) or []
        ),
        "wall_clock_s": round(elapsed, 1),
        "interrupted": interrupted,
        "validation_token_ce": validation["loss"],
        "validation_token_accuracy": validation["accuracy"],
        "validation_accuracy_by_control": validation[
            "accuracy_by_control"
        ],
        "validation_mae_in_std": validation["mae_in_std"],
        "validation_mae_in_std_by_control": validation[
            "mae_in_std_by_control"
        ],
        "validation_mae_raw_by_control": validation[
            "mae_raw_by_control"
        ],
        "history": history,
        "tensorboard_log_dir": (
            str(tensorboard_dir) if tensorboard_dir is not None else None
        ),
    }
    if arguments.open_loop_batches > 0:
        import itertools

        metrics = evaluate_open_loop(
            head,
            itertools.islice(
                validation_loader, arguments.open_loop_batches
            ),
            tokenizer,
            stats,
            backbone,
            device,
        )
        summary["open_loop_mae"] = metrics["mae"].tolist()
        summary["open_loop_mae_in_std"] = float(
            metrics["mae_in_standard_deviations"].nanmean()
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if history:
        import matplotlib.pyplot as plt

        from ch04.diagnostics import plot_per_joint_metrics
        from ch04.so101 import SO101_ACTION_NAMES
        from ch04.style import head_label, use_manuscript_style

        use_manuscript_style()
        figure = plot_per_joint_metrics(
            history, SO101_ACTION_NAMES, title=head_label(name)
        )
        figure.savefig(checkpoint_dir / "per_joint_metrics.png")
        plt.close(figure)
    (checkpoint_dir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def _make_loaders(arguments: argparse.Namespace):
    from ch04.data import (
        collect_normalized_actions,
        make_chunked_dataloaders,
    )

    train_loader, validation_loader, stats = make_chunked_dataloaders(
        arguments.dataset_id,
        horizon=arguments.horizon,
        batch_size=arguments.batch_size,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.split_seed,
        num_workers=arguments.num_workers,
    )
    normalized = collect_normalized_actions(train_loader, stats)
    tokenizer = ActionTokenizer.fit(normalized, arguments.n_bins)
    return train_loader, validation_loader, stats, tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ch04-train",
        description="Train Chapter 4 discrete behavior-cloning heads.",
    )
    parser.add_argument(
        "--head",
        nargs="+",
        default=["parallel"],
        choices=[*HEAD_NAMES, "all"],
        help="heads to train; each gets a fresh Chapter 3 backbone",
    )
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--n-bins", type=int, default=ACTION_BINS)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-learning-rate", type=float, default=1e-5
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=32,
        help="held-out batches per validation pass; 0 uses all of them",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="optional root for per-head TensorBoard event logs",
    )
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="*",
        default=(),
        help=(
            "steps to keep permanently, on top of latest.pt and best.pt; "
            "each snapshot is roughly a gigabyte"
        ),
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--open-loop-batches",
        type=int,
        default=4,
        help="held-out batches for the open-loop MAE; 0 disables it",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="shuffle episodes before the split; omit to keep order",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.steps < 2:
        raise SystemExit("--steps must be at least 2")
    if arguments.validation_batches == 0:
        arguments.validation_batches = None
    if arguments.dataset_id is None:
        from ch04.data import DEFAULT_DATASET_ID

        arguments.dataset_id = DEFAULT_DATASET_ID
    names = (
        list(HEAD_NAMES)
        if "all" in arguments.head
        else list(dict.fromkeys(arguments.head))
    )
    device = resolve_device(arguments.device)
    print(f"device={device} heads={names} dataset={arguments.dataset_id}")
    loaders = _make_loaders(arguments)

    summaries = []
    for name in names:
        print(f"=== training {name} head ===")
        summaries.append(_train_one_head(name, arguments, loaders, device))
    root = Path(arguments.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs.json").write_text(json.dumps(summaries, indent=2))
    histories = {
        summary["head"]: summary["history"]
        for summary in summaries
        if summary["history"]
    }
    if histories:
        import matplotlib.pyplot as plt

        from ch04.diagnostics import plot_training_curves
        from ch04.style import use_manuscript_style

        use_manuscript_style()
        figure = plot_training_curves(histories)
        figure.savefig(root / "training_curves.png")
        plt.close(figure)
    for summary in summaries:
        print(
            f"{summary['head']:>15}: "
            f"val_token_ce={summary['validation_token_ce']:.4f} "
            f"({summary['wall_clock_s']}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
