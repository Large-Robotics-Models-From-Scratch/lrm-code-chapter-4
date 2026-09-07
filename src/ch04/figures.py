"""Regenerate the chapter's figures from a trained checkpoint.

Every figure the manuscript attributes to code is produced here:
figure 4.4 (MSE against a mixture), figure 4.8 (the listing 4.9
neighbourhood mode recovery), figure 4.9 (joint mismatch from deployment
samples), figure 4.10 (execution schedules), and figure 4.11 (an open-loop
episode).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ch04.action_tokenizer import ActionTokenizer
from ch04.cli import HEAD_NAMES, build_action_head, resolve_device
from ch04.constants import ACTION_DIM, ACTION_HORIZON


def load_policy(
    checkpoint_path: str | Path,
    head_name: str,
    device: torch.device | str,
    horizon: int | None = None,
):
    """Rebuild a head, backbone, tokenizer, and stats from a checkpoint.

    The tokenizer bounds and normalization statistics come from the
    checkpoint rather than being refitted, so a figure always uses the
    quantization the policy was actually trained against.
    """
    from ch03 import VLABackbone

    from ch04.train import load_policy_state_dict

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or {}
    saved = checkpoint["tokenizer"]
    tokenizer = ActionTokenizer(
        torch.as_tensor(saved["lo"]).cpu().numpy(),
        torch.as_tensor(saved["hi"]).cpu().numpy(),
        n_bins=int(saved["n_bins"]),
    )
    backbone = VLABackbone().to(device)
    head = build_action_head(
        head_name,
        backbone,
        horizon=horizon or config.get("horizon") or ACTION_HORIZON,
        action_dim=config.get("action_dim") or ACTION_DIM,
        n_bins=tokenizer.n_bins,
    ).to(device)
    load_policy_state_dict(head, backbone, checkpoint["model"])
    head.eval()
    backbone.eval()
    return head, backbone, tokenizer, checkpoint["normalization"]


def regression_trap_figure(output_dir: Path):
    """Figure 4.4 from the section 4.2 toy problem; needs no checkpoint."""
    import torch as _torch

    from ch04.diagnostics import plot_bimodal_comparison
    from ch04.exercises import (
        make_bimodal_actions,
        train_gmm_baseline,
        train_mse_baseline,
    )

    _, actions = make_bimodal_actions()
    mse_model, _ = train_mse_baseline()
    mixture, _ = train_gmm_baseline()
    with _torch.no_grad():
        prediction = float(mse_model(_torch.zeros(1, 1)).item())
    figure = plot_bimodal_comparison(
        actions.numpy(), prediction, mixture=mixture
    )
    return _save(figure, output_dir / "figure_4_4_regression_trap.png")


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def generate_all(
    checkpoint_path: str | Path,
    head_name: str,
    output_dir: str | Path,
    device: torch.device | str,
    dataset_id: str | None = None,
    batch_size: int = 8,
    validation_fraction: float = 0.1,
    split_seed: int | None = None,
    anchor_index: int = 0,
    n_neighbors: int = 32,
    dims: tuple[int, int] = (4, 5),
    timestep: int = 0,
    max_batches: int = 64,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, str]:
    """Write every code-backed figure and return the written paths."""
    import matplotlib

    matplotlib.use("Agg")

    from ch04.style import use_manuscript_style

    use_manuscript_style()

    from ch04.analysis import (
        collect_action_softmaxes,
        collect_expert_pairs,
        collect_generated_pairs,
        decoded_chunk_stream,
        neighborhood_mode_recovery_figure,
        open_loop_episode_trace,
        select_bimodal_anchor,
        select_pair_mode_support,
        select_representative_open_loop_window,
        set_seed,
    )
    from ch04.data import DEFAULT_DATASET_ID, make_chunked_dataloaders
    from ch04.diagnostics import (
        plot_execution_schedules,
        plot_joint_sample_panels,
        plot_open_loop_episode,
    )
    from ch04.execution import execution_schedules

    def announce(message: str) -> None:
        if verbose:
            print(message, flush=True)

    set_seed(seed)
    output = Path(output_dir)
    announce("figure 4.4: the section 4.2 regression trap")
    written = {"figure_4_4": str(regression_trap_figure(output))}

    head, backbone, tokenizer, stats = load_policy(
        checkpoint_path, head_name, device
    )
    # The split must match the training run, or the "held-out" frames in
    # these figures may be frames the checkpoint was fitted on.
    _, validation_loader, _ = make_chunked_dataloaders(
        dataset_id or DEFAULT_DATASET_ID,
        horizon=head.horizon,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    announce("figure 4.8: held-out neighborhood mode recovery")
    all_cells = collect_action_softmaxes(
        head,
        backbone,
        validation_loader,
        stats,
        tokenizer,
        device,
        max_batches=max_batches,
    )
    valid = all_cells["valid"][:, timestep, dims[0]]
    collected = {
        "states": all_cells["states"][valid],
        "probabilities": all_cells["probabilities"][
            valid, timestep, dims[0]
        ],
        "target_bins": all_cells["target_bins"][
            valid, timestep, dims[0]
        ],
    }
    selection = select_bimodal_anchor(
        collected["states"],
        collected["target_bins"],
        n_neighbors=n_neighbors,
    )
    written["figure_4_8"] = str(
        _save(
            neighborhood_mode_recovery_figure(
                collected,
                anchor_index=selection["anchor_index"],
                n_neighbors=n_neighbors,
                checkpoint=str(checkpoint_path),
                seed=seed,
            ),
            output / "figure_4_8_neighborhood_softmax.png",
        )
    )

    announce("figure 4.9: joint mismatch from deployment samples")
    expert_pairs = collect_expert_pairs(
        validation_loader,
        stats,
        tokenizer,
        device,
        dims=dims,
        timestep=timestep,
        max_batches=max_batches,
    )
    support = select_pair_mode_support(expert_pairs)
    generated_pairs = collect_generated_pairs(
        head,
        backbone,
        validation_loader,
        stats,
        tokenizer,
        device,
        dims=dims,
        timestep=timestep,
        max_batches=max_batches,
    )
    written["figure_4_9"] = str(
        _save(
            plot_joint_sample_panels(
                {head_name: generated_pairs},
                expert_pairs,
                support["splits"],
                support["supported_quadrants"],
                bin_range=(0, tokenizer.n_bins),
                dim_labels=(str(dims[0]), str(dims[1])),
            ),
            output / "figure_4_9_joint_mismatch.png",
        )
    )
    announce("figures 4.10 and 4.11: execution and open-loop episode")
    trace = open_loop_episode_trace(
        head,
        backbone,
        validation_loader,
        tokenizer,
        stats,
        device,
        max_batches=max_batches,
    )
    chunks = decoded_chunk_stream(
        head,
        backbone,
        validation_loader,
        tokenizer,
        stats,
        device,
        max_batches=max_batches,
    )
    written["figure_4_10"] = str(
        _save(
            plot_execution_schedules(
                execution_schedules(chunks), expert=trace["expert"]
            ),
            output / "figure_4_10_execution_schedules.png",
        )
    )

    selected = select_representative_open_loop_window(
        trace,
        max_steps=min(180, len(trace["expert"])),
        scale=torch.as_tensor(stats["action"]["std"]).cpu().numpy(),
    )
    written["figure_4_11"] = str(
        _save(
            plot_open_loop_episode(
                selected["predicted"],
                selected["expert"],
                selected["valid"],
                head_name=head_name,
            ),
            output / "figure_4_11_open_loop_episode.png",
        )
    )
    (output / "figures.json").write_text(json.dumps(written, indent=2))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ch04-figures",
        description="Regenerate Chapter 4 figures from a checkpoint.",
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--head", default="parallel", choices=HEAD_NAMES)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="must match the training run that wrote the checkpoint",
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=32)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--dims", type=int, nargs=2, default=[4, 5])
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    written = generate_all(
        arguments.checkpoint,
        arguments.head,
        arguments.output_dir,
        resolve_device(arguments.device),
        dataset_id=arguments.dataset_id,
        batch_size=arguments.batch_size,
        validation_fraction=arguments.validation_fraction,
        split_seed=arguments.split_seed,
        anchor_index=arguments.anchor_index,
        n_neighbors=arguments.n_neighbors,
        dims=tuple(arguments.dims),
        timestep=arguments.timestep,
        max_batches=arguments.max_batches,
        seed=arguments.seed,
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
