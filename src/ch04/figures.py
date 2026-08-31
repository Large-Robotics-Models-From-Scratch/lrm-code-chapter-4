"""Regenerate the chapter's figures from a trained checkpoint.

Every figure the manuscript attributes to code is produced here:
figure 4.4 (MSE against a mixture), figure 4.8 (the listing 4.9
neighbourhood softmaxes), figure 4.9 (joint mismatch per head), the
section 4.6.2 temporal traces, figure 4.10 (execution schedules), and
figure 4.11 (an open-loop episode).
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
    n_samples: int = 128,
    max_batches: int = 8,
    seed: int = 0,
    compare_heads: bool = False,
    verbose: bool = True,
) -> dict[str, str]:
    """Write every code-backed figure and return the written paths."""
    import matplotlib

    matplotlib.use("Agg")

    from ch04.style import use_manuscript_style

    use_manuscript_style()

    from ch04.analysis import (
        collect_cell_softmaxes,
        collect_expert_pairs,
        collect_joint_logit_mass,
        decoded_chunk_stream,
        logit_mismatch_rates,
        neighborhood_softmax_figure,
        open_loop_episode_trace,
        sampled_grids_by_head,
        set_seed,
    )
    from ch04.data import DEFAULT_DATASET_ID, make_chunked_dataloaders
    from ch04.decoding import evaluation_mode
    from ch04.diagnostics import (
        plot_execution_schedules,
        plot_joint_logit_panels,
        plot_open_loop_episode,
        plot_temporal_traces,
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

    announce("figure 4.8: held-out softmax neighbourhood")
    collected = collect_cell_softmaxes(
        head,
        backbone,
        validation_loader,
        stats,
        tokenizer,
        device,
        timestep=timestep,
        control=dims[0],
        max_batches=max_batches,
    )
    written["figure_4_8"] = str(
        _save(
            neighborhood_softmax_figure(
                collected,
                anchor_index=anchor_index,
                n_neighbors=min(n_neighbors, collected["states"].shape[0]),
                checkpoint=str(checkpoint_path),
                seed=seed,
            ),
            output / "figure_4_8_neighborhood_softmax.png",
        )
    )

    from ch04.data import prepare_batch

    batch = next(iter(validation_loader))
    model_inputs = prepare_batch(batch, stats, device, backbone)
    heads = {head_name: head}
    if compare_heads:
        for other in HEAD_NAMES:
            if other != head_name:
                heads[other] = build_action_head(
                    other, backbone, horizon=head.horizon
                ).to(device).eval()

    announce("figure 4.9: joint mismatch from logits")
    masses = {
        name: collect_joint_logit_mass(
            peer,
            backbone,
            validation_loader,
            stats,
            tokenizer,
            device,
            dims=dims,
            timestep=timestep,
            max_batches=max_batches,
        )
        for name, peer in heads.items()
    }
    with evaluation_mode(head), evaluation_mode(backbone):
        grids = sampled_grids_by_head(
            heads, backbone, model_inputs, n_samples=min(n_samples, 12)
        )
    expert_pairs = collect_expert_pairs(
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
            plot_joint_logit_panels(
                masses,
                expert_pairs,
                bin_range=(0, tokenizer.n_bins),
                dim_labels=(str(dims[0]), str(dims[1])),
            ),
            output / "figure_4_9_joint_mismatch.png",
        )
    )
    written["temporal_traces"] = str(
        _save(
            plot_temporal_traces(grids, control=dims[0]),
            output / "section_4_6_2_temporal_traces.png",
        )
    )
    written["mismatch_rates"] = json.dumps(
        logit_mismatch_rates(
            masses, tokenizer.n_bins // 2, tokenizer.n_bins // 2
        )
    )

    announce("figure 4.10: execution schedules")
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
            plot_execution_schedules(execution_schedules(chunks)),
            output / "figure_4_10_execution_schedules.png",
        )
    )

    announce("figure 4.11: open-loop episode")
    trace = open_loop_episode_trace(
        head,
        backbone,
        validation_loader,
        tokenizer,
        stats,
        device,
        max_batches=max_batches,
    )
    written["figure_4_11"] = str(
        _save(
            plot_open_loop_episode(
                trace["predicted"],
                trace["expert"],
                trace["valid"],
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
    parser.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="draws per head for figure 4.9; the AR head is the slow one",
    )
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--compare-heads",
        action="store_true",
        help=(
            "add the other two heads to the figure 4.9 comparison; the "
            "autoregressive head decodes H x D positions in series, so "
            "prefer a GPU when this is set"
        ),
    )
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
        n_samples=arguments.n_samples,
        max_batches=arguments.max_batches,
        seed=arguments.seed,
        compare_heads=arguments.compare_heads,
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
