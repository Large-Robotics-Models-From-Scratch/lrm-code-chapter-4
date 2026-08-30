"""Section 4.6/4.7 drivers, the figure helpers, and the CLI factory."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from ch04 import ActionTokenizer
from ch04.analysis import (
    collect_cell_softmaxes,
    collect_expert_pairs,
    collect_joint_logit_mass,
    decoded_chunk_stream,
    expert_pairs_from_batch,
    joint_mismatch_samples,
    logit_mismatch_rates,
    mismatch_rates,
    neighborhood_softmax_figure,
    open_loop_episode_trace,
    sampled_grids_by_head,
    set_seed,
)
from ch04.cli import (
    HEAD_NAMES,
    build_action_head,
    resolve_device,
)
from ch04.diagnostics import (
    joint_logit_mass,
    nearest_state_neighbors,
    plot_bimodal_comparison,
    plot_execution_schedules,
    plot_head_comparison,
    plot_joint_logit_panels,
    plot_joint_mismatch_panels,
    plot_neighbor_softmaxes,
    plot_open_loop_episode,
    plot_per_joint_metrics,
    plot_temporal_traces,
    plot_training_curves,
)
from ch04.execution import execution_schedules


def _tokenizer():
    return ActionTokenizer(
        -np.ones(6, dtype=np.float32), np.ones(6, dtype=np.float32)
    )


@pytest.fixture
def parallel_head(fake_backbone):
    from ch04 import ParallelDecodeActionHead

    return ParallelDecodeActionHead(fake_backbone, d_embed=12).eval()


# --- diagnostics: pure plotting ------------------------------------------


def test_nearest_neighbors_ranks_by_state_distance():
    states = np.array([[0.0], [10.0], [0.5], [-0.4]])
    order = nearest_state_neighbors(states, anchor_index=0, n_neighbors=3)
    assert order.tolist() == [0, 3, 2]
    with pytest.raises(IndexError):
        nearest_state_neighbors(states, anchor_index=9)
    with pytest.raises(ValueError):
        nearest_state_neighbors(states, 0, n_neighbors=99)


def test_neighbor_softmax_figure_has_targets_and_curves():
    probabilities = np.full((8, 16), 1 / 16, dtype=np.float32)
    targets = np.arange(8)
    figure = plot_neighbor_softmaxes(
        probabilities, targets, n_curves=3, caption="ckpt=x seed=0"
    )
    upper, lower = figure.axes
    assert "Held-out action distribution" in upper.get_title()
    assert "action bin" in lower.get_xlabel()
    # Provenance moves to a figure footnote so it cannot stretch the axes.
    assert any("ckpt=x seed=0" in text.get_text() for text in figure.texts)
    # Three individual curves plus the cluster mean.
    assert len(lower.lines) == 4
    plt.close(figure)
    with pytest.raises(ValueError, match="one target bin"):
        plot_neighbor_softmaxes(probabilities, targets[:2])


def test_bimodal_comparison_overlays_the_mixture_density():
    from ch04.exercises import make_bimodal_actions, train_gmm_baseline

    _, actions = make_bimodal_actions(n_samples=128)
    mixture, _ = train_gmm_baseline(steps=5)
    figure = plot_bimodal_comparison(
        actions.numpy(), 0.0, mixture=mixture
    )
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert any("Gaussian mixture" in label for label in labels)
    plt.close(figure)


def test_joint_mismatch_panels_render_one_axis_per_head():
    samples = {
        "factorized": np.random.default_rng(0).integers(0, 256, (50, 2)),
        "parallel": np.random.default_rng(1).integers(0, 256, (50, 2)),
    }
    expert = np.array([[10, 10], [240, 240]])
    figure = plot_joint_mismatch_panels(samples, expert)
    titles = [axis.get_title() for axis in figure.axes[:2]]
    assert titles[0].startswith("factorized")
    assert titles[1].startswith("parallel")
    plt.close(figure)
    with pytest.raises(ValueError, match="shape"):
        plot_joint_mismatch_panels({"bad": np.zeros((4, 3))}, expert)


def test_joint_logit_panels_compare_probability_without_sampling():
    logits = torch.zeros(3, 1, 2, 4)
    mass = joint_logit_mass(logits, dims=(0, 1))
    np.testing.assert_allclose(mass, np.full((4, 4), 1 / 16))
    rates = logit_mismatch_rates({"parallel": mass}, 2, 2)
    assert rates["parallel"] == pytest.approx(0.5)
    figure = plot_joint_logit_panels(
        {"factorized": mass, "parallel": mass},
        np.array([[0, 0], [3, 3]]),
        n_bins=4,
        bin_range=(0, 4),
    )
    # Expert plus two policy panels, followed by the shared colorbar.
    assert len(figure.axes) == 4
    assert figure.axes[0].get_title() == "held-out demonstrations"
    plt.close(figure)


def test_temporal_and_schedule_and_episode_figures():
    rng = np.random.default_rng(0)
    grids = {"parallel": rng.integers(0, 256, (4, 5, 6))}
    figure = plot_temporal_traces(grids, control=2)
    assert figure.axes[0].get_ylabel() == "sampled bin, control 2"
    plt.close(figure)

    chunks = [torch.arange(6.0).reshape(3, 2) + t for t in range(4)]
    schedules = execution_schedules(chunks)
    figure = plot_execution_schedules(schedules, expert=np.zeros((4, 2)))
    assert "expert" in figure.axes[0].get_legend_handles_labels()[1]
    plt.close(figure)

    figure = plot_open_loop_episode(
        np.zeros((5, 2)),
        np.ones((5, 2)),
        valid=np.array([True, True, False, True, True]),
    )
    assert len(figure.axes) == 2
    assert figure.axes[-1].get_xlabel().startswith("episode timestep")
    plt.close(figure)
    with pytest.raises(ValueError, match="no valid timesteps"):
        plot_open_loop_episode(
            np.zeros((2, 1)), np.zeros((2, 1)), valid=np.zeros(2, bool)
        )


def test_training_curves_plot_train_and_sparse_held_out_points():
    history = [
        {"step": float(i), "loss": 5.5 - 0.1 * i, "entropy": 5.5,
         "accuracy": 0.1 * i,
         "validation_loss": 5.4 - 0.1 * i if i % 2 else float("nan"),
         "validation_accuracy": 0.1 * i if i % 2 else float("nan")}
        for i in range(6)
    ]
    figure = plot_training_curves({"parallel": history})
    loss_axis, accuracy_axis = figure.axes
    assert loss_axis.get_ylabel() == "nats / token"
    assert accuracy_axis.get_xlabel() == "training step"
    assert accuracy_axis.get_ylabel() == "exact bin accuracy"
    labels = loss_axis.get_legend_handles_labels()[1]
    assert any("train" in label for label in labels)
    assert any("held out" in label for label in labels)
    assert any("uniform" in label for label in labels)
    # Held-out markers are drawn only where a measurement exists.
    held_out = [
        line for line in loss_axis.lines
        if line.get_linestyle() == "None"
    ]
    assert held_out and len(held_out[0].get_xdata()) == 3
    plt.close(figure)

    with pytest.raises(ValueError, match="at least one"):
        plot_training_curves({})
    with pytest.raises(ValueError, match="empty history"):
        plot_training_curves({"parallel": []})


def test_training_curves_handle_a_run_without_validation():
    history = [
        {"step": 0.0, "loss": 5.5, "entropy": 5.5,
         "accuracy": 0.0, "validation_loss": float("nan"),
         "validation_accuracy": float("nan")}
    ]
    figure = plot_training_curves({"factorized": history})
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert not any("held out" in label for label in labels)
    plt.close(figure)


def test_per_joint_metrics_use_small_multiples():
    history = []
    for step in range(3):
        record = {"step": float(step)}
        for dimension in range(2):
            record[f"accuracy_dim_{dimension}"] = 0.1 * step
            record[f"mae_in_std_dim_{dimension}"] = 1.0 - 0.1 * step
            record[f"validation_accuracy_dim_{dimension}"] = (
                0.2 if step == 2 else float("nan")
            )
            record[f"validation_mae_in_std_dim_{dimension}"] = (
                0.8 if step == 2 else float("nan")
            )
        history.append(record)
    figure = plot_per_joint_metrics(history, ["pan.pos", "lift.pos"])
    assert len(figure.axes) == 4
    assert figure.axes[0].get_title() == "pan"
    assert figure.axes[2].get_ylabel() == "MAE / training std"
    plt.close(figure)


def test_head_comparison_bars_are_labelled_and_coloured():
    from ch04.style import head_color

    summary = {
        name: {"validation_ce": 5.0 + i, "mae_std": 1.0 + i,
               "jitter": 10.0 + i}
        for i, name in enumerate(HEAD_NAMES)
    }
    figure = plot_head_comparison(summary)
    assert len(figure.axes) == 3
    first = figure.axes[0]
    assert "Held-out CE" in first.get_title()
    colours = [patch.get_facecolor() for patch in first.patches]
    assert len(colours) == 3
    import matplotlib.colors as mcolors
    assert colours[0][:3] == mcolors.to_rgb(head_color("factorized"))
    plt.close(figure)

    with pytest.raises(KeyError, match="validation_ce"):
        plot_head_comparison({"parallel": {"mae_std": 1.0}})


# --- analysis: model-driven drivers ---------------------------------------


def test_collect_cell_softmaxes_returns_aligned_rows(
    parallel_head, fake_backbone, fake_stats, chunk_batch
):
    loader = [chunk_batch(batch_size=2), chunk_batch(batch_size=3)]
    collected = collect_cell_softmaxes(
        parallel_head,
        fake_backbone,
        loader,
        fake_stats,
        _tokenizer(),
        "cpu",
        timestep=1,
        control=3,
    )
    assert collected["states"].shape == (5, 6)
    assert collected["probabilities"].shape == (5, 256)
    assert collected["target_bins"].shape == (5,)
    np.testing.assert_allclose(
        collected["probabilities"].sum(axis=1), 1.0, atol=1e-5
    )


def test_collect_cell_softmaxes_drops_padded_frames(
    parallel_head, fake_backbone, fake_stats, chunk_batch
):
    batch = chunk_batch(batch_size=4)
    batch["action_is_pad"][:2, 0] = True
    collected = collect_cell_softmaxes(
        parallel_head,
        fake_backbone,
        [batch],
        fake_stats,
        _tokenizer(),
        "cpu",
        timestep=0,
        control=0,
    )
    assert collected["states"].shape[0] == 2


def test_collect_cell_softmaxes_validates_the_cell(
    parallel_head, fake_backbone, fake_stats, chunk_batch
):
    with pytest.raises(IndexError, match="horizon"):
        collect_cell_softmaxes(
            parallel_head, fake_backbone, [chunk_batch()], fake_stats,
            _tokenizer(), "cpu", timestep=99,
        )


def test_neighborhood_figure_caption_records_the_provenance():
    collected = {
        "states": np.random.default_rng(0).normal(size=(10, 6)),
        "probabilities": np.full((10, 32), 1 / 32),
        "target_bins": np.arange(10),
    }
    figure = neighborhood_softmax_figure(
        collected, anchor_index=3, n_neighbors=5, checkpoint="best.pt",
        seed=11,
    )
    footnote = " ".join(text.get_text() for text in figure.texts)
    for token in ("best.pt", "anchor=3", "neighbors=5", "seed=11"):
        assert token in footnote, token
    plt.close(figure)


def test_joint_samples_and_mismatch_rate_per_head(
    parallel_head, fake_backbone, model_inputs
):
    from ch04 import AutoregressiveActionHead, FactorizedActionHead

    heads = {
        "parallel": parallel_head,
        "factorized": FactorizedActionHead(d_embed=12).eval(),
        "autoregressive": AutoregressiveActionHead(
            fake_backbone, d_embed=12, horizon=16, action_dim=6
        ).eval(),
    }
    samples = joint_mismatch_samples(
        heads, fake_backbone, model_inputs, dims=(4, 5), n_samples=32
    )
    assert set(samples) == set(heads)
    for pairs in samples.values():
        assert pairs.shape == (32, 2)
        assert pairs.min() >= 0 and pairs.max() < 256
    rates = mismatch_rates(samples, 128, 128)
    assert set(rates) == set(heads)
    assert all(0.0 <= value <= 1.0 for value in rates.values())


def test_collect_joint_logits_and_expert_pairs(
    parallel_head, fake_backbone, fake_stats, chunk_batch
):
    loader = [chunk_batch(batch_size=3), chunk_batch(batch_size=2)]
    mass = collect_joint_logit_mass(
        parallel_head,
        fake_backbone,
        loader,
        fake_stats,
        _tokenizer(),
        "cpu",
        dims=(4, 5),
    )
    assert mass.shape == (256, 256)
    assert mass.sum() == pytest.approx(1.0)
    pairs = collect_expert_pairs(
        loader, fake_stats, _tokenizer(), "cpu", dims=(4, 5)
    )
    assert pairs.shape == (5, 2)


def test_sampled_grids_keep_the_full_action_grid(
    parallel_head, fake_backbone, model_inputs
):
    grids = sampled_grids_by_head(
        {"parallel": parallel_head}, fake_backbone, model_inputs,
        n_samples=4,
    )
    assert grids["parallel"].shape == (4, 16, 6)


def test_open_loop_trace_and_chunk_stream(
    parallel_head, fake_backbone, fake_stats, chunk_batch
):
    loader = [chunk_batch(batch_size=2), chunk_batch(batch_size=2)]
    trace = open_loop_episode_trace(
        parallel_head, fake_backbone, loader, _tokenizer(), fake_stats,
        "cpu",
    )
    assert trace["predicted"].shape == (4, 6)
    assert trace["expert"].shape == (4, 6)
    assert trace["valid"].tolist() == [True] * 4

    chunks = decoded_chunk_stream(
        parallel_head, fake_backbone, loader, _tokenizer(), fake_stats,
        "cpu",
    )
    assert len(chunks) == 4
    assert all(chunk.shape == (16, 6) for chunk in chunks)
    with pytest.raises(ValueError, match="no batches"):
        decoded_chunk_stream(
            parallel_head, fake_backbone, [], _tokenizer(), fake_stats,
            "cpu",
        )


def test_expert_pairs_come_from_the_tokenized_targets(
    fake_stats, chunk_batch
):
    batch = chunk_batch(batch_size=3)
    pairs = expert_pairs_from_batch(
        batch, fake_stats, _tokenizer(), "cpu", dims=(4, 5), timestep=0
    )
    assert pairs.shape == (3, 2)


def test_set_seed_makes_sampling_repeatable(
    parallel_head, fake_backbone, model_inputs
):
    set_seed(3)
    first = sampled_grids_by_head(
        {"p": parallel_head}, fake_backbone, model_inputs, n_samples=4
    )["p"]
    set_seed(3)
    second = sampled_grids_by_head(
        {"p": parallel_head}, fake_backbone, model_inputs, n_samples=4
    )["p"]
    np.testing.assert_array_equal(first, second)


# --- CLI wiring -----------------------------------------------------------


@pytest.mark.parametrize("name", HEAD_NAMES)
def test_head_factory_builds_every_manuscript_head(name, fake_backbone):
    head = build_action_head(name, fake_backbone, d_embed=12, n_bins=32)
    assert head.horizon == 16
    assert head.action_dim == 6
    assert head.n_bins == 32
    owns_backbone = hasattr(head, "backbone")
    assert owns_backbone == (name != "factorized")


def test_head_factory_rejects_an_unknown_name(fake_backbone):
    with pytest.raises(ValueError, match="unknown head"):
        build_action_head("diffusion", fake_backbone, d_embed=12)


def test_cli_parser_defaults_and_all_expansion():
    from ch04.cli import build_parser

    arguments = build_parser().parse_args([])
    assert arguments.head == ["parallel"]
    assert arguments.steps == 20_000
    assert arguments.learning_rate == 1e-4
    assert arguments.backbone_learning_rate == 1e-5
    assert arguments.label_smoothing == 0.05
    every = build_parser().parse_args(["--head", "all"])
    assert every.head == ["all"]
    assert resolve_device("cpu").type == "cpu"


def test_figures_parser_accepts_a_checkpoint_and_head():
    from ch04.figures import build_parser

    arguments = build_parser().parse_args(
        ["ckpt/best.pt", "--head", "autoregressive", "--dims", "1", "2"]
    )
    assert arguments.checkpoint == "ckpt/best.pt"
    assert arguments.head == "autoregressive"
    assert arguments.dims == [1, 2]
