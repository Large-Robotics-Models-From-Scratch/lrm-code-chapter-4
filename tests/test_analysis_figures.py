"""Section 4.6/4.7 drivers, the figure helpers, and the CLI factory."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from ch04 import ActionTokenizer
from ch04.analysis import (
    collect_cell_softmaxes,
    collect_expert_action_grids,
    collect_expert_pairs,
    collect_joint_logit_mass,
    decoded_chunk_stream,
    expert_pairs_from_batch,
    joint_mismatch_samples,
    logit_mismatch_rates,
    measure_inference_flops,
    measure_inference_latency,
    mismatch_rates,
    neighborhood_mode_recovery_figure,
    open_loop_episode_trace,
    sampled_grids_by_head,
    select_bimodal_anchor,
    select_coupled_control_pair,
    select_pair_mode_support,
    select_representative_open_loop_window,
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
    plot_joint_sample_panels,
    plot_neighborhood_mode_recovery,
    plot_open_loop_episode,
    plot_open_loop_offset_errors,
    plot_per_joint_metrics,
    plot_quality_compute_tradeoff,
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


def test_representative_probe_finds_a_balanced_local_split():
    states = np.arange(12, dtype=np.float32)[:, None] * 0.01
    targets = np.array([20] * 6 + [220] * 6)
    selected = select_bimodal_anchor(states, targets, n_neighbors=12)
    assert selected["separation_bins"] >= 190
    assert selected["balance"] == pytest.approx(1.0)
    assert len(selected["neighbor_indices"]) == 12


def test_coupled_pair_selector_uses_held_out_targets_only():
    rng = np.random.default_rng(0)
    grids = rng.integers(0, 256, (128, 2, 4))
    grids[:, 0, 3] = grids[:, 0, 1]
    selected = select_coupled_control_pair(grids, timestep=0)
    assert selected["dims"] == (1, 3)
    assert selected["score"] > 0.9


def test_neighborhood_mode_recovery_compares_the_same_rows():
    probabilities = np.full((8, 16), 1 / 16, dtype=np.float32)
    probabilities[:4, 2] = 0.8
    probabilities[4:, 13] = 0.8
    targets = np.array([2] * 4 + [13] * 4)
    figure = plot_neighborhood_mode_recovery(
        probabilities, targets, caption="ckpt=x seed=0"
    )
    axis = figure.axes[0]
    assert "selects the expert mode on 100.0%" in axis.get_title()
    assert "8 nearby held-out frames" in axis.get_xlabel()
    assert "action bin" in axis.get_ylabel()
    # Provenance moves to a figure footnote so it cannot stretch the axes.
    assert any("ckpt=x seed=0" in text.get_text() for text in figure.texts)
    labels = axis.get_legend_handles_labels()[1]
    assert labels == [
        "expert action bin",
        "policy argmax bin",
        "expert mode centers",
    ]
    plt.close(figure)
    with pytest.raises(ValueError, match="one target bin"):
        plot_neighborhood_mode_recovery(probabilities, targets[:2])


def test_joint_sample_panels_use_expert_derived_support():
    rng = np.random.default_rng(4)
    low = rng.normal(55, 3, size=(64, 2))
    high = rng.normal(200, 3, size=(64, 2))
    expert = np.concatenate([low, high])
    support = select_pair_mode_support(expert)
    mismatch = np.column_stack([
        rng.normal(55, 3, 128), rng.normal(200, 3, 128)
    ])
    figure = plot_joint_sample_panels(
        {"factorized": mismatch},
        expert,
        support["splits"],
        support["supported_quadrants"],
    )
    assert "off support" in figure.axes[1].get_title()
    plt.close(figure)


def test_bimodal_comparison_overlays_the_mixture_density():
    from ch04.exercises import make_bimodal_actions, train_gmm_baseline

    _, actions = make_bimodal_actions(n_samples=128)
    mixture, _ = train_gmm_baseline(steps=5)
    figure = plot_bimodal_comparison(
        actions.numpy(), 0.0, mixture=mixture
    )
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert any("Gaussian mixture" in label for label in labels)
    assert figure.axes[0].get_title() == ""
    legend = figure.axes[0].get_legend()
    anchor = legend.get_bbox_to_anchor().transformed(
        figure.axes[0].transAxes.inverted()
    )
    assert anchor.y0 < 0
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
        head_name="factorized",
    )
    assert len(figure.axes) == 2
    assert figure.axes[-1].get_xlabel().startswith("episode timestep")
    assert figure.axes[0].get_title().startswith("factorized")
    assert figure.axes[0].texts[0].get_text().startswith("Control MAE")
    assert "factorized (one-shot)" in (
        figure.axes[0].get_legend_handles_labels()[1]
    )
    plt.close(figure)
    with pytest.raises(ValueError, match="no valid timesteps"):
        plot_open_loop_episode(
            np.zeros((2, 1)), np.zeros((2, 1)), valid=np.zeros(2, bool)
        )

    figure = plot_open_loop_offset_errors(
        {"factorized": np.ones((4, 2)), "parallel": np.full((4, 2), 0.5)},
        joint_names=["pan.pos", "lift.pos"],
    )
    assert len(figure.axes) == 3  # two heads and a shared colorbar
    assert figure.axes[0].get_xlabel() == "prediction offset"
    assert figure._suptitle.get_text().startswith("Held-out Control MAE")
    assert figure.axes[-1].get_ylabel() == "Control MAE"
    plt.close(figure)


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
    # Held-out measurements are joined to show progression and use
    # prominent markers.
    held_out = [
        line for line in loss_axis.lines
        if "held out" in line.get_label()
    ]
    held_accuracy = [
        line for line in accuracy_axis.lines
        if "held out" in line.get_label()
    ]
    assert held_out and len(held_out[0].get_xdata()) == 3
    assert held_accuracy and len(held_accuracy[0].get_xdata()) == 3
    validation_lines = held_out + held_accuracy
    assert all(line.get_linestyle() != "None" for line in validation_lines)
    assert all(line.get_markersize() >= 7 for line in validation_lines)
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
    assert figure.axes[2].get_ylabel() == "Control MAE"
    plt.close(figure)


def test_head_comparison_bars_are_labelled_and_coloured():
    from ch04.style import head_color

    summary = {
        name: {"validation_ce": 5.0 + i, "mae_std": 1.0 + i,
               "rollout_accuracy": 0.2 + 0.1 * i,
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

    figure = plot_quality_compute_tradeoff(summary)
    assert figure.axes[0].get_xscale() == "log"
    assert "decode-schedule proxy" in figure.axes[0].get_xlabel()
    plt.close(figure)


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
    figure = neighborhood_mode_recovery_figure(
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

    grids = collect_expert_action_grids(
        loader, fake_stats, _tokenizer(), "cpu"
    )
    assert grids["target_bins"].shape == (5, 16, 6)
    assert grids["valid"].all()


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
    assert trace["episode_index"].tolist() == [0] * 4
    assert trace["frame_index"].tolist() == list(range(4))

    moving = {
        "predicted": np.zeros((12, 2)),
        "expert": np.r_[np.zeros((6, 2)), np.arange(12).reshape(6, 2)],
        "valid": np.ones(12, dtype=bool),
        "episode_index": np.array([0] * 6 + [1] * 6),
        "frame_index": np.array(list(range(6)) * 2),
    }
    selected = select_representative_open_loop_window(moving, max_steps=4)
    assert selected["episode_index"] == 1
    assert selected["predicted"].shape == (4, 2)
    assert np.all(np.asarray(selected["indices"]) >= 6)

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
    assert arguments.checkpoint_mirror_dir is None
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


# --- section 4.6: measured inference cost ---------------------------------


def test_measure_inference_flops_reports_work_and_serial_depth(
    fake_backbone, model_inputs
):
    heads = {
        name: build_action_head(name, fake_backbone, d_embed=12).eval()
        for name in HEAD_NAMES
    }
    measured = measure_inference_flops(heads, fake_backbone, model_inputs)

    assert set(measured) == set(HEAD_NAMES)
    for name, values in measured.items():
        assert values["flops"] > 0, f"{name} recorded no arithmetic"
        assert np.isfinite(values["flops"])

    assert measured["autoregressive"]["serial_steps"] == 96
    assert measured["factorized"]["serial_steps"] == 1
    assert measured["parallel"]["serial_steps"] == 1


def test_measure_inference_flops_rejects_an_out_of_range_example(
    fake_backbone, model_inputs
):
    heads = {
        "parallel": build_action_head(
            "parallel", fake_backbone, d_embed=12
        ).eval()
    }
    with pytest.raises(IndexError):
        measure_inference_flops(
            heads, fake_backbone, model_inputs, example=9
        )


def test_tradeoff_plot_prefers_measured_flops_over_the_proxy():
    summary = {
        "factorized": {"rollout_accuracy": 0.31},
        "parallel": {"rollout_accuracy": 0.38},
        "autoregressive": {"rollout_accuracy": 0.46},
    }
    measured = {
        "factorized": {"flops": 2.0e9, "serial_steps": 1},
        "parallel": {"flops": 6.0e9, "serial_steps": 1},
        "autoregressive": {"flops": 7.0e9, "serial_steps": 96},
    }
    figure = plot_quality_compute_tradeoff(summary, flops=measured)
    axis = figure.axes[0]
    assert "FLOPs" in axis.get_xlabel()
    assert "proxy" not in axis.get_xlabel()
    labels = [text.get_text() for text in axis.texts]
    assert any("96 serial" in label for label in labels), labels
    plt.close(figure)


def test_measure_inference_latency_ranks_serial_decoding_slowest(
    fake_backbone, model_inputs
):
    heads = {
        name: build_action_head(name, fake_backbone, d_embed=12).eval()
        for name in HEAD_NAMES
    }
    measured = measure_inference_latency(
        heads, fake_backbone, model_inputs, repeats=5, warmup=2
    )

    assert set(measured) == set(HEAD_NAMES)
    for name, values in measured.items():
        assert values["latency_ms"] > 0, f"{name} recorded no time"
        assert np.isfinite(values["latency_ms"])
        assert values["p10_ms"] <= values["latency_ms"] <= values["p90_ms"]
        assert values["device"] == "cpu"

    assert measured["autoregressive"]["serial_steps"] == 96
    # 96 dependent steps against one pass: the margin is large enough that
    # this ordering does not depend on a quiet machine.
    assert (
        measured["autoregressive"]["latency_ms"]
        > measured["parallel"]["latency_ms"]
    )


def test_measure_inference_latency_validates_its_arguments(
    fake_backbone, model_inputs
):
    heads = {
        "parallel": build_action_head(
            "parallel", fake_backbone, d_embed=12
        ).eval()
    }
    with pytest.raises(IndexError):
        measure_inference_latency(
            heads, fake_backbone, model_inputs, example=9
        )
    with pytest.raises(ValueError, match="repeats"):
        measure_inference_latency(
            heads, fake_backbone, model_inputs, repeats=0
        )


def test_tradeoff_plot_prefers_latency_over_flops_and_proxy():
    summary = {
        "parallel": {"rollout_accuracy": 0.38},
        "autoregressive": {"rollout_accuracy": 0.46},
    }
    flops = {
        "parallel": {"flops": 6.0e9, "serial_steps": 1},
        "autoregressive": {"flops": 4.8e9, "serial_steps": 96},
    }
    latency = {
        "parallel": {
            "latency_ms": 12.0, "serial_steps": 1, "device": "cuda",
        },
        "autoregressive": {
            "latency_ms": 320.0, "serial_steps": 96, "device": "cuda",
        },
    }
    figure = plot_quality_compute_tradeoff(
        summary, flops=flops, latency=latency
    )
    axis = figure.axes[0]
    assert "latency" in axis.get_xlabel()
    assert "FLOPs" not in axis.get_xlabel()
    # Latency, unlike FLOPs, must place the serial head at higher cost.
    positions = {
        collection.get_offsets()[0][0]: collection
        for collection in axis.collections
    }
    assert max(positions) == 320.0
    assert "accuracy" in axis.get_ylabel()
    plt.close(figure)
