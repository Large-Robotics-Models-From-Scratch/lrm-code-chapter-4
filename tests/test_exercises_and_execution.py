"""Section 4.2 baselines and the section 4.7.2 execution schedules."""

import math

import pytest
import torch

from ch04.execution import (
    chunk_by_chunk_trace,
    execution_schedules,
    receding_horizon_trace,
    temporal_ensemble_trace,
)
from ch04.exercises import (
    MixtureDensityNetwork,
    make_bimodal_actions,
    train_gmm_baseline,
    train_mse_baseline,
)


def test_mse_baseline_collapses_to_the_valley_between_modes():
    """Section 4.2.1's claim: the MSE optimum is the conditional mean."""
    model, history = train_mse_baseline(steps=300)
    with torch.no_grad():
        prediction = float(model(torch.zeros(1, 1)).item())
    assert abs(prediction) < 0.1, "MSE should land between -1 and +1"
    assert history[-1] < history[0]
    _, actions = make_bimodal_actions()
    # The collapsed point has essentially no support in the data.
    near = (actions - prediction).abs() < 0.25
    assert float(near.float().mean()) < 0.01


def test_mixture_recovers_both_demonstrated_modes():
    """Section 4.2.2's claim: a two-component mixture keeps both peaks."""
    mixture, history = train_gmm_baseline(steps=600)
    assert history[-1] < history[0]
    with torch.no_grad():
        log_weights, means, sigmas = mixture(torch.zeros(1, 1))
    centers = sorted(float(value) for value in means[0])
    assert centers[0] == pytest.approx(-1.0, abs=0.15)
    assert centers[1] == pytest.approx(1.0, abs=0.15)
    weights = log_weights.exp()[0]
    assert torch.allclose(weights.sum(), torch.ones(()), atol=1e-5)
    assert float(weights.min()) > 0.3, "one mode was dropped"
    assert float(sigmas.max()) < 0.5

    # Density is high at both modes and near zero in the valley between.
    grid = torch.tensor([-1.0, 0.0, 1.0])
    density = mixture.density(torch.zeros(1, 1), grid)
    assert float(density[1]) < 0.01 * float(density[0].min())
    assert float(density[1]) < 0.01 * float(density[2].min())


def test_mixture_log_prob_matches_a_hand_written_two_component_sum():
    torch.manual_seed(0)
    mixture = MixtureDensityNetwork(n_components=2)
    observations = torch.zeros(3, 1)
    actions = torch.tensor([[-1.0], [0.0], [1.0]])
    log_weights, means, sigmas = mixture(observations)
    manual = torch.logsumexp(
        log_weights
        + torch.distributions.Normal(means, sigmas).log_prob(actions),
        dim=-1,
    )
    torch.testing.assert_close(
        mixture.log_prob(observations, actions), manual
    )


def test_mixture_rejects_malformed_inputs():
    mixture = MixtureDensityNetwork()
    with pytest.raises(ValueError, match="observations"):
        mixture(torch.zeros(3))
    with pytest.raises(ValueError, match="actions"):
        mixture.log_prob(torch.zeros(3, 1), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="n_components"):
        MixtureDensityNetwork(n_components=0)


def _ramp_chunks(steps=6, horizon=3, dim=1):
    """Chunk ``t`` predicts ``t, t+1, ...`` plus a per-chunk offset."""
    return [
        torch.tensor(
            [[float(t + h) + 0.1 * t] for h in range(horizon)]
        ).expand(horizon, dim).clone()
        for t in range(steps)
    ]


def test_chunk_by_chunk_consumes_only_every_h_th_chunk():
    chunks = _ramp_chunks(steps=6, horizon=3)
    trace = chunk_by_chunk_trace(chunks)
    assert trace.shape == (6, 1)
    # Steps 0-2 come from chunk 0 (offset 0.0); steps 3-5 from chunk 3.
    expected = [0.0, 1.0, 2.0, 3.3, 4.3, 5.3]
    torch.testing.assert_close(
        trace[:, 0], torch.tensor(expected), atol=1e-6, rtol=0
    )


def test_receding_horizon_keeps_the_freshest_first_row():
    chunks = _ramp_chunks(steps=4, horizon=3)
    trace = receding_horizon_trace(chunks)
    expected = [0.0, 1.1, 2.2, 3.3]
    torch.testing.assert_close(
        trace[:, 0], torch.tensor(expected), atol=1e-6, rtol=0
    )


def test_temporal_ensemble_blends_the_overlapping_predictions():
    chunks = _ramp_chunks(steps=3, horizon=3)
    decay = math.log(2)
    trace = temporal_ensemble_trace(chunks, decay=decay)
    # At t=2 three chunks overlap with weights 1, 1/2, 1/4 in age order.
    values = torch.tensor([2.0, 2.1, 2.2])
    weights = torch.tensor([1.0, 0.5, 0.25])
    expected = float((values * weights).sum() / weights.sum())
    assert float(trace[2, 0]) == pytest.approx(expected)
    # A blend is bounded by the predictions it averages.
    assert values.min() <= trace[2, 0] <= values.max()


def test_zero_decay_ensemble_is_the_plain_mean():
    chunks = _ramp_chunks(steps=3, horizon=3)
    trace = temporal_ensemble_trace(chunks, decay=0.0)
    assert float(trace[2, 0]) == pytest.approx(
        float(torch.tensor([2.0, 2.1, 2.2]).mean())
    )


def test_execution_schedules_agree_at_the_first_timestep():
    chunks = _ramp_chunks(steps=5, horizon=3)
    schedules = execution_schedules(chunks)
    assert set(schedules) == {
        "chunk-by-chunk",
        "receding horizon",
        "temporal ensemble",
    }
    firsts = {name: float(t[0, 0]) for name, t in schedules.items()}
    assert len(set(firsts.values())) == 1, firsts
    for trace in schedules.values():
        assert trace.shape == (5, 1)


def test_execution_schedules_reject_empty_and_malformed_streams():
    with pytest.raises(ValueError, match="at least one"):
        execution_schedules([])
    with pytest.raises(ValueError, match=r"\[H, D\]"):
        chunk_by_chunk_trace([torch.zeros(3)])
