"""Reproducible exercises for section 4.2's continuous baselines.

Section 4.2.1 argues that mean squared error collapses onto the
conditional mean, and section 4.2.2 answers with a Gaussian mixture.
Both claims are reproduced here on the same one-dimensional bimodal
target so figure 4.4 can be regenerated from code.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def make_bimodal_actions(
    n_samples: int = 1_024,
    noise: float = 0.05,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return identical observations with actions near -1 and +1."""
    generator = torch.Generator().manual_seed(seed)
    observations = torch.zeros(n_samples, 1)
    modes = torch.randint(0, 2, (n_samples, 1), generator=generator)
    actions = modes.float().mul(2).sub(1)
    actions += noise * torch.randn(n_samples, 1, generator=generator)
    return observations, actions


def train_mse_baseline(
    steps: int = 400,
    learning_rate: float = 0.03,
    seed: int = 7,
) -> tuple[nn.Module, list[float]]:
    """Fit the optional MSE exercise and return model plus loss history."""
    torch.manual_seed(seed)
    observations, actions = make_bimodal_actions(seed=seed)
    model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(steps):
        prediction = model(observations)
        loss = torch.nn.functional.mse_loss(prediction, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


class MixtureDensityNetwork(nn.Module):
    """A minimal mixture density network for section 4.2.2.

    The network emits one mixture weight, mean, and standard deviation
    per component, so the policy is the weighted sum of Gaussians
    written in the manuscript. ``log_sigma`` is clamped because the
    mixture likelihood diverges as a component variance shrinks to zero.
    """

    def __init__(
        self,
        n_components: int = 2,
        d_input: int = 1,
        d_hidden: int = 32,
        min_log_sigma: float = -4.0,
        max_log_sigma: float = 2.0,
    ) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if min_log_sigma >= max_log_sigma:
            raise ValueError("min_log_sigma must be below max_log_sigma")
        self.n_components = n_components
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.trunk = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.Tanh()
        )
        self.readout = nn.Linear(d_hidden, 3 * n_components)
        self._spread_components()

    def _spread_components(self) -> None:
        """Separate the component means before the first update.

        Section 4.2.2 notes that mixture training becomes unstable when
        several components converge to the same mode. Starting the mean
        biases spread across the target range and the widths at a
        moderate value avoids that failure on the chapter's toy problem.
        """
        count = self.n_components
        with torch.no_grad():
            nn.init.normal_(self.readout.weight, std=0.01)
            bias = self.readout.bias
            bias.zero_()
            spread = (
                torch.linspace(-1.0, 1.0, count)
                if count > 1
                else torch.zeros(1)
            )
            bias[count : 2 * count] = spread
            bias[2 * count :] = math.log(0.5)

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_weights, means, sigmas)`` shaped ``[B, K]``."""
        if observations.ndim != 2:
            raise ValueError("observations must have shape [B, d_input]")
        raw = self.readout(self.trunk(observations))
        weight_logits, means, log_sigma = raw.chunk(3, dim=-1)
        log_sigma = log_sigma.clamp(self.min_log_sigma, self.max_log_sigma)
        return weight_logits.log_softmax(-1), means, log_sigma.exp()

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return ``log p(a | s)`` under the mixture, shaped ``[B]``."""
        if actions.shape != (observations.shape[0], 1):
            raise ValueError("actions must have shape [B, 1]")
        log_weights, means, sigmas = self(observations)
        component = (
            -0.5 * ((actions - means) / sigmas) ** 2
            - sigmas.log()
            - 0.5 * math.log(2 * math.pi)
        )
        return torch.logsumexp(log_weights + component, dim=-1)

    @torch.no_grad()
    def density(
        self, observation: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the mixture density of one state over ``grid``."""
        if observation.ndim != 2 or observation.shape[0] != 1:
            raise ValueError("observation must have shape [1, d_input]")
        if grid.ndim != 1:
            raise ValueError("grid must be one-dimensional")
        repeated = observation.expand(grid.shape[0], -1)
        return self.log_prob(repeated, grid[:, None]).exp()


def train_gmm_baseline(
    n_components: int = 2,
    steps: int = 400,
    learning_rate: float = 0.03,
    seed: int = 7,
) -> tuple[MixtureDensityNetwork, list[float]]:
    """Fit the section 4.2.2 mixture and return model plus NLL history."""
    torch.manual_seed(seed)
    observations, actions = make_bimodal_actions(seed=seed)
    model = MixtureDensityNetwork(n_components=n_components)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(steps):
        loss = -model.log_prob(observations, actions).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history
