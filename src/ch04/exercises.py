"""Reproducible exercise for the multimodal regression trap."""

from __future__ import annotations

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
