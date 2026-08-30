"""Discrete behavior cloning components from Chapter 4.

The package is importable without the dataset or backbone stacks: the
tokenizer, losses, and heads depend only on NumPy and Torch, while the
loaders, training loop, and figure drivers import Chapter 2 and Chapter 3
lazily at call time.
"""

from ch04.action_tokenizer import ActionTokenizer, fit_action_tokenizer
from ch04.autoregressive_action_head import AutoregressiveActionHead
from ch04.constants import (
    ACTION_BINS,
    ACTION_DIM,
    ACTION_GRID,
    ACTION_HORIZON,
    SMOLLM_WIDTH,
)
from ch04.factorized_action_head import FactorizedActionHead
from ch04.losses import (
    expand_timestep_pad_mask,
    masked_token_cross_entropy,
)
from ch04.parallel_action_head import ParallelDecodeActionHead

__all__ = [
    "ACTION_BINS",
    "ACTION_DIM",
    "ACTION_GRID",
    "ACTION_HORIZON",
    "SMOLLM_WIDTH",
    "ActionTokenizer",
    "AutoregressiveActionHead",
    "FactorizedActionHead",
    "ParallelDecodeActionHead",
    "expand_timestep_pad_mask",
    "fit_action_tokenizer",
    "masked_token_cross_entropy",
]


def build_action_head(*args, **kwargs):
    """Lazy re-export of :func:`ch04.cli.build_action_head`."""
    from ch04.cli import build_action_head as builder

    return builder(*args, **kwargs)
