"""Chapter 4: discrete behavior cloning.

Modules land here as PRs merge:
  PR1 action_tokenizer   PR2 fusion_adapter
  PR3 parallel_action_head   PR4 autoregressive_action_head
  PR5 chunk_data   PR6 train   PR7 policy   PR8 rollout/diagnostics

The public API is re-exported lazily (PEP 562): ``from ch04 import
DiscretePolicy`` works, but importing a name only pulls in that name's
module. So ``from ch04 import ACT_TOKEN_BASE`` (or the tokenizer) stays
free of torch / lerobot for the edge-CPU deployment path, while the
training and rollout names import their heavier dependencies on demand.
"""

from __future__ import annotations

N_BINS = 256
CHUNK_H = 16
ACTION_DIM = 6
SMOLLM_VOCAB = 49152
ACT_TOKEN_BASE = SMOLLM_VOCAB - N_BINS  # 48896

# name -> submodule that defines it. Kept explicit (not star-imported)
# so the edge tokenizer path never imports torch just to read a
# constant, and so the notebook can use clean ``from ch04 import ...``.
_LAZY: dict[str, str] = {
    "ActionTokenizer": "action_tokenizer",
    "FusionAdapter": "fusion_adapter",
    "ParallelActionHead": "parallel_action_head",
    "adjacent_bin_targets": "parallel_action_head",
    "AutoregressiveActionHead": "autoregressive_action_head",
    "chunk_delta_timestamps": "chunk_data",
    "make_chunk_dataset": "chunk_data",
    "make_chunk_loader": "chunk_data",
    "prepare_images": "chunk_data",
    # NOTE: the ``train`` *function* is not re-exported here because the
    # ``ch04.train`` submodule shadows it on ``from ch04 import train``
    # (submodule import wins). Use ``from ch04.train import train``.
    "TrainConfig": "train",
    "load_config": "train",
    "build_optimizer": "train",
    "warmup_cosine": "train",
    "unfreeze_backbone": "train",
    "save_checkpoint": "train",
    "load_checkpoint": "train",
    "DiscretePolicy": "policy",
    "sample_bin": "policy",
    "evaluate": "rollout",
    "maniskill_obs_adapter": "rollout",
    "make_maniskill_obs_adapter": "rollout",
    "softmax_entropy": "diagnostics",
    "bin_frequency_histogram": "diagnostics",
    "canary_snapshot": "diagnostics",
    "plot_convergence_ridges": "diagnostics",
    "plot_bimodal_comparison": "diagnostics",
    "plot_joint_coordination": "diagnostics",
}

__all__ = [
    "N_BINS",
    "CHUNK_H",
    "ACTION_DIM",
    "SMOLLM_VOCAB",
    "ACT_TOKEN_BASE",
    *_LAZY,
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(
            f"module 'ch04' has no attribute {name!r}"
        )
    import importlib

    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)
