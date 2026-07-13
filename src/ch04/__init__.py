"""Chapter 4: discrete behavior cloning.

Modules land here as PRs merge:
  PR1 action_tokenizer   PR2 fusion_adapter
  PR3 parallel_action_head   PR4 autoregressive_action_head
  PR5 chunk_data   PR6 train   PR7 policy   PR8 rollout/diagnostics
"""

N_BINS = 256
CHUNK_H = 16
ACTION_DIM = 6
SMOLLM_VOCAB = 49152
ACT_TOKEN_BASE = SMOLLM_VOCAB - N_BINS  # 48896
