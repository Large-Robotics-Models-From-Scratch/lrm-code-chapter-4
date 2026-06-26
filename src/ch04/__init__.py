"""ch04 package — discrete behavior cloning, built in Chapter 4.

Turns the Chapter 3 VLA backbone into a working policy: an action
tokenizer, a parallel baseline head (built to expose its one weakness),
and the autoregressive action head we ship.

Modules land here as PRs merge:
- PR 1: action_tokenizer (uniform bins, Q01/Q99, reserved-vocab map)
- PR 2: action_head_parallel + loss (the baseline foil)
- PR 3: action_head_ar + decode (the shipped head; KV-cache, constrained)
- PR 4: dataset (LeRobot delta_timestamps chunking)
- PR 5: train (teacher forcing, freeze->unfreeze)
- PR 6: eval (closed-loop PickCubeSO100)
- PR 7: viz (entropy heatmap, parallel-vs-AR coherence)
"""

from ch04.action_tokenizer import ActionTokenizer

# Re-exports added as PRs land:
# from ch04.parallel_action_head import ParallelActionHead  # PR 2
# from ch04.autoregressive_action_head import (  # PR 3
#     AutoregressiveActionHead,
# )

__all__: list[str] = ["ActionTokenizer"]
