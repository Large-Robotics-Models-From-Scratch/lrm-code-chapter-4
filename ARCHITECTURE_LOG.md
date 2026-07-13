# Architecture log — chapter 4

Cross-chapter decisions tracked here. Update with each significant architectural change. Maintained alongside the `lrm-code-agents/chapter-continuity` agent.

## Locked decisions inherited from earlier chapters

| Decision | Set in | Notes |
|---|---|---|
| Robot platform | Ch 2 | SO-100 family, 6-DOF (5 arm + 1 gripper); SO-101 is the hardware revision the dataset was teleoperated on |
| Sim engine | Ch 2 | ManiSkill3 + SAPIEN |
| Anchor dataset | Ch 2 | `lerobot/svla_so101_pickplace` (50 ep, 11,939 frames, two cams, 30 FPS) |
| Dataset format | Ch 2 | LeRobotDataset v2.1 (Parquet + AV1) |
| Action dim | Ch 2 | 6 (dataset-native action format) |
| Sim env vs dataset | Ch 2 | `PickCubeSO100-v1` is the SIM env; `svla_so101_pickplace` is the DATASET; same 6-DOF interface |
| Backbone (v5, 2026-06) | Ch 3 | `UnifiedEmbeddingBackbone` — SigLIP (frozen) + SmolLM2-135M + state MLP, unified-embedding fusion via `masked_scatter` |
| Hidden width | Ch 3 | **576** (SmolLM2 native; no bridge projection) |
| Cameras | Ch 3 | **two** (`up` + `side`), 392 image tokens = 2 × 196 |
| Output contract | Ch 3 | `[B, 392 + L + 1, 576]`; call via `build_sequence_ids` / `sequence_ids` |
| Vocabulary | Ch 3 | native SmolLM2, 49,152 ids — **no expansion in any chapter** (Ch 3's older docs saying Ch 4 expands by 1,536 ids are superseded; see below) |

## Locked decisions made in chapter 4

| Decision | Why | Affects downstream |
|---|---|---|
| Action tokenizer = uniform bins, N_BINS = 256, Q01/Q99 per-dim range | RT-1 recipe; robust to saturated endpoints in the raw min/max | Ch 5 replaces with continuous head; interfaces stay |
| Percentiles computed from the dataset action column | `svla_so101_pickplace` stats ship **without** q01/q99 (only min/max/mean/std/count); raw min/max are saturated | PR 1 (`from_lerobot_stats`) |
| Reserved vocab = last 256 SmolLM2 ids, **48896–49151**, SHARED across all 6 dims | OpenVLA recipe: sequence position disambiguates the dimension; no `resize_token_embeddings`, no `add_tokens` — supersedes the "1,536 ids + resize" description in Ch 3's early hand-off notes | Ch 5+; guardrail-tested |
| Chunk H = 16 | Standard action-chunk length for 30 FPS teleop data | `chunk_data` (PR 5), heads |
| Autoregressive head 576→256, teacher forcing at train, constrained decode at inference | The chapter's thesis: fixes inter-dim incoherence the parallel head exposes | Ch 5 contrasts with flow matching |
| Backbone dtype loaded explicitly float32 | transformers 5.x loads SmolLM2 in native bfloat16; SigLIP stays float32 → dtype mismatch unless explicit | all training/eval code |
| Environment pin set (transformers 5.3.0 / lerobot 0.5.1 / torch 2.10.0 / hub 1.23.0 / mani-skill 3.0.1 / gymnasium 1.3.0 / numpy 2.2.6, python 3.12 via uv) | One env holds data + models + sim; verified end-to-end | `docs/decisions/000-environment-pins.md` (ADR 000) |
| Ch 3 installed with `--no-deps` | ch3's pyproject still pins transformers<5.0; ch4 supplies the verified dep set | README Setup |
| Sim eval env = `PickCubeSO100-v1` (fallback `SO100GraspCube-v1`) | Registered in mani-skill 3.0.1; verified in ADR 000 | PR 8 rollout |

## Hand-off contracts

### From chapter 3 (consumed here)

```python
import torch
from ch03 import UnifiedEmbeddingBackbone

backbone = UnifiedEmbeddingBackbone()
text_ids = backbone.tokenize_instruction(instruction)
sequence_ids = torch.tensor(
    [backbone.build_sequence_ids(text_ids)], dtype=torch.long
)  # [1, N]
hidden = backbone(images, sequence_ids, state)
# images: [B, 2, 3, 224, 224]; state: [B, 6]
# hidden: [B, N, 576], N = 392 + L + 1
```

Ch 4 appends action-token positions at the rightmost end of the fused sequence and reads each action position's logits through a 576→256 head. The backbone is consumed read-only.

### To chapter 5

- The trained discrete policy and its measured ceilings (quantization error, decode latency, mode collapse at decode time) are the motivation for continuous flow matching.
- The autoregressive factorization `p(a|o) = ∏ p(a_i | o, a_<i)` is the interface a world action model extends with future-observation tokens; the decode loop stays generic.

## Open decisions

- Exact T4 demo wall-clock (README carries a preliminary 15–30 min estimate; the training PR measures it).
- Label-smoothing epsilon and adjacent-bin smoothing variant — decided in PR 6 with training evidence.
- Freeze/unfreeze schedule for SmolLM2 during BC training — PR 6.
