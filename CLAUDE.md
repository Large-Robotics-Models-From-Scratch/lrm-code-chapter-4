# lrm-code-chapter-4 — Claude Code project guide

You are working on **Chapter 4** of "Build a Large Robot Model (From Scratch)" (Manning).

## Chapter scope

This repo contains the code for the **action head and discrete behavior cloning**: an action tokenizer that quantizes continuous SO-101 actions and reserves them inside the SmolLM2 vocabulary, a parallel categorical head (the narrative foil), and the shipped **autoregressive** head that decodes an action chunk through the Chapter 3 unified-embedding backbone.

## Locked architectural decisions (Ch 4, built on the Ch 3 v5 contract)

| Component | Choice | Source of truth |
|---|---|---|
| Tokenizer | Uniform, 256 bins, Q01/Q99 per-dim range | `src/ch04/action_tokenizer.py` (PR 1) |
| Vocabulary | Last 256 SmolLM2 ids (**48896–49151**), **shared across all 6 dims** (position disambiguates) | `src/ch04/action_tokenizer.py` (PR 1) |
| Fusion adapter | Bridges action-token embeddings into the Ch 3 fused sequence | `src/ch04/fusion_adapter.py` (PR 2) |
| Parallel head | One-shot categorical — foil only, not shipped | `src/ch04/parallel_action_head.py` (PR 3) |
| Action head | Autoregressive, 576→256, teacher forcing → constrained decode | `src/ch04/autoregressive_action_head.py` (PR 4) |
| Loss | Per-token categorical cross-entropy + label smoothing | `src/ch04/train.py` (PR 6) |
| Robot | SO-101 data / SO-100 sim (D = 6: 5 arm joints + gripper) | Ch 2 hand-off |
| Chunk | H = 16 | Ch 4 owns |
| Backbone | `ch03.UnifiedEmbeddingBackbone` — hidden dim 576, two cameras (`up` + `side`), native SmolLM2 vocab 49,152 | Ch 3 hand-off |
| Sim eval | `PickCubeSO100-v1` (fallback `SO100GraspCube-v1`) | ADR 000 |
| Environment pins | transformers 5.3.0 / lerobot 0.5.1 / torch 2.10.0 / hub 1.23.0 / mani-skill 3.0.1, python 3.12 via uv | `docs/decisions/000-environment-pins.md` |

**Vocabulary convention — IMPORTANT:** reserve the **existing** last 256 ids (48896–49151). Do **NOT** call `tokenizer.add_tokens` or `model.resize_token_embeddings`. The 256 ids are **shared** across all action dimensions — sequence position disambiguates which joint (manuscript + OpenVLA recipe). Enforced by `tests/test_guardrails.py`.

**Dataset stats gotcha:** `lerobot/svla_so101_pickplace` `meta.stats["action"]` has only `min/max/mean/std/count` — **no q01/q99**. So `ActionTokenizer.from_lerobot_stats` *raises* on this dataset (it only accepts stats that already carry the percentiles); use `ActionTokenizer.from_lerobot_dataset(dataset)` instead, which computes the 1st/99th percentiles itself with one pass over the action column (11,939 rows, cheap `np.percentile`). Do not fall back to raw min/max: they contain saturated endpoints that waste bin resolution.

**Dtype gotcha (transformers 5.x):** `AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")` now loads in the checkpoint's native **bfloat16** (4.x defaulted to float32) while SigLIP loads float32, so the Ch 3 forward raises a dtype mismatch on the first matmul. Always load the backbone with an explicit `dtype=torch.float32` (or cast with `.float()`). Never rely on the 4.x float32 default.

## Hand-off contracts

**From Chapter 3** (v5 contract, ch3 `main` tip `deeead0`):
```python
import torch
from ch03 import UnifiedEmbeddingBackbone

backbone = UnifiedEmbeddingBackbone()
text_ids = backbone.tokenize_instruction(instruction)
sequence_ids = torch.tensor(
    [backbone.build_sequence_ids(text_ids)], dtype=torch.long
)  # [1, N]
hidden = backbone(images, sequence_ids, state)
# images: [B, 2, 3, 224, 224] two cameras (up + side)
# state:  [B, 6]
# hidden: [B, N, 576]  where N = 392 + L + 1
# tokenizer: native SmolLM2 (49,152 vocab) — no expansion
```
Use `build_sequence_ids` / `sequence_ids` (OUR fused layout), never HF's `input_ids`, for the fused sequence. Install ch3 with `--no-deps` (its pyproject still pins transformers<5.0 — see README Setup).

**To Chapter 5+ / world action models:**
- The autoregressive action factorization `p(a|o) = ∏_t ∏_d p(a_{t,d} | o, a_{<(t,d)})` is the interface a world action model extends by also predicting future-observation tokens. Keep the decode loop generic enough that a future chapter can interleave observation tokens.

## When editing code

- Locked names: `action_tokenizer`, `fusion_adapter`, `parallel_action_head`, `autoregressive_action_head`, `chunk_data`, `policy`, `action_head` — never abbreviate or rename
- Chapter constants live in `src/ch04/__init__.py`: `N_BINS=256`, `CHUNK_H=16`, `ACTION_DIM=6`, `ACT_TOKEN_BASE=48896` — import them, don't re-declare magic numbers
- The tokenizer stays **pure NumPy** (no torch): it must run at deployment on an edge CPU. Enforced by `tests/test_guardrails.py`.
- Never mutate the Ch 3 backbone (`ch03.*` attribute assignment is guardrail-tested); consume it read-only
- Banned: JAX, TensorFlow imports
- Line length: 76 chars (55 for annotated lines)
- Python 3.12, indent 4 spaces
- Tests must include shape and dtype assertions
- `main` is scaffold-only; land each module in its own PR

## When writing prose for the book chapter

- Prose drafts live in `../lrm-book/chapter_4/manuscript/chapter_4.md`
- **Always run `/book lint` after any manuscript edit** (em dashes near-zero, DOF not DoF, lowercase inline cross-refs, no marketing/meta words, "Vision-Language-Action model (VLA)", 76-char code lines)
- Match the writing style of Ch 1–3 manuscripts for coherency

## Cross-references

- Book repo plan: `../lrm-book/chapter_4/chapter_4_structure_and_plan.md` (this repo's `docs/internal/chapter_4_plan.md` is a synced copy)
- Manuscript: `../lrm-book/chapter_4/manuscript/chapter_4.md`
- Design spec: `../lrm-book/docs/superpowers/specs/2026-06-19-ch4-autoregressive-pivot-design.md`
- Style guide: `../lrm-book/STYLEGUIDE.md`
- Operating manual for this repo: `docs/internal/program.md`
- Environment-pin ADR: `docs/decisions/000-environment-pins.md`

## Known manuscript follow-up

The worked quantization example (§4.2.2, §4.4.2, Figure 4.3) has an arithmetic error: 0.347 rad over [−π, π] lands in bin 142, whose center is ≈0.356 rad (error ≈0.009 rad, ≈0.5°, ≈0.85 mm at a 10 cm wrist), **not** the "0.349 rad / 0.002 rad / ≈0.1° / 0.2 mm" the prose claims. `tests/test_action_tokenizer.py::test_worked_numeric_example` (PR 1) derives the center from the formula and confirms the corrected numbers. Manuscript still needs the fix.
