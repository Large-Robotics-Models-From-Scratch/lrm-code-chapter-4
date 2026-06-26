# lrm-code-chapter-4 — Claude Code project guide

You are working on **Chapter 4** of "Build a Large Robot Model (From Scratch)" (Manning).

## Chapter scope

This repo contains the code for the **action head and discrete behavior cloning**: an action tokenizer that quantizes continuous SO-101 actions and reserves them inside the SmolLM vocabulary, a parallel categorical head (the narrative foil), and the shipped **autoregressive** head that decodes an action chunk through the Chapter 3 causal fusion transformer.

## Locked architectural decisions (Ch 4 plan)

| Component | Choice | Source of truth |
|---|---|---|
| Tokenizer | Uniform, 256 bins, Q01/Q99 per-dim range | `src/ch04/action_tokenizer.py` |
| Vocabulary | Last 256 SmolLM ids, **shared across all 6 dims** (position disambiguates) | `src/ch04/action_tokenizer.py` |
| Action head | Autoregressive, 512→256, teacher forcing → constrained decode | `src/ch04/autoregressive_action_head.py` (PR 3) |
| Parallel head | One-shot categorical — foil only, not shipped | `src/ch04/parallel_action_head.py` (PR 2) |
| Loss | Per-token categorical cross-entropy + label smoothing | `src/ch04/train.py` (PR 5) |
| Robot | SO-101 (D = 6: 5 arm joints + gripper) | Ch 2 hand-off |
| Chunk | H = 16 | Ch 4 owns |
| Backbone | `ch03.VLABackbone`, hidden dim 512, native SmolLM vocab 49,152 | Ch 3 hand-off |

**Vocabulary convention — IMPORTANT:** reserve the **existing** last 256 ids. Do **NOT** call `tokenizer.add_tokens` or `model.resize_token_embeddings`. The 256 ids are **shared** across all action dimensions, not 1,536 distinct ids. (Ch 3's docs say "1,536 IDs (256 bins × 6 dims)" — that is a stale description; this repo follows the manuscript and OpenVLA, which share 256 ids and disambiguate by sequence position. Flagged for the Ch 3 owner.)

## Hand-off contracts

**From Chapter 3** (`from ch03 import VLABackbone`):
```python
backbone = VLABackbone(hidden_dim=512)
hidden = backbone(image, instruction, state)   # [B, 196 + L + 1, 512]
# tokenizer: native SmolLM (49,152 vocab) — no expansion
```

**To Chapter 5+ / world action models:**
- The autoregressive action factorization `p(a|o) = ∏_t ∏_d p(a_{t,d} | o, a_{<(t,d)})` is the interface a world action model extends by also predicting future-observation tokens. Keep the decode loop generic enough that a future chapter can interleave observation tokens.

## When editing code

- Locked names: `action_tokenizer`, `parallel_action_head`, `autoregressive_action_head`, `action_head` — never abbreviate or rename
- The tokenizer stays **pure NumPy** (no torch): it must run at deployment on an edge CPU. Enforced by `tests/test_guardrails.py`.
- Banned: JAX, TensorFlow imports
- Line length: 76 chars
- Python 3.12, indent 4 spaces
- Tests must include shape and dtype assertions
- `main` is scaffold-only; land each module in its own PR

## When writing prose for the book chapter

- Prose drafts live in `../lrm-book/chapter_4/manuscript/chapter_4.md`
- **Always run `/book lint` after any manuscript edit** (em dashes near-zero, DOF not DoF, lowercase inline cross-refs, no marketing/meta words, "Vision-Language-Action model (VLA)", 76-char code lines)
- Match the writing style of Ch 1–3 manuscripts for coherency

## Cross-references

- Book repo plan: `../lrm-book/chapter_4/chapter_4_structure_and_plan.md` (this repo's `docs/chapter_4_plan.md` is a synced copy)
- Manuscript: `../lrm-book/chapter_4/manuscript/chapter_4.md`
- Design spec: `../lrm-book/docs/superpowers/specs/2026-06-19-ch4-autoregressive-pivot-design.md`
- Style guide: `../lrm-book/STYLEGUIDE.md`

## Known manuscript follow-up

The worked quantization example (§4.2.2, §4.4.2, Figure 4.3) has an arithmetic error: 0.347 rad over [−π, π] lands in bin 142, whose center is ≈0.356 rad (error ≈0.009 rad, ≈0.5°, ≈0.85 mm at a 10 cm wrist), **not** the "0.349 rad / 0.002 rad / ≈0.1° / 0.2 mm" the prose claims. `tests/test_action_tokenizer.py::test_worked_numeric_example` derives the center from the formula and confirms the corrected numbers. Manuscript still needs the fix.
