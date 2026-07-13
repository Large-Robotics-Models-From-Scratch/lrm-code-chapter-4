# Program: How to Implement Chapter 4 Code

This is the **operating manual** for this repo, modeled on `program.md` in `lrm-code-chapter-2` and `lrm-code-chapter-3`. It tells Claude Code (or a human collaborator) how to use the four inputs — chapter plan, agent toolkit, Manning style guide, and the Ch 3 hand-off — to go from this scaffold to a finished, validated Chapter 4 codebase.

This file lives under `docs/internal/` because it is author tooling: readers never need it, and the reader half of `README.md` does not reference it.

Read this end-to-end before writing any Python.

---

## 1. The Four Inputs

| Input | Where | Role |
|---|---|---|
| **Chapter plan** | `docs/internal/chapter_4_plan.md` (synced with `../lrm-book/chapter_4/chapter_4_structure_and_plan.md`) | The *what*. Sections, listings, figures, the Ch3→Ch4 import contract and the Ch4→Ch5 bridge. |
| **Agent toolkit** | `../lrm-code-agents/` | The *check*. Code agents (style-check, listing-check, chapter-continuity, test-gen, resource-check). |
| **Style guide** | `../lrm-book/STYLEGUIDE.md` (+ ch3's `docs/MANNING_STYLE.md` for the distilled Manning conventions) | The *how*. Annotations, line widths, banned words, code style. |
| **Ch 3 export contract** | `from ch03 import UnifiedEmbeddingBackbone` — `forward(images, sequence_ids, state)` → `[B, 392 + L + 1, 576]` | The *upstream*. Frozen interface; consume read-only, never modify. |

Also authoritative: `docs/decisions/000-environment-pins.md` (ADR 000) — the verified pin set (transformers 5.3.0, lerobot 0.5.1, torch 2.10.0, hub 1.23.0, mani-skill 3.0.1; python 3.12 via uv), the dataset facts (no q01/q99 in stats; two cameras; 6-dim actions), and the sim env id (`PickCubeSO100-v1`).

---

## 2. One-Time Setup (author machines)

Link the author toolkit and the reader-facing chapter agent into `.claude/agents/` so Claude Code discovers everything at startup:

```bash
cd lrm-code-chapter-4
mkdir -p .claude/agents

# Author toolkit (lrm-code-agents): style-check, listing-check, etc.
for f in ../lrm-code-agents/agents/*; do
    ln -sf "../$f" .claude/agents/
done
ln -s ../../lrm-code-agents/CLAUDE.md .claude/CLAUDE.md

# Reader-facing chapter agent (committed in this repo at agents/)
ln -s ../../agents/chapter-04-guide.md .claude/agents/chapter-04-guide.md
```

`.lrm-agents.yml` (repo root) already carries the chapter-4 overrides. **Restart Claude Code from this directory** — subagents load at session start.

---

## 3. The Build Loop

Same Raschka-style hybrid model as Ch 2 / Ch 3:

- **`src/ch04/*.py`** — the importable package. Every function/class downstream chapters or tests touch.
- **`notebooks/ch04.ipynb`** — the reader's primary artifact. Imports from `src/ch04/`, runs cell-by-cell, mirrors the book's prose.

Modules land in PR order:

| PR | Module | Reader role |
|---|---|---|
| 1 | `action_tokenizer.py` — uniform 256-bin quantizer, Q01/Q99 (computed — stats lack them), reserved-vocab map (256 shared ids, 48896–49151) | type-along |
| 2 | `fusion_adapter.py` — action-token embeddings into the Ch 3 fused sequence | type-along |
| 3 | `parallel_action_head.py` — the baseline foil | type-along |
| 4 | `autoregressive_action_head.py` — the shipped head (576→256, KV-cache, constrained decode) | type-along |
| 5 | `chunk_data.py` — H=16 chunk batching, teacher forcing | type-along |
| 6 | `train.py` — cross-entropy + label smoothing, freeze/unfreeze | type-along |
| 7 | `policy.py` — decode wrapper | type-along |
| 8 | `rollout.py` + diagnostics — `PickCubeSO100-v1` eval, coherence plots | provided utility |

For each listing: read the plan spec → write the code matching it exactly → mirror in the notebook → validate with `style-check` / `listing-check` → `test-gen` stubs → run the notebook end-to-end.

---

## 4. The Import Contract

`UnifiedEmbeddingBackbone.forward(images, sequence_ids, state) → [B, 392 + L + 1, 576]` is the frozen Ch 3 → Ch 4 interface.

**Implications:**
- Build the fused layout with `backbone.build_sequence_ids(text_ids)`; the forward argument is `sequence_ids` (OUR layout). HF's `input_ids` is a different, untouched API name.
- `images` is `[B, 2, 3, 224, 224]` (two cameras, `up` + `side`); `state` is `[B, 6]`.
- Load the backbone with an explicit `dtype=torch.float32` — transformers 5.x loads SmolLM2 in native bfloat16 and the mixed-dtype forward fails otherwise.
- The tokenizer is SmolLM2's native 49,152-id vocab. Ch 4 **reserves** the top 256 ids (48896–49151), shared across dimensions. **Never** `add_tokens` / `resize_token_embeddings`, never assign into `ch03.*` — both guardrail-tested in `tests/test_guardrails.py`.
- Install ch3 with `--no-deps` (its pyproject pins transformers<5.0; this repo supplies the verified dep set).

When implementation reveals a problem with an interface, update `docs/internal/chapter_4_plan.md` *and* `../lrm-book/chapter_4/chapter_4_structure_and_plan.md` together, in the same commit, and log it in `ARCHITECTURE_LOG.md`.

---

## 5. Definition of Done

A listing is done when: code matches the plan; annotations are contiguous from `#A` with 1–2 sentence explanations; `style-check` and `listing-check` pass; the notebook cell imports it and runs; a test stub exists.

A module is done when: all its listings are done; `tests/` covers its public interface with shape and dtype assertions; `resource-check` flags no critical issues.

The chapter is done when: all modules implemented and validated; `notebooks/ch04.ipynb` runs top-to-bottom from a fresh kernel; figures rendered to `figures/`; `agents/chapter-04-guide.md` roadmap matches the plan; `chapter-continuity` confirms the Ch 3 import stays read-only and the Ch 5 bridge is stable; `pytest -m "not integration"` and `ruff check src tests` are clean.

---

## 6. Working Notes

- **One listing at a time.** Don't write three files in parallel.
- **Update the plan when reality diverges.** Code-first, plan-after creates silent drift; `tests/test_docs_sync.py` catches README/tree drift automatically.
- **Chapter constants live in `src/ch04/__init__.py`** (`N_BINS`, `CHUNK_H`, `ACTION_DIM`, `ACT_TOKEN_BASE`). Import them everywhere; no magic numbers.
- **The action tokenizer stays pure NumPy** — it must run on an edge CPU at deployment.
- **Don't pre-emptively optimize.** The demo tier is a T4.
- **Clear notebook outputs before committing** (nbstripout pre-commit hook).

---

## 7. Where Things Live

| Need | Path |
|---|---|
| What to build | `docs/internal/chapter_4_plan.md` |
| Environment truth (pins, dataset facts, env id) | `docs/decisions/000-environment-pins.md` |
| Importable Python (export contract) | `src/ch04/` |
| Reader's canonical walkthrough | `notebooks/ch04.ipynb` (created in module PRs) |
| Reader's optional companion agent | `agents/chapter-04-guide.md` |
| Author-tooling agents | `../lrm-code-agents/agents/` (symlinked into `.claude/agents/`) |
| Cross-chapter decision log | `ARCHITECTURE_LOG.md` |
| Tests (CI infrastructure) | `tests/` |
| Full prose/code style rules | `../lrm-book/STYLEGUIDE.md` |
| Chapter prose draft | `../lrm-book/chapter_4/manuscript/chapter_4.md` |
