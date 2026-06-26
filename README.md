# lrm-code-chapter-4

Companion code for **Chapter 4** of *Build a Large Robot Model (From Scratch)* (Manning).

This chapter adds the **action head** on top of the Chapter 3 VLA backbone and trains it with **discrete behavior cloning**. Continuous robot actions are quantized into tokens, those tokens are reserved inside the language model's existing vocabulary, and an **autoregressive** head decodes an action chunk one token at a time through the same causal fusion transformer that already carries image, language, and state.

## The three-act arc

1. **MSE collapses.** Regressing continuous actions averages multiple valid demonstrations into a single invalid one (the multimodal trap).
2. **Per-dimension categorical fixes within-joint multimodality.** Discretize each action dimension into 256 bins and predict a distribution — but a *parallel* head samples every dimension independently and produces incoherent joint actions like `(close-gripper, move-away)`.
3. **Autoregressive fixes inter-dimension coherence.** Decode the chunk left to right so each token is conditioned on the tokens already emitted. This is the head the chapter ships.

```
image  ─┐
text   ─┤
state  ─┼─▶ VLABackbone (Ch 3, frozen-ish)  ─▶  hidden  ─▶  ActionHead  ─▶  a_1 … a_{H·D}
actions ┘        causal fusion transformer                  (AR decode)     (token ids)
```

## What you build

| Module | PR | Role |
|---|---|---|
| `action_tokenizer.py` | **PR 1** | Uniform Q01/Q99 quantizer; maps bins to reserved LM token ids (256 **shared** across dims) |
| `parallel_action_head.py` | PR 2 | One-shot categorical head — the foil that exposes inter-dim incoherence |
| `autoregressive_action_head.py` | PR 3 | The shipped head: left-to-right decode with KV-cache |
| `dataloader.py` | PR 4 | Action-chunk batching + teacher forcing |
| `train.py` | PR 5 | Cross-entropy training loop with label smoothing |
| `rollout.py` | PR 6 | Constrained decoding + sim eval on `PickCubeSO100-v1` |
| diagnostics / figures | PR 7 | Joint-coordination plots, latency reconciliation |

`main` carries only the scaffold; each module lands in its own PR (same convention as Ch 2 / Ch 3).

## Locked architecture (Ch 4 plan)

| Component | Choice |
|---|---|
| Tokenizer | Uniform, 256 bins, Q01/Q99 range per dimension |
| Vocabulary | Last 256 SmolLM ids reserved as `<act_0>…<act_255>`, **shared across all 6 dimensions** (position disambiguates) — no `resize_token_embeddings`, no `add_tokens` |
| Action head | Autoregressive, 512→256 projection, teacher forcing at train, constrained decode at inference |
| Loss | Per-token categorical cross-entropy + label smoothing |
| Robot | SO-101 (D = 6: 5 arm joints + gripper) |
| Chunk | H = 16 |
| Backbone | `ch03.VLABackbone` (hidden dim 512, native SmolLM vocab 49,152) |
| Sim eval | `PickCubeSO100-v1` |

## Setup

```bash
git clone git@github.com:Large-Robotics-Models-From-Scratch/lrm-code-chapter-4.git
cd lrm-code-chapter-4
pip install -e ".[dev]"          # tokenizer + tests only
pip install -e ".[dev,data,sim,backbone]"   # full chapter
```

`backbone` pulls `lrm-ch03` so the action head can import `ch03.VLABackbone`.

### Tests

```bash
pytest -m "not integration"   # unit tests; what CI runs
pytest -m integration         # downloads HF models + sim eval
```

### Pre-commit hooks

```bash
pre-commit install
```

Installs `nbstripout` (clears notebook outputs) and `ruff` (lint + autofix, line length 76).

## Repository layout

```
lrm-code-chapter-4/
├── README.md
├── CLAUDE.md                       # Project guide for Claude Code
├── pyproject.toml
├── .lrm-agents.yml
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/ch04/
│   ├── __init__.py
│   └── action_tokenizer.py         # PR 1 — uniform quantizer + reserved vocab map
├── docs/
│   └── chapter_4_plan.md           # Synced copy of the book chapter's structure/plan
└── tests/
    ├── conftest.py
    ├── test_smoke.py
    ├── test_action_tokenizer.py
    └── test_guardrails.py
```

## Built on

- **Chapter 3 repo**: [`lrm-code-chapter-3`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-3) — `from ch03 import VLABackbone`
- **Chapter 2 repo**: [`lrm-code-chapter-2`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-2) — dataloader, normalization stats

## Hand-off note (vocab convention)

Chapter 3's README/CLAUDE describe the action vocabulary as "1,536 IDs (256 bins × 6 dims)" plus `resize_token_embeddings`. The manuscript and the OpenVLA recipe instead reserve **256 ids shared across all dimensions** (position in the decoded sequence disambiguates which joint), with **no** vocab expansion. This repo implements the manuscript-faithful 256-shared convention. The Ch 3 docs need a one-line correction — flagged for the Ch 3 owner.

## Forward pointer

The autoregressive factorization here is also the on-ramp to **world action models**, an emerging LRM class that tokenizes future observations alongside actions. Ch 4 sets up that interface without building it.

## License

Apache 2.0.
