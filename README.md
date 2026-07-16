# lrm-code-chapter-4

Companion code for **Chapter 4** of *Build a Large Robot Model (From Scratch)* (Manning).

This chapter adds the **action head** on top of the Chapter 3 backbone and trains it with **discrete behavior cloning**. Continuous robot actions are quantized into tokens, those tokens are reserved inside the language model's existing vocabulary, and an **autoregressive** head decodes an action chunk one token at a time through the same pretrained backbone that already fuses image, language, and state.

## What you build

```
images (up + side) ─┐
text               ─┤
state              ─┼─▶ UnifiedEmbeddingBackbone (Ch 3)  ─▶  hidden [B, 392+L+1, 576]
                    │        the pretrained LM fuses               │
actions ────────────┘                                              ▼
                                                        ActionHead (AR decode)
                                                                   │
                                                                   ▼
                                                     a_1 … a_{H·D}  (token ids)
```

The three-act arc:

1. **MSE collapses.** Regressing continuous actions averages multiple valid demonstrations into a single invalid one (the multimodal trap).
2. **Per-dimension categorical fixes within-joint multimodality.** Discretize each action dimension into 256 bins and predict a distribution — but a *parallel* head samples every dimension independently and produces incoherent joint actions like `(close-gripper, move-away)`.
3. **Autoregressive fixes inter-dimension coherence.** Decode the chunk left to right so each token is conditioned on the tokens already emitted. This is the head the chapter ships.

| Module | PR | Role |
|---|---|---|
| `action_tokenizer.py` | PR 1 | Uniform Q01/Q99 quantizer; maps bins to reserved LM token ids (**256 shared** across dims) |
| `fusion_adapter.py` | PR 2 | Bridges action-token embeddings into the Ch 3 fused sequence |
| `parallel_action_head.py` | PR 3 | One-shot categorical head — the foil that exposes inter-dim incoherence |
| `autoregressive_action_head.py` | PR 4 | The shipped head: left-to-right decode with KV-cache |
| `chunk_data.py` | PR 5 | Action-chunk batching + teacher forcing |
| `train.py` | PR 6 | Cross-entropy training loop with label smoothing |
| `policy.py` | PR 7 | Constrained decoding wrapper around backbone + head |
| `rollout.py` / diagnostics | PR 8 | Sim eval on `PickCubeSO100-v1`, joint-coordination plots |

`main` carries only the scaffold; each module lands in its own PR (same convention as Ch 2 / Ch 3).

## Quickstart

Disk budget: ~1 GB of HuggingFace model weights (SmolLM2-135M ≈ 260 MB, SigLIP ≈ 775 MB), an 82 MB dataset download (`lerobot/svla_so101_pickplace` streams video lazily), and ~2 GB for the virtualenv.

Wall-clock per tier:

- **Colab T4 (demo run):** ~15–30 min *(preliminary estimate; measured numbers land with the training PR)*
- **RTX 4090 (full recipe):** ~6–12 h; an A100 is proportionally faster
- **CPU:** forward-pass demos only — training is not practical

```bash
git clone git@github.com:Large-Robotics-Models-From-Scratch/lrm-code-chapter-4.git
cd lrm-code-chapter-4

# Python 3.12 via uv (recommended; on some macOS hosts the Homebrew
# python@3.12 build ships a broken pyexpat that kills pip in venvs)
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e ".[dev]"            # tokenizer + tests only
uv pip install -e ".[dev,data,sim]"   # full chapter
```

### Installing the Chapter 3 backbone

The `ch04` action heads import `from ch03 import UnifiedEmbeddingBackbone`. Chapter 3's own `pyproject.toml` still pins `transformers<5.0`, which conflicts with this chapter's `transformers==5.3.0` (required by lerobot 0.5.1). Install it **without its dependency constraints** — this repo supplies the full, verified dependency set:

```bash
# Local clone alongside this repo:
uv pip install --no-deps -e ../lrm-code-chapter-3

# Or straight from GitHub:
uv pip install --no-deps "lrm-ch03 @ git+https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-3.git@main"
```

Then open the end-to-end notebook (launch Jupyter from the repo root so
the notebook finds `configs/` and `figures/`):

```bash
jupyter lab notebooks/ch04.ipynb
```

`notebooks/ch04_executed.ipynb` is the same notebook already run top to
bottom on the real dataset and backbone, with outputs, loss/entropy
curves, and figures committed — read it to see the payoff without
running anything. On a CPU/MPS host the notebook runs a short *real*
training smoke (a few dozen steps); the full 800-step demo and the
closed-loop ManiSkill rollout are GPU/Colab paths (guarded by
`RUN_FULL` / `RUN_SIM` flags). The demo recipe also runs standalone:

```bash
python -m ch04.train --config configs/demo.yaml   # ~800 steps, T4-safe
```

### Colab

Open `notebooks/ch04.ipynb` in Colab; the first cell pip-installs this
package (and ch3 with `--no-deps`), clones the repo for `configs/`, and
prints a MODE banner confirming the GPU tier and that the real dataset
and backbone loaded. There is no silent synthetic fallback.

### Exercises

`exercises/` has scaffolded stubs (focal loss, multimodal stress test,
entropy diagnostic, k-means binning) and `exercises/solutions.ipynb`
with worked answers.

### SO-100 vs SO-101

**SO-100** names the simulated arm/env family (the ManiSkill `PickCubeSO100-v1` environment); **SO-101** is the current hardware revision the public dataset `lerobot/svla_so101_pickplace` was teleoperated on. Same 6-DOF family — this chapter trains on SO-101 data and evaluates in the SO-100 sim env.

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

## Troubleshooting

- **HF download timeouts.** SmolLM2 + SigLIP total ~1 GB on first run. Set `HF_HUB_DOWNLOAD_TIMEOUT=60` and retry; downloads resume. Behind a proxy, set `HF_ENDPOINT` accordingly.
- **`ImportError: huggingface-hub>=0.34.0,<1.0 is required ... found huggingface-hub==1.23.0`.** You have transformers 4.x next to lerobot 0.5.1 (which needs hub 1.x). The resolver does not catch this — it fails only at `import transformers`. Fix: `uv pip install "transformers==5.3.0"`.
- **`RuntimeError: expected m1 and m2 to have the same dtype ... float != c10::BFloat16`.** transformers 5.x loads SmolLM2 in its native bfloat16 (4.x defaulted to float32) while SigLIP loads float32. Load the backbone with an explicit `dtype=torch.float32` (or call `.float()` on it).
- **VRAM.** The full backbone + AR head trains comfortably in 12 GB (T4). Under 8 GB, drop the batch size before anything else; the models themselves total well under 2 GB in float32.
- **Broken pip inside the venv on macOS.** Symptom: `pyexpat` import errors from Homebrew's python@3.12. Use a uv-managed interpreter as in Quickstart.

## Repository layout

```
lrm-code-chapter-4/
├── README.md
├── CLAUDE.md
├── ARCHITECTURE_LOG.md
├── pyproject.toml
├── .lrm-agents.yml
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/ch04/
│   ├── __init__.py
│   ├── action_tokenizer.py
│   ├── fusion_adapter.py
│   ├── parallel_action_head.py
│   ├── autoregressive_action_head.py
│   ├── chunk_data.py
│   ├── train.py
│   ├── policy.py
│   ├── rollout.py
│   └── diagnostics.py
├── configs/
│   ├── demo.yaml
│   └── full.yaml
├── scripts/
│   ├── build_notebook.py
│   ├── toy_bimodal.py
│   └── make_figures.py
├── notebooks/
│   ├── ch04.ipynb
│   └── ch04_executed.ipynb
├── exercises/
│   ├── README.md
│   ├── exercise_4_2_focal_loss.py
│   ├── exercise_4_3_multimodal_stress.py
│   ├── exercise_4_4_entropy_diagnostic.py
│   ├── exercise_4_6_kmeans_binning.py
│   └── solutions.ipynb
├── figures/
│   ├── .gitkeep
│   └── .gitignore
├── agents/
│   └── chapter-04-guide.md
├── docs/
│   ├── manuscript_fixes.md
│   ├── decisions/
│   │   └── 000-environment-pins.md
│   └── internal/
│       ├── program.md
│       └── chapter_4_plan.md
└── tests/
    ├── conftest.py
    ├── fakes.py
    ├── test_smoke.py
    ├── test_guardrails.py
    ├── test_docs_sync.py
    ├── test_action_tokenizer.py
    ├── test_fusion_adapter.py
    ├── test_parallel_action_head.py
    ├── test_autoregressive_action_head.py
    ├── test_chunk_data.py
    ├── test_train.py
    ├── test_policy.py
    ├── test_rollout.py
    ├── test_diagnostics.py
    └── test_toy_bimodal.py
```

The three-act arc lives in `src/ch04/` (tokenizer → heads → training →
policy → rollout/diagnostics). `notebooks/ch04.ipynb` assembles them end
to end; `notebooks/ch04_executed.ipynb` is that notebook run top to
bottom on real data with outputs committed. `configs/` holds the demo
and full training recipes; `exercises/` holds the exercise stubs and
worked solutions. `tests/test_docs_sync.py` keeps this block honest
against `git ls-files`.

## Locked architecture (Ch 4, on the Ch 3 v5 contract)

| Component | Choice | Source file |
|---|---|---|
| Tokenizer | Uniform, 256 bins, Q01/Q99 range per dimension (percentiles computed from the dataset — the published stats lack q01/q99) | `src/ch04/action_tokenizer.py` (PR 1) |
| Vocabulary | Last 256 SmolLM2 ids (**48896–49151**) reserved as `<act_0>…<act_255>`, **shared across all 6 dimensions** (position disambiguates) — no `resize_token_embeddings`, no `add_tokens` | `src/ch04/action_tokenizer.py` (PR 1) |
| Action head | Autoregressive, 576→256 projection, teacher forcing at train, constrained decode at inference | `src/ch04/autoregressive_action_head.py` (PR 4) |
| Parallel head | One-shot categorical — foil only, not shipped | `src/ch04/parallel_action_head.py` (PR 3) |
| Loss | Per-token categorical cross-entropy + label smoothing | `src/ch04/train.py` (PR 6) |
| Robot | SO-101 data / SO-100 sim (D = 6: 5 arm joints + gripper) | Ch 2 hand-off |
| Chunk | H = 16 | Ch 4 owns |
| Backbone | `ch03.UnifiedEmbeddingBackbone` — hidden dim 576, two cameras (`up` + `side`), native SmolLM2 vocab 49,152 | Ch 3 hand-off |
| Sim eval | `PickCubeSO100-v1` (fallback `SO100GraspCube-v1`) | ADR 000 |

## Hand-off contract from Chapter 3

```python
import torch
from ch03 import UnifiedEmbeddingBackbone

backbone = UnifiedEmbeddingBackbone()
text_ids = backbone.tokenize_instruction(instruction)
sequence_ids = torch.tensor(
    [backbone.build_sequence_ids(text_ids)], dtype=torch.long
)  # [1, N]
hidden = backbone(images, sequence_ids, state)
# images:       [B, 2, 3, 224, 224]   two cameras (up + side), preprocessed
# sequence_ids: [B, N] long tensor    image + text + state template rows
# state:        [B, 6]                SO-101 state (5 joints + gripper)
# hidden:       [B, N, 576]           N = 392 + L + 1
# tokenizer:    native SmolLM2 (49,152 vocab) — no expansion, ever
```

Chapter 4 reserves the **existing** top 256 ids for action tokens; it never resizes or extends the vocabulary. (Guardrail-tested in `tests/test_guardrails.py`.)

## Built on

- **Chapter 3 repo**: [`lrm-code-chapter-3`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-3) — `from ch03 import UnifiedEmbeddingBackbone`
- **Chapter 2 repo**: [`lrm-code-chapter-2`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-2) — dataloader, normalization stats

## Hand-off to Chapter 5

Chapter 5 replaces the discrete categorical head with continuous flow
matching, motivated by the ceilings this chapter's policy makes
measurable: quantization error (half a bin width), decode latency
(`H·D` serial steps), and the discrete action space itself. Two
artifacts carry over:

- **Checkpoint format** — `ch04.train.save_checkpoint(path, head,
  fusion, step)` writes `{"step", "head", "fusion"}`, where `head` is
  the action head's own state dict and `fusion` is only the *trainable*
  backbone modules (`img_proj`, `state_proj`, `embed_tokens`,
  `language_backbone`); frozen SigLIP reloads from HuggingFace.
  `load_checkpoint` restores both. Chapter 5 swaps the head and reuses
  the fusion state.
- **Tokenizer stats** — the tokenizer's per-dimension `q01`/`q99`
  ranges (`tokenizer.lo` / `tokenizer.hi`, computed by
  `ActionTokenizer.from_lerobot_dataset`) define the action
  normalization Chapter 5's continuous head also needs.

The autoregressive factorization `p(a | o) = ∏_t ∏_d p(a_{t,d} | o,
a_{<(t,d)})` is also the on-ramp to **world action models**, an emerging
LRM class that tokenizes future observations alongside actions; the
decode loop stays generic enough for a future chapter to interleave
observation tokens.

## License

Apache 2.0.

---

## Internal: book authoring

Everything below is for the book's authors, not for readers working through the chapter.

- Operating manual for this repo: `docs/internal/program.md` (covers the `../lrm-code-agents` toolkit setup and the build loop)
- Chapter plan (synced copy of `../lrm-book/chapter_4/chapter_4_structure_and_plan.md`): `docs/internal/chapter_4_plan.md`
- Cross-chapter decision log: `ARCHITECTURE_LOG.md`
- Environment-pin ADR: `docs/decisions/000-environment-pins.md`
- Claude Code project guide: `CLAUDE.md`
