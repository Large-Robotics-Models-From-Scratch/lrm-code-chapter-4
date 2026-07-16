# Chapter 4 Companion Repo ("Discrete Behavior Cloning") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `lrm-code-chapter-4` against the real chapter-3 v5 contract (`UnifiedEmbeddingBackbone`, 576-dim, dual camera) and implement every runnable artifact Chapter 4 v3 of the manuscript promises: action tokenizer, parallel + autoregressive action heads, chunked LeRobot dataloader, training loop, sampling policy wrapper, closed-loop ManiSkill eval, diagnostics/figures, and an E2E Colab notebook.

**Architecture:** `src/ch04` is an importable package that composes (never mutates) chapter 3's `UnifiedEmbeddingBackbone` through a thin `FusionAdapter` exposing `encode_prefix / embed / forward` — the three methods the manuscript listings assume. The tokenizer is pure NumPy; heads are PyTorch; everything trains on `lerobot/svla_so101_pickplace` and evaluates closed-loop in ManiSkill. One PR per module, matching ch2/ch3 cadence.

**Tech Stack:** Python 3.12, PyTorch, transformers (SmolLM2-135M, SigLIP), lerobot, mani-skill, numpy, pytest, ruff (76 col), nbformat.

## Global Constraints

- Python `>=3.12,<3.13`; ruff line-length **76**, select `["E","F","W","I"]` (copy ch3 pyproject).
- PyTorch only — **no JAX/TF** (guardrail test enforces).
- **Never** call `resize_token_embeddings` / `tokenizer.add_tokens` (guardrail test enforces). Action tokens reuse the **last 256 native ids**: `ACT_TOKEN_BASE = 49152 - 256 = 48896`, ids `48896..49151`, one shared block for all joints/timesteps (OpenVLA recipe, manuscript §4.3.1).
- Chapter 3 contract (frozen; from `main` (tip deeead0, v5 code merged 2026-07) of `../lrm-code-chapter-3`): `UnifiedEmbeddingBackbone.forward(images, sequence_ids, state) -> [B, 392+L+1, 576]`; images `[B, 2, 3, 224, 224]` in `[0,1]` (cameras `up` then `side`); state `[B, 6]`; `SMOLLM_VOCAB=49152`, `SMOLLM_WIDTH=576`, `IMAGE_TOKENS=392`. (Ch3 `main` now carries the v5 code — merged 2026-07, tip `deeead0`; it adds a placeholder-count guard in `forward` and expects Chapter 4's collate to pad variable-length `sequence_ids` rows.)
- Do **not** modify anything under `../lrm-code-chapter-3` or ch3's model objects at runtime (compose, don't mutate — guardrail).
- Locked hyperparameters (manuscript Table 4.3): `N_BINS=256`, `CHUNK_H=16`, `ACTION_DIM=6`, AdamW β=(0.9, 0.95), wd 0.05, head LR 1e-4, backbone LR 1e-5 from epoch 1, warmup 500 steps then cosine to 0, grad clip 1.0, bf16 autocast.
- Tokenizer bounds from dataset `q01`/`q99` stats, decode to bin **centers**.
- Dataset: `lerobot/svla_so101_pickplace` (50 eps, 11,939 frames, **30 fps** — the manuscript's `t/50.0` is wrong; code must derive offsets from `ds.fps`).
- Locked terminology in code/docs: "action head", "vision encoder", "language backbone"; `sequence_ids` (ch3 naming), not `input_ids`, for the fused layout.
- Code style: full-word names, `#A`/`#B` end-of-line annotations only where a listing mirrors the book, imports at top, straight quotes.
- Tests: pytest markers `integration` (network/model download/sim) and `slow`; unit tests never hit the network. CI runs `ruff check src tests` + `pytest -m "not integration"`.
- Commit style: `Ch4 (v2) PR<N>: <module> ...`; work on branch `ch4-v2-build` (repo `main` keeps the revert history).

## Tutorial-quality bar (from the ch3 audit + exemplar research)

Chapter 3's repo is the *convention* reference, not the *quality* reference. A 2026-07-12 audit found its notebook silently degrades to synthetic `torch.rand` data (unresolved lerobot/transformers pin conflict), commits zero cell outputs, never saves its figures, buries reader guidance in author-only docs, and ships stale file-tree diagrams. Benchmarks: rasbt/LLMs-from-scratch, karpathy/nanoGPT, LeRobot official notebooks, d2l.ai. The following are hard requirements for ch4, enforced by the tasks below:

- **Real data or loud failure — never a silent synthetic fallback.** Task 0's dependency resolution is a gate: if the pin set cannot load `lerobot/svla_so101_pickplace` alongside SmolLM2/SigLIP, implementation stops and the conflict is escalated, not worked around with fake data. Any optional degraded mode must announce itself in rendered markdown, not a swallowed exception.
- **Visible payoff without running anything.** `figures/` exists and every notebook figure is saved into it; a fully **executed** copy of the notebook (`notebooks/ch04_executed.ipynb`, outputs + loss curves + rollout video thumbnail intact) is committed alongside the nbstripout'd working copy.
- **Reader/author split.** README top half is 100% reader-facing: quickstart (clone → install → open notebook), disk/download sizes, wall-clock estimates per hardware tier (T4 / 4090 / CPU), SO-100-vs-SO-101 disambiguation sentence, troubleshooting section. Author/agent tooling (`program.md`, `ARCHITECTURE_LOG.md`, `agents/`, `.lrm-agents.yml`, plan docs) lives under a clearly-marked "Internal: book authoring" README divider and `docs/internal/` where possible.
- **Config presets, not prose hyperparameters** (nanoGPT): `configs/demo.yaml` (Colab T4, ~500–1,000 steps, subset) and `configs/full.yaml` (Table 4.3, 4090, 20k steps); `python -m ch04.train --config configs/demo.yaml` works as a standalone script path in addition to the notebook.
- **Pretrained checkpoint shipped** (Raschka/LeRobot): the full recipe is trained once for real and the checkpoint + tokenizer stats uploaded to the HF Hub, so readers can skip training and still run eval/figures/rollouts (Task 11).
- **Dataset visualization before training** (LeRobot): the notebook plays back a demonstration episode (both cameras + action traces) before any training cell.
- **Rollout-video finale** (LeRobot): the notebook ends by reloading the best checkpoint and rendering an annotated sim rollout video + success-rate readout, not a fizzle.
- **Exercise scaffolding** (Raschka): `exercises/` stubs with hints for Exercises 4.2/4.3/4.4/4.6 and a separate `exercises/solutions/` notebook, instead of prose-only pointers.
- **No doc drift**: a test (`tests/test_docs_sync.py`) diffs the README file-tree diagram against `git ls-files` so stale diagrams fail CI; the executed notebook is re-run before any release-tagged commit.

## Known manuscript bugs to code around (and log in `docs/manuscript_fixes.md`, Task 10)

1. Import path `lerobot.common.datasets.lerobot_dataset` is stale → modern lerobot is `lerobot.datasets.lerobot_dataset` (verify against pinned version in Task 0).
2. `"action": [t / 50.0 for t in range(16)]` assumes 50 Hz; dataset is 30 fps → use `t / ds.fps`.
3. Listing 4.10's `fusion(seq, causal=True)` / `fusion.embed(...)` / `fusion.encode_prefix(batch)` don't exist on ch3's backbone → `FusionAdapter` (Task 3) provides them; flag for manuscript reconciliation.
4. §4.5.5 "mask the non-action vocabulary ids to −∞" is moot with a 256-way `readout` head (Listing 4.10) — the head can only emit bins.
5. `ParallelActionHead` has no listing (prose spec only, §4.5.2); repo synthesizes it.
6. `PickCubeSO100-v1` env id must be verified against the pinned mani-skill (Task 0); record the real id.

---

### Task 0: Environment + contract verification spike

No TDD here — this task de-risks every later task and records pins.

**Files:**
- Create: `docs/decisions/000-environment-pins.md`

**Steps:**

- [ ] **Step 1:** `../lrm-code-chapter-3` is already on `main` at tip `deeead0` (verified 2026-07-12). `pip install -e ../lrm-code-chapter-3` into a fresh venv (py312) and verify:
  ```python
  from ch03 import UnifiedEmbeddingBackbone
  from ch03.vla_backbone import SMOLLM_VOCAB, SMOLLM_WIDTH, IMAGE_TOKENS
  assert (SMOLLM_VOCAB, SMOLLM_WIDTH, IMAGE_TOKENS) == (49152, 576, 392)
  ```
- [ ] **Step 2:** Resolve the lerobot/transformers dependency conflict noted in ch3 (`lerobot==0.5.1` wants `huggingface-hub>=1.0`, ch3's transformers wants `<1.0`). Try, in order: (a) newest `lerobot` + `transformers` pair that co-installs and still loads SmolLM2/SigLIP; (b) `lerobot==0.5.1` with a newer transformers. Record the working pin set. Confirm the actual import path for `LeRobotDataset` under the chosen pin.
- [ ] **Step 3:** Load `LeRobotDataset("lerobot/svla_so101_pickplace")` (this downloads; ok locally). Record: `ds.fps`, feature keys (expect `observation.images.up`, `observation.images.side`, `observation.state`, `action`), action dim, and confirm `ds.meta.stats["action"]` has `q01`/`q99` (if only `min/max/mean/std` exist, record that — the tokenizer's `from_lerobot_stats` must then compute percentiles from a pass over `action` values; decide and record).
- [ ] **Step 4:** `pip install mani-skill==3.0.1` (ch3's sim pin); check whether `PickCubeSO100-v1` registers. If not, enumerate `gym.registry` for SO100/SO101 task ids and record the closest pick-cube task (known candidates: `SO100GraspCube-v1`). Record chosen env id + fallback.
- [ ] **Step 5:** Write all findings to `docs/decisions/000-environment-pins.md` (pin table, import paths, dataset facts, env id). Commit: `Ch4 (v2) PR0: environment pins and contract verification`.
- [ ] **Step 6 (GATE):** If no pin set loads the real dataset alongside SmolLM2/SigLIP in one environment, STOP and escalate to the user with the options found (vendor the lerobot dataset reader, split envs, or upstream fix). Do not proceed to Task 5+ with a synthetic-data plan.

---

### Task 1: Scaffold rebuild against the v5 contract

**Files:**
- Create: `README.md`, `CLAUDE.md`, `program.md`, `ARCHITECTURE_LOG.md`, `pyproject.toml`, `.pre-commit-config.yaml`, `.lrm-agents.yml`, `.gitignore`, `.github/workflows/ci.yml`, `src/ch04/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`, `tests/test_guardrails.py`, `agents/chapter-04-guide.md`, `docs/chapter_4_plan.md`
- Reference (read-only): reverted commit `86a1eb2` (`git show 86a1eb2:<file>`) as the starting text for each doc, and ch3 `main` (tip deeead0, v5 code merged 2026-07) for the current conventions.

**Interfaces:**
- Produces: package `ch04` importable; constants module-level in `src/ch04/__init__.py`: `N_BINS = 256`, `CHUNK_H = 16`, `ACTION_DIM = 6`, `ACT_TOKEN_BASE = 48896`.

**Steps:**

- [ ] **Step 1:** `git checkout -b ch4-v2-build`. Recover each scaffold file from `86a1eb2` via `git show`, then correct every stale v3 reference: `VLABackbone` → `UnifiedEmbeddingBackbone`; hidden dim 512 → **576**; single camera → **two cameras (`up`+`side`)**; "1,536 action ids" → **256 shared reserved ids (48896–49151)**; add the locked-names table (component → choice → source file) in README and CLAUDE.md; pyproject deps per Task 0 pins with extras `dev`, `data`, `sim`, `backbone = ["lrm-ch03 @ git+...@main"]` (or local path instructions).
- [ ] **Step 1b (reader/author split, per quality bar):** README structured as: (1) what you build (one figure), (2) Quickstart — clone/install/open-notebook block with disk (~5 GB models + ~1 GB dataset) and wall-clock per tier (T4 demo ~X min, 4090 full ~6–12 h, CPU: forward-pass demos only), (3) SO-100 (sim env) vs SO-101 (dataset robot) disambiguation sentence, (4) Troubleshooting (HF download timeouts, transformers/lerobot pin conflict symptoms, VRAM guidance), (5) layout + handoff contracts, then a divider `## Internal: book authoring` below which the agent/tooling docs are referenced. `program.md`, plan-doc copies go under `docs/internal/`. No `../lrm-code-agents` symlink instructions in the reader path. Add `tests/test_docs_sync.py`: parse the README layout block and assert every listed path exists in `git ls-files` output.
- [ ] **Step 2:** `src/ch04/__init__.py`:

```python
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
```

- [ ] **Step 3:** `tests/conftest.py` with shared fixtures:

```python
import numpy as np
import pytest
import torch


@pytest.fixture
def action_bounds():
    lo = -np.ones(6, dtype=np.float32)
    hi = np.ones(6, dtype=np.float32)
    return lo, hi


@pytest.fixture
def fake_stats():
    return {
        "action": {
            "q01": [-1.0, -0.5, -2.0, -1.0, -1.0, 0.0],
            "q99": [1.0, 0.5, 2.0, 1.0, 1.0, 1.0],
        }
    }


@pytest.fixture
def dummy_chunk():
    torch.manual_seed(0)
    return torch.rand(2, 16, 6) * 2 - 1  # [B, H, D] in [-1, 1]
```

- [ ] **Step 4:** `tests/test_guardrails.py` — recover from `ddb3ec6` and extend: (1) `action_tokenizer.py` source contains no `import torch`; (2) no file under `src/ch04` imports jax/tensorflow; (3) no file calls `resize_token_embeddings` or `add_tokens`; (4) no file under `src/ch04` assigns into `ch03.` module attributes (mutation guard); (5) `ACT_TOKEN_BASE + N_BINS == SMOLLM_VOCAB`.
- [ ] **Step 5:** `tests/test_smoke.py`: `import ch04` and assert the four constants. Run `pytest -m "not integration"` → PASS; `ruff check src tests` → clean. Commit: `Ch4 (v2) PR0b: scaffold rebuilt on ch3 v5 contract`.

---

### Task 2 (PR1): Action tokenizer

**Files:**
- Create: `src/ch04/action_tokenizer.py`, `tests/test_action_tokenizer.py`
- Reference: `git show ddb3ec6:src/ch04/action_tokenizer.py` and `git show ddb3ec6:tests/test_action_tokenizer.py` (clean, reusable — the revert was about scaffold docs, not this code).

**Interfaces:**
- Produces: `ActionTokenizer(lo, hi, n_bins=256)` with `.encode(action) -> ids` and `.decode(ids) -> actions` (both `[..., D] -> [..., D]`, pure NumPy); `.to_token_ids(bin_ids, vocab_size=49152) -> bin_ids + vocab_size - n_bins`; `.from_token_ids(token_ids, vocab_size=49152)`; `ActionTokenizer.from_lerobot_stats(stats, key="action", n_bins=256)` classmethod reading `q01`/`q99`.

**Steps:**

- [ ] **Step 1:** Restore both files from `ddb3ec6` (`git show ddb3ec6:path > path`). Reconcile with the manuscript's Listing 4.3 semantics (clip → normalize → floor → clip; centers `lo + (arange(n)+0.5) * width`; decode via `np.take_along_axis`). Keep the reverted commit's extra API (`to_token_ids`/`from_token_ids`/`from_lerobot_stats`) but fix any v3-era default (`vocab_size` default must be 49152, reserved base 48896).
- [ ] **Step 2:** Add/keep tests: worked numeric example derived from the formula (lo=-1, hi=1, B=256, value 0.347 → bin 172, center within half a bin width); round-trip error `<= width/2` on random `[B, H, D]` batches; edge clipping (`hi` maps to bin 255, out-of-range clips); `to_token_ids` range is `[48896, 49151]` and inverts exactly; `from_lerobot_stats(fake_stats)` produces per-dim bounds.
- [ ] **Step 3:** Run `pytest tests/test_action_tokenizer.py -v` → PASS; guardrails still pass (torch-free). Commit: `Ch4 (v2) PR1: action tokenizer + reserved-vocab map`.

---

### Task 3 (PR2): Fusion adapter — the ch3→ch4 bridge

**Files:**
- Create: `src/ch04/fusion_adapter.py`, `tests/test_fusion_adapter.py`

**Interfaces:**
- Consumes: `ch03.UnifiedEmbeddingBackbone` (its submodules `vision_encoder`, `img_proj`, `state_proj`, `embed_tokens`, `language_backbone`, `tokenize_instruction`, `build_sequence_ids`, and its `image_id`/`state_id` masked_scatter splice — replicate the embedding construction from its `forward`, stopping **before** the `language_backbone(...)` call).
- Produces:

```python
class FusionAdapter(torch.nn.Module):
    def __init__(self, backbone):  # ch3 UnifiedEmbeddingBackbone
        ...
    def encode_prefix(self, batch) -> torch.Tensor:
        """[image(392) | text(L) | state(1)] embeddings, [B, P, 576].

        batch: dict with "observation.images.up"/"...side"
        ([B,3,H,W] in [0,1], resized to 224 via ch03.preprocess),
        "observation.state" [B,6], "task" list[str] (single shared
        instruction padded/truncated consistently per batch).
        """
    def embed(self, token_ids) -> torch.Tensor:
        """Embedding lookup, [B, T] -> [B, T, 576]."""
    def forward(self, seq_embeds, past_key_values=None,
                use_cache=False):
        """Run SmolLM2 on inputs_embeds (causal by construction).

        Returns (hidden_states [B, T, 576], past_key_values).
        """
    def parameters_backbone(self):
        """Trainable backbone params (SmolLM + projections),
        excluding frozen SigLIP — the optimizer's second group."""
```

**Steps:**

- [ ] **Step 1:** Write failing unit tests using a **fake backbone** (tiny `nn.Embedding(64, 8)`, stub vision/state encoders, a 1-layer `nn.TransformerEncoder`-free stub LM that echoes shapes and supports `inputs_embeds` + optional cache): `encode_prefix` output shape `[B, 392+L+1, 8]` with image rows equal to projected image features (verify masked_scatter placement at positions 0..391 and last position = state embedding); `embed` shape; `forward` returns same-length hidden states.
- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement `FusionAdapter`. `encode_prefix` replicates ch3's `forward` body up to (not including) `self.language_backbone(...)`: flatten cameras, SigLIP features, `img_proj`, `embed_tokens(sequence_ids)`, two `masked_scatter` splices. Build `sequence_ids` inside from `batch["task"]` via `backbone.tokenize_instruction` + `backbone.build_sequence_ids` (cache the ids when the instruction is constant across the dataset — this dataset has a single task string). `forward` calls `self.backbone.language_backbone(inputs_embeds=..., past_key_values=..., use_cache=...)` and returns `(last_hidden_state, past_key_values)`. `parameters_backbone()` yields `img_proj`, `state_proj`, `embed_tokens`, and `language_backbone` parameters (SigLIP is frozen and excluded).
- [ ] **Step 4:** Run unit tests → PASS. Add `@pytest.mark.integration` test with the real `UnifiedEmbeddingBackbone`: prefix shape `[1, 392+L+1, 576]`, and `forward(encode_prefix(batch))` matches `backbone(images, sequence_ids, state)` to within dtype tolerance (proving the adapter is faithful).
- [ ] **Step 5:** `ruff check` clean; commit: `Ch4 (v2) PR2: fusion adapter over UnifiedEmbeddingBackbone`.

---

### Task 4 (PR3): Parallel action head (the foil)

**Files:**
- Create: `src/ch04/parallel_action_head.py`, `tests/test_parallel_action_head.py`

**Interfaces:**
- Produces:

```python
class ParallelActionHead(nn.Module):
    def __init__(self, d_embed=576, chunk_h=16, action_dim=6,
                 n_bins=256, hidden=1024):
        # Linear(d_embed, hidden) -> GELU ->
        # Linear(hidden, chunk_h * action_dim * n_bins)
        # nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, pooled) -> torch.Tensor:
        """[B, d_embed] -> logits [B, H, D, n_bins]."""
    def loss(self, pooled, target_bins, label_smoothing_eps=0.0):
        """Cross-entropy on [B, H, D] targets; optional
        adjacent-bin smoothing: (1-eps) on target, eps/2 on each
        neighbor (edge bins put eps on the single neighbor)."""
    def sample(self, pooled, temperature=1.0) -> torch.Tensor:
        """Independent per-position sampling, [B, H, D] bin ids."""


def adjacent_bin_targets(target_bins, n_bins, eps):
    """[B, H, D] int -> [B, H, D, n_bins] soft distribution."""
```

**Steps:**

- [ ] **Step 1:** Failing tests: forward shape `[2, 16, 6, 256]`; fresh-init loss on random targets within 0.05 of `log(256) = 5.545` (the manuscript's §4.5.2 sanity check — zero-init final bias makes softmax uniform up to first-layer noise); `adjacent_bin_targets` rows sum to 1, put `1-eps` on target and `eps/2` on neighbors, edge bin puts `eps` on its one neighbor; `sample` shape/dtype and values in `[0, 256)`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (two-layer MLP per prose spec §4.5.2; `loss` uses soft targets when `eps > 0` via `-(soft * log_softmax).sum(-1).mean()`, else `F.cross_entropy`). **Step 4:** Run → PASS. **Step 5:** Commit: `Ch4 (v2) PR3: parallel action head + adjacent-bin smoothing`.

---

### Task 5 (PR4): Autoregressive action head (the shipped head)

**Files:**
- Create: `src/ch04/autoregressive_action_head.py`, `tests/test_autoregressive_action_head.py`

**Interfaces:**
- Consumes: `FusionAdapter` (`.embed`, `.forward`).
- Produces (mirrors Listing 4.10, with the repo's real defaults):

```python
class AutoregressiveActionHead(nn.Module):
    def __init__(self, fusion, d_embed=576, n_bins=256,
                 act_token_base=48896, bos_id=1):
        self.readout = nn.Linear(d_embed, n_bins)
    def forward(self, prefix, target_bins):
        """Teacher-forced CE loss. prefix [B, P, d_embed];
        target_bins [B, T] with T = H * D. Shift-right with BOS,
        embed via fusion.embed, concat after prefix, run
        fusion.forward, readout last T positions."""
    def logits(self, prefix, target_bins):
        """Same path, returns [B, T, n_bins] (for diagnostics)."""
```

**Steps:**

- [ ] **Step 1:** Failing tests (fake fusion from Task 3's test helpers): loss is scalar and ≈ `log(256)` at init on random targets (readout zero-bias init); `logits` shape `[B, T, 256]`; **causality test** — changing `target_bins[:, j]` must not change `logits[:, :j+1 ...]`... use: compute logits, then perturb the last target token and assert logits at all positions `<= T-1` unchanged (position i's logits depend only on tokens `< i` because of shift-right + causal LM); token mapping test — the ids handed to `fusion.embed` equal `act_token_base + target_bins` shifted right with `bos_id` first.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement exactly per Listing 4.10 semantics, adapted to `fusion.forward(seq)` returning `(hidden, cache)`. Zero-init `readout.bias`. **Step 4:** Run → PASS (use a real causal stub LM in the fake fusion — a 1-layer `transformers` LlamaModel with tiny config is acceptable and keeps the causality test honest without network access via local config instantiation). **Step 5:** Commit: `Ch4 (v2) PR4: autoregressive action head (teacher forcing)`.

---

### Task 6 (PR5): Chunked data loading

**Files:**
- Create: `src/ch04/chunk_data.py`, `tests/test_chunk_data.py`

**Interfaces:**
- Produces:

```python
def chunk_delta_timestamps(fps, chunk_h=16):
    """{"observation.images.up": [0.0], "...side": [0.0],
        "observation.state": [0.0],
        "action": [t / fps for t in range(chunk_h)]}"""

def make_chunk_dataset(repo_id="lerobot/svla_so101_pickplace",
                       chunk_h=16):
    """LeRobotDataset with delta_timestamps derived from ds.fps."""

def make_chunk_loader(ds, batch_size=16, num_workers=4):
    """Shuffled DataLoader; batches carry action [B, H, D]."""

def prepare_images(batch):
    """Stack up+side to [B, 2, 3, 224, 224] in [0,1] using
    ch03.preprocess.preprocess_image."""
```

**Steps:**

- [ ] **Step 1:** Failing unit tests: `chunk_delta_timestamps(30)` action list is `[0, 1/30, ..., 15/30]` and observation keys map to `[0.0]`; `prepare_images` on a fake batch of `[B,3,480,640]` tensors returns `[B,2,3,224,224]` with values in `[0,1]`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (import path per Task 0 pin; two-line wrapper around `LeRobotDataset`). **Step 4:** Unit tests PASS; add `@pytest.mark.integration` test loading the real dataset: one batch has `action` shape `[B, 16, 6]`, both camera keys present, no chunk crosses episode boundary (lerobot asserts this at construction). **Step 5:** Commit: `Ch4 (v2) PR5: chunked LeRobot data loading`.

---

### Task 7 (PR6): Training loop

**Files:**
- Create: `src/ch04/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `ActionTokenizer`, `FusionAdapter`, `AutoregressiveActionHead` (or `ParallelActionHead`), `make_chunk_loader`.
- Produces:

```python
def warmup_cosine(step, warmup=500, total=20_000):
    """LR multiplier: linear 0->1 over warmup, cosine 1->0 after."""

def build_optimizer(head, fusion, head_lr=1e-4):
    """Two AdamW groups: head @ head_lr, backbone @ 0.0 (frozen
    epoch 0), betas=(0.9, 0.95), weight_decay=0.05."""

def unfreeze_backbone(opt, fusion, backbone_lr=1e-5): ...

@dataclass
class TrainConfig:
    n_epochs: int = 4
    steps_per_checkpoint: int = 5_000
    grad_accum: int = 2
    microbatch: int = 16
    warmup_steps: int = 500
    total_steps: int = 20_000
    bf16: bool = True
    out_dir: str = "checkpoints"

def train(head, fusion, tokenizer, loader, cfg, log_fn=print):
    """Manuscript Listing 4.13 semantics + accumulation, autocast
    (bf16 if supported else fp16+GradScaler), clip 1.0, freeze
    epoch 0 -> unfreeze at epoch 1, per-step LR from
    warmup_cosine, entropy logging, canary softmax snapshots and
    checkpoint save every steps_per_checkpoint. Returns history
    dict (loss, lr, entropy per logged step)."""

def save_checkpoint(path, head, fusion, step): ...
def load_checkpoint(path, head, fusion): ...
```

**Steps:**

- [ ] **Step 0:** Add `configs/demo.yaml` (microbatch 8, grad_accum 1, total_steps 800, warmup 100, fp16, episode subset) and `configs/full.yaml` (Table 4.3 verbatim: microbatch 16, accum 2, 20k steps, warmup 500, bf16) plus a `__main__` entry so `python -m ch04.train --config configs/demo.yaml` runs standalone (nanoGPT config-preset pattern).
- [ ] **Step 1:** Failing tests: `warmup_cosine(0)==0`, `warmup_cosine(500)==1`, `warmup_cosine(20_000)≈0`, monotone rise then fall; `build_optimizer` group LRs `(1e-4, 0.0)` and group 1 contains no SigLIP params; `unfreeze_backbone` flips `requires_grad` and sets 1e-5; a 30-step `train()` run on a synthetic loader (random prefix embeddings via fake fusion, random chunks) **reduces loss** vs step 0 and starts near `log(256)`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. Encoding path per Listing 4.13: `bins = tokenizer.encode(batch["action"].numpy())` → flatten to `[B, H*D]`; `prefix = fusion.encode_prefix(batch)`; `loss = head(prefix, bins)`. **Step 4:** PASS. **Step 5:** Commit: `Ch4 (v2) PR6: training loop (freeze/unfreeze, warmup+cosine, bf16)`.

---

### Task 8 (PR7): Policy wrapper + decode-time sampling

**Files:**
- Create: `src/ch04/policy.py`, `tests/test_policy.py`

**Interfaces:**
- Consumes: `FusionAdapter`, `AutoregressiveActionHead`, `ActionTokenizer`.
- Produces:

```python
def sample_bin(logits, strategy="temperature_top_p",
               temperature=1.0, top_p=0.95, centers=None):
    """[B, n_bins] logits -> [B] bin ids (or float actions for
    "expected_value"). Strategies: "argmax", "temperature",
    "temperature_top_p", "expected_value" (sum_b p_b * c_b,
    needs centers). Top-p: sort desc, keep smallest set with
    cumsum >= top_p, renormalize, sample."""

class DiscretePolicy:
    def __init__(self, fusion, head, tokenizer, chunk_h=16,
                 action_dim=6, temperature=1.0, top_p=0.95,
                 device="cuda"):
    def reset(self):
        """Clear the chunk buffer and KV cache."""
    def select_action(self, obs) -> np.ndarray:
        """Return one [D] action per call. If the buffer is
        empty: encode_prefix(obs), decode H*D tokens one at a
        time with use_cache=True, sample each via sample_bin,
        tokenizer.decode -> [H, D] actions, fill buffer. Then pop
        and return the next buffered action (open-loop within a
        chunk, manuscript §4.5.5)."""
```

**Steps:**

- [ ] **Step 1:** Failing tests: `sample_bin` argmax returns argmax; top-p with a peaked distribution never samples tail bins (construct logits where tail mass < 1-p); expected-value equals hand-derived `sum p*c` on a 4-bin toy; temperature→0 approaches argmax. `DiscretePolicy` with fake fusion/head: first `select_action` triggers exactly one decode of `H*D` steps (count fake-forward calls), the next `H-1` calls trigger zero decodes and return buffered rows in order; `reset()` forces re-decode; returned actions equal `tokenizer.decode` of the sampled bins.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with incremental decode: first forward on `[prefix | BOS-embed]` with `use_cache=True`, then loop `H*D` single-token forwards feeding sampled `act_token_base + bin` embeddings. **Step 4:** PASS. **Step 5:** Commit: `Ch4 (v2) PR7: KV-cached discrete policy + sampling strategies`.

---

### Task 9 (PR8): Rollout eval, toy demos, diagnostics + figures

**Files:**
- Create: `src/ch04/rollout.py`, `src/ch04/diagnostics.py`, `scripts/toy_bimodal.py`, `scripts/make_figures.py`, `tests/test_rollout.py`, `tests/test_diagnostics.py`

**Interfaces:**
- Produces:

```python
# rollout.py
def evaluate(policy, env_id, n_seeds=50, obs_adapter=None):
    """Listing 4.14: gym.make(env_id, obs_mode="rgb"), reset per
    seed, policy.reset(), step until term/trunc, collect
    info["success"]. Returns success_rate, per-seed list.
    obs_adapter maps ManiSkill obs -> the batch dict
    encode_prefix expects (camera renames, resize, state)."""

# diagnostics.py
def softmax_entropy(logits): ...          # [.., n_bins] -> [..]
def bin_frequency_histogram(loader, tokenizer): ...
def canary_snapshot(head, fusion, batch, target_bins): ...
def plot_convergence_ridges(snapshots, out_path): ...   # Fig 4.8
def plot_bimodal_comparison(mse_pred, cat_logits, centers,
                            out_path): ...              # Fig 4.9
def plot_joint_coordination(par_samples, ar_samples,
                            out_path): ...              # Fig 4.10
```

- `scripts/toy_bimodal.py` = manuscript Listing 4.1 verbatim (MSE collapse toy), printing the near-zero prediction and saving a figure.

**Steps:**

- [ ] **Step 1:** Failing tests: `evaluate` with a scripted fake env/policy (fixed success pattern) returns the exact rate and calls `policy.reset()` once per seed; `softmax_entropy` of uniform logits equals `log(256)` and of one-hot ≈ 0; `plot_*` functions write a PNG file (smoke).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. Figures follow FIGURE_STYLE_GUIDE (dpi=300, grayscale-safe, `figures/figure_4_<slug>.png`). **Step 4:** PASS; add `@pytest.mark.integration` ManiSkill smoke test (env from Task 0's recorded id, 1 seed, random policy). **Step 5:** Commit: `Ch4 (v2) PR8: closed-loop eval, diagnostics dashboard, figure scripts`.

---

### Task 10 (PR9): E2E Colab notebook, docs, manuscript feedback

**Files:**
- Create: `scripts/build_notebook.py`, `notebooks/ch04.ipynb` (generated), `docs/manuscript_fixes.md`
- Modify: `README.md` (final layout/handoff sections), `ARCHITECTURE_LOG.md` ("Update (ch4 v2)" entry), `src/ch04/__init__.py` (re-export all public names).

**Steps:**

- [ ] **Step 1:** Write `scripts/build_notebook.py` following ch3 v5's `md()`/`code()` nbformat generator pattern (`git show ch3-v5-s5-notebook-docs` branch or the ch3 tip for the template). Notebook sections, mirroring the manuscript:
  1. Title + recap; Colab bootstrap cell (`pip install "lrm-ch04 @ git+...@ch4-v2-build"` + the Task 0 pin set; GPU check; fp16 fallback note for T4). The bootstrap cell prints an explicit MODE banner (GPU tier, real-data confirmed) — never a silent fallback.
  2. **Dataset visualization first** (LeRobot pattern): load `svla_so101_pickplace`, play back one demonstration episode inline (both cameras side by side + per-joint action traces), print episode/frame counts and fps. The reader sees the task before any model code.
  3. §4.2 — run `toy_bimodal` inline: watch MSE predict 0.0 on a bimodal target (figure saved to `figures/`).
  4. §4.3 — build `ActionTokenizer`, fit on real dataset stats, worked round-trip (0.347 rad example), reserved-vocab map demo showing ids 48896–49151, and a bin-frequency histogram of the real training set (diagnostics §4.6.7).
  5. §4.5 — load ch3 backbone + `FusionAdapter`; instantiate both heads; verify the `log(256) = 5.545` fresh-init loss sanity check for each.
  6. §4.6 — chunk loader; **demo training run** driven by `configs/demo.yaml` (~800 steps, fp16, T4-safe) with live loss curve + entropy plot; a clearly-marked optional "full recipe" cell pointing at `configs/full.yaml` for 4090/A100 users; then a **checkpoint-download cell** that fetches the released full-recipe checkpoint from the HF Hub (Task 11) so readers who skip training still get real results below.
  7. Figures — canary softmax ridge plot (Fig 4.8, from full-recipe checkpoint snapshots), MSE-vs-categorical bimodal comparison (Fig 4.9), parallel-vs-AR joint-coordination histogram (Fig 4.10, sampling both heads a few hundred times). All saved to `figures/`.
  8. **Finale** (LeRobot pattern): closed-loop eval cell (ManiSkill install marked heavy/optional; 5–10 seeds in Colab, 50 for full recipe) that renders an annotated rollout **video** inline with the success-rate readout, then summary/handoff to ch5.
- [ ] **Step 2:** Generate; keep two artifacts: `notebooks/ch04.ipynb` (nbstripout'd working copy) and `notebooks/ch04_executed.ipynb` (fully executed on real data with outputs, curves, and video thumbnail committed — d2l/LeRobot "visible payoff" pattern). Every cell must run; record demo wall-clock in the notebook prose.
- [ ] **Step 2b:** Create `exercises/` with scaffolded stubs + hints for Exercises 4.2 (focal loss swap), 4.3 (multimodal stress test), 4.4 (entropy diagnostic), 4.6 (k-means binning), and `exercises/solutions.ipynb` with worked solutions (Raschka pattern).
- [ ] **Step 3:** Write `docs/manuscript_fixes.md` — the six items from "Known manuscript bugs" above plus anything discovered during implementation, each with manuscript line refs, so the chapter text can be reconciled (per the no-lecture-refs commit rule, keep this out of commit messages).
- [ ] **Step 4:** Final `README.md`: what-you-build diagram, locked-arch table, setup (local + Colab), layout, ch3→ch4 handoff contract, ch4→ch5 handoff (checkpoint format from `save_checkpoint`, tokenizer stats file). `ARCHITECTURE_LOG.md` update entry documenting the revert-and-rebuild and the v5 alignment.
- [ ] **Step 5:** Full gate: `ruff check src tests scripts`, `pytest -m "not integration"`, `pytest -m integration` (locally), pre-commit clean. Commit: `Ch4 (v2) PR9: E2E notebook, docs, handoff`.

---

### Task 11 (PR10): Full training run + released checkpoint

Decision (2026-07-12): the full run happens on a **paid Colab A100**, driven from the notebook's full-recipe cell (`configs/full.yaml`; A100 row of the manuscript's hardware note: microbatch 32 × accum 2). The user runs it; the repo must make the A100 path one-click (bootstrap cell handles pins, checkpoint saves to Drive/HF).

**Files:**
- Create: `scripts/release_checkpoint.py` (upload checkpoint + tokenizer bounds + config + metrics card to HF Hub), `docs/decisions/001-released-checkpoint.md`
- Modify: `notebooks/*` (point the checkpoint-download cell at the real HF repo id), `README.md` (results table)

**Steps:**

- [ ] **Step 1:** Run `python -m ch04.train --config configs/full.yaml` on the agreed GPU. During the run, canary snapshots every 5k steps (feeds Fig 4.8) and 50-seed closed-loop eval per checkpoint (§4.6.7 cadence).
- [ ] **Step 2:** Verify the manuscript's reproducibility claims: loss ≈3.5 @ 500 / ≈2.0 @ 5k / ≈1.0 @ 20k steps; closed-loop success in the 60–80% band. Where reality diverges, record the real numbers in `docs/manuscript_fixes.md` (the book must match the repo, not vice versa).
- [ ] **Step 3:** Upload best checkpoint + tokenizer stats + config + metrics via `scripts/release_checkpoint.py` to a user-approved HF Hub repo id (**ask the user before publishing** — this is a public artifact). Update the notebook download cell and README results table; regenerate `ch04_executed.ipynb` from the released artifacts.
- [ ] **Step 4:** Final full-gate re-run (ruff, unit, integration, notebook execution) and commit: `Ch4 (v2) PR10: full-recipe checkpoint + release`.

---

## Self-review notes

- Spec coverage: every manuscript listing (4.1, 4.3, 4.4, 4.10, 4.12, 4.13, 4.14) maps to Tasks 9, 2, 2, 5, 6, 7, 9 respectively; prose-only promises (ParallelActionHead §4.5.2, smoothing, sampling table 4.4, diagnostics §4.6.7, figures 4.8–4.10) map to Tasks 4, 4, 8, 9, 9. §4.7 (world models) is survey-only — no code, correctly absent.
- Exercises 4.2/4.3/4.4/4.6 are reader exercises; the repo provides the hooks they need (focal loss is a ~5-line swap documented in README's exercises section — deliberate non-goal to implement solutions).
- Type consistency: `target_bins` is `[B, H*D]` for the AR head (flattened, per Listing 4.13) and `[B, H, D]` for the parallel head's loss — README must state this asymmetry; tokenizer works on `[..., D]` and is shape-agnostic.
