# ADR 000: Environment pins and contract verification

Status: Accepted
Date: 2026-07-12
Task: Task 0 (environment + contract verification spike)

This ADR records the dependency pin set that lets Chapter 4 load the **real**
`lerobot/svla_so101_pickplace` dataset in the **same** environment that loads
SmolLM2 and SigLIP and imports the Chapter 3 backbone. Every value below was
produced by a command actually run against the built environment; the exact
commands are logged in `.superpowers/sdd/task-0-report.md`.

---

## Outcome

**One environment holds the entire stack — no split env, no synthetic-data
fallback, no STOP gate.** The lerobot ↔ transformers/huggingface-hub conflict
that Chapter 3 documented is resolved by moving transformers to the **5.x**
line (`transformers==5.3.0`), which is exactly the version lerobot 0.5.1's own
`transformers-dep` extra pins.

## Toolchain note (important for reproducibility)

The machine's Homebrew `python@3.12` (3.12.13_2) is **broken**: its
`pyexpat.cpython-312-darwin.so` references `_XML_SetAllocTrackerActivationThreshold`
from `/usr/lib/libexpat.1.dylib`, a symbol the OS-provided libexpat does not
export. This makes `ensurepip` / `pip` unusable in any venv built from that
interpreter, and dyld env-var overrides cannot fix it (the reference is an
absolute install-name). The venv is therefore built from a **uv-managed
standalone CPython 3.12.13** (`uv python install 3.12`), which bundles its own
expat and works cleanly. Packages are installed with `uv pip install --python
.venv/bin/python` (the venv has no `pip` binary; use `uv pip`).

## Pin table (from `importlib.metadata.version`, built env)

| Package          | Version  | Notes |
|------------------|----------|-------|
| python           | 3.12.13  | uv standalone build (not Homebrew — see toolchain note) |
| torch            | 2.10.0   | downgraded from 2.13.0 by lerobot 0.5.1's constraints |
| torchvision      | 0.25.0   | pulled by lerobot |
| torchcodec       | 0.10.0   | lerobot video decode backend |
| transformers     | 5.3.0    | **key pin** — 5.x accepts huggingface-hub 1.x; matches lerobot's `transformers-dep` extra |
| tokenizers       | 0.22.2   | |
| huggingface-hub  | 1.23.0   | lerobot 0.5.1 requires `>=1.0.0,<2.0.0` |
| lerobot          | 0.5.1    | target pin |
| numpy            | 2.2.6    | downgraded from 2.5.1 by lerobot |
| mani-skill       | 3.0.1    | installs additively — no conflict with training env |
| sapien           | 3.0.3    | mani-skill physics backend |
| gymnasium        | 1.3.0    | satisfies both lerobot and mani-skill |
| pyarrow          | 25.0.0   | lerobot parquet |
| pandas           | 3.0.3    | lerobot |
| av               | 15.1.0   | lerobot video (see ffmpeg dylib note below) |
| safetensors      | 0.8.0    | |

Chapter 3 (`lrm-ch03==0.1.0`) is installed **editable** from
`../lrm-code-chapter-3` (tip `deeead0`, `main`).

### The conflict and why 5.3.0 resolves it

- lerobot 0.5.1 metadata: `huggingface-hub<2.0.0,>=1.0.0`, and its
  `transformers-dep` extra pins `transformers==5.3.0`.
- transformers **4.57.6** hardcodes a *runtime* check
  (`transformers/dependency_versions_check.py`) requiring
  `huggingface-hub>=0.34.0,<1.0`. uv's resolver does **not** catch this (the
  package metadata bound is looser than the runtime assertion), so a
  `lerobot==0.5.1 + transformers==4.57.6` env installs "successfully" but
  **fails at `import transformers`** with:
  `ImportError: huggingface-hub>=0.34.0,<1.0 is required ... but found huggingface-hub==1.23.0`.
- transformers **5.3.0** drops the `<1.0` hub ceiling, so lerobot 0.5.1 +
  huggingface-hub 1.23.0 + transformers 5.3.0 import and run together.

### Action required in Chapter 3's pyproject (follow-up, not this task)

`../lrm-code-chapter-3/pyproject.toml` pins `transformers>=4.45,<5.0`. That
upper bound is **incompatible** with the pin set above. Chapter 4 installs
`transformers==5.3.0` explicitly (which overrides the editable ch3 constraint
at install time), but for a clean reproducible install the ch3 pin should be
loosened to allow 5.x. The ch3 backbone code itself runs on 5.3.0 (verified
below) with one required adaptation (bfloat16 default, see next section).

---

## Backbone contract (Chapter 3 → Chapter 4)

Import path and constants — assert passes:

```python
from ch03 import UnifiedEmbeddingBackbone
from ch03.vla_backbone import SMOLLM_VOCAB, SMOLLM_WIDTH, IMAGE_TOKENS
assert (SMOLLM_VOCAB, SMOLLM_WIDTH, IMAGE_TOKENS) == (49152, 576, 392)  # PASSES
```

- Models: `HuggingFaceTB/SmolLM2-135M` (language), `google/siglip-base-patch16-224`
  (vision, `SiglipVisionModel`).
- `forward(images, sequence_ids, state)` → `[B, 392 + L + 1, 576]`. Verified
  with a real forward: instruction "pick up the red cube" → L=5 tokens → output
  `(1, 398, 576)`. **Correct.**

### transformers 5.x migration wrinkle (feed into Task 1)

Under transformers 5.3.0, `AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")`
loads weights in the checkpoint's **native bfloat16** (transformers 4.x defaulted
to float32). SigLIP still loads float32. The ch3 `forward` therefore raises
`RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16`
on the first backbone matmul. Casting the module with `.float()` (or loading with
an explicit `dtype=torch.float32`) fixes it and yields the correct `(1, 398, 576)`
output. **Task 1 (scaffold rebuild on v5 contract) must set an explicit dtype**
rather than relying on the 4.x float32 default. The `SiglipVisionModel LOAD REPORT`
listing "UNEXPECTED text_model.* keys" is benign — only the vision tower is used.

---

## Dataset facts: `lerobot/svla_so101_pickplace`

`LeRobotDataset` import path (lerobot 0.5.1):
`from lerobot.datasets.lerobot_dataset import LeRobotDataset`

| Fact | Value |
|------|-------|
| `ds.fps` | 30 |
| `ds.num_episodes` | 50 |
| `ds.num_frames` | 11939 |
| action dim | 6 (`action`, dtype float32, shape `(6,)`) |
| state dim | 6 (`observation.state`, dtype float32, shape `(6,)`) |
| cameras | `observation.images.up`, `observation.images.side` — video, `(480, 640, 3)`; decoded per-sample as CHW `torch.Size([3, 480, 640])` float32 |

Full feature keys: `action`, `observation.state`, `observation.images.up`,
`observation.images.side`, `timestamp`, `frame_index`, `episode_index`,
`index`, `task_index`.

### Action stats — NO q01/q99 (decision for the tokenizer)

`ds.meta.stats["action"]` sub-keys are **`min, max, mean, std, count`**.
**`q01`/`q99` are absent.** Values (per the 6 action dims):

- `min`  = [-93.4559, -100.0, 12.9722, 33.5317, -92.7717, 0.0]
- `max`  = [88.0147, 8.1263, 100.0, 99.49, -20.0, 32.9985]
- `mean` = [8.0211, -55.9624, 65.2557, 69.1819, -53.4199, 6.8489]
- `std`  = [44.563, 36.4851, 29.012, 13.2381, 17.7644, 8.999]
- `count` = 11939

**Decision:** the action tokenizer's `from_lerobot_stats` **cannot** read
`q01`/`q99` from `ds.meta.stats`. It must compute the 1st/99th percentiles
itself with a single pass over the dataset's `action` column (11,939 rows — a
cheap in-memory numpy `np.percentile` on the concatenated actions). Do **not**
fall back to raw `min`/`max` for tokenization bounds: `min`/`max` here contain
saturated endpoints (e.g. exactly `-100.0`, `100.0`, `0.0`) that would waste bin
resolution on outliers. Record this in Task 2 (PR1 action tokenizer).

---

## Simulator env id (ManiSkill 3.0.1)

`import mani_skill.envs` registers 137 gymnasium envs. Registry search results:

- **`PickCubeSO100-v1` IS registered** (the brief was uncertain — it exists).
- SO100/SO101 ids present: `PickCubeSO100-v1`, `SO100GraspCube-v1`.
- Other pick-cube envs: `PickCube-v1`, `PickCubeWidowXAI-v1`, `TwoRobotPickCube-v1`.

**Decision:** chosen env id **`PickCubeSO100-v1`**; fallback **`SO100GraspCube-v1`**.
ManiSkill 3.0.1 co-installs into the *same* venv as the training/data stack with
no conflict (additive install; no downgrade of torch, transformers, lerobot, or
gymnasium), so there is **no need for a second venv**. On this macOS host SAPIEN
warns `Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.` —
registration works; actual GPU rollout rendering is expected to require a
CUDA/Vulkan host (the A100 run in Task 11).

---

## Download / disk / wall-clock numbers (for the README)

Measured on first run (uv cache warm for wheels; HF downloads cold):

| Item | Size on disk | Notes |
|------|--------------|-------|
| SmolLM2-135M (HF cache) | 260 MB | `~/.cache/huggingface/hub` |
| SigLIP base-patch16-224 (HF cache) | 775 MB | `~/.cache/huggingface/hub` |
| HF model hub total | 1.0 GB | |
| `svla_so101_pickplace` dataset | 82 MB | `~/.cache/huggingface/lerobot/hub/` — **not ~1 GB**; lerobot streams video lazily via torchcodec |
| `.venv` | 2.0 GB | full stack incl. torch + mani-skill/sapien |

Wall-clock: backbone instantiation (SmolLM2 + SigLIP download + load) ≈ 21 s
cold; dataset first load ≈ 4 s (metadata + parquet fetch).

### Known benign warning

`av` bundles `libavdevice.61` while the host Homebrew ffmpeg provides
`libavdevice.62`; importing the dataset prints an `objc[...] Class
AVFFrameReceiver is implemented in both ...` duplicate-class warning. It is
cosmetic here (dataset loads and samples decode correctly), but worth silencing
or pinning ffmpeg if it ever causes decode instability.

---

## Reproduce

```bash
# 1. Toolchain: uv-managed standalone CPython (Homebrew py3.12 is broken here)
uv python install 3.12
uv venv --python 3.12 .venv

# 2. Chapter 3 backbone (editable) + data + sim, in one env
uv pip install --python .venv/bin/python -e ../lrm-code-chapter-3
uv pip install --python .venv/bin/python "lerobot==0.5.1"
uv pip install --python .venv/bin/python "transformers==5.3.0"   # overrides ch3's <5.0 pin
uv pip install --python .venv/bin/python "mani-skill==3.0.1"
```
