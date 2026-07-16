# Manuscript fixes — Chapter 4

Discrepancies found while implementing the companion code against the
real pinned stack (transformers 5.3.0 / lerobot 0.5.1 / torch 2.10.0 /
mani-skill 3.0.1). Each item cites the manuscript location in
`../lrm-book/chapter_4/manuscript/chapter_4_v3.md` (line numbers as of
this writing) and states the reconciliation the code adopts. This file
is the single place the chapter text should be checked against before
the next revision.

Severity legend: **[bug]** wrong/won't-run as printed · **[gap]** code
must exist that the prose only implies · **[note]** correct but worth a
caveat in the text.

---

## 1. [bug] Stale LeRobot import path — Listings 4.4 and 4.12

- **Where:** line 173 (Listing 4.4) and line 386 (Listing 4.12):
  `from lerobot.common.datasets.lerobot_dataset import LeRobotDataset`.
- **Problem:** lerobot 0.5.1 moved this module. The `lerobot.common.*`
  path does not exist.
- **Fix:** `from lerobot.datasets.lerobot_dataset import LeRobotDataset`
  (used in `src/ch04/chunk_data.py`).

## 2. [bug] Hardcoded 50 Hz chunk timestamps — Listing 4.12

- **Where:** line 394:
  `"action": [t / 50.0 for t in range(16)]  #C H=16 @ 50 Hz`.
- **Problem:** `svla_so101_pickplace` is **30 fps**, not 50. Dividing
  by 50 asks lerobot for offsets that span the wrong wall-clock window.
- **Fix:** derive the offsets from the dataset's own fps:
  `t / ds.fps`. See `chunk_delta_timestamps(fps, chunk_h)` in
  `src/ch04/chunk_data.py` (the notebook prints `episode.fps == 30`).

## 3. [gap] `fusion.*` surface does not exist on the ch3 backbone — Listing 4.10, 4.15

- **Where:** line 319 `self.fusion.embed(tok_in)`, line 320
  `hs = self.fusion(seq, causal=True)`, line 481
  `self.fusion.encode_prefix(obs)`, lines 487–488
  `self.fusion(seq, past_kv=past_kv, causal=True)`.
- **Problem:** Chapter 3 ships `UnifiedEmbeddingBackbone`, which exposes
  none of `embed` / `encode_prefix` / a `causal=`-keyworded call. The
  listings assume a `fusion` object that does not exist.
- **Fix:** `src/ch04/fusion_adapter.py` (`FusionAdapter`) *composes* the
  ch3 backbone read-only and provides exactly that surface
  (`encode_prefix`, `embed`, `forward(seq, past_key_values, use_cache)`).
  The manuscript should either introduce the adapter or note the code
  supplies it. (The `causal=True` keyword is also moot: SmolLM2 is a
  causal decoder, so the mask is inherent — no keyword needed.)

## 4. [note] "mask the non-action vocabulary ids to −∞" is moot — §4.5.5

- **Where:** line 334 (§4.5.5): the decode "mask[s] the non-action
  vocabulary ids to `−∞` first so the head cannot emit an English word."
- **Problem:** the shipped head reads out through a **256-way** `Linear`
  (`AutoregressiveActionHead.readout`), so it can only ever emit one of
  the 256 action bins. There is no full-vocabulary softmax to mask.
- **Fix:** drop the masking sentence, or reframe it as describing the
  alternative design where the head reads out over the full 49,152-id
  vocabulary (which this repo deliberately does not do).

## 5. [gap] `ParallelActionHead` has no listing — §4.5.2

- **Where:** §4.5.2 (lines 276–280) describes the parallel head in prose
  (two-layer MLP, `d_embed → H·D·n_bins`, per-position softmax) with no
  code listing, while the AR head gets Listing 4.10.
- **Fix:** the repo synthesizes it in
  `src/ch04/parallel_action_head.py`, matching the prose (1024-wide GELU
  hidden layer, zeroed final bias, optional adjacent-bin label
  smoothing). Consider adding a listing or pointing at the module.

## 6. [bug] Dataset stats ship no `q01`/`q99` — Listing 4.4

- **Where:** line 177: `lo, hi = np.array(stats["q01"]),
  np.array(stats["q99"])`.
- **Problem:** `svla_so101_pickplace`'s `meta/stats.json` for `action`
  carries only `min/max/mean/std/count` — **no** `q01`/`q99`. Listing
  4.4 raises `KeyError` as printed.
- **Fix:** `ActionTokenizer.from_lerobot_dataset` computes the 1st/99th
  percentiles directly from the action column
  (`np.percentile(actions, [1, 99], axis=0)`). `from_lerobot_stats`
  still accepts stats that *do* carry the percentiles, and raises a
  clear error pointing at `from_lerobot_dataset` when they are absent.

## 7. [note] Fresh-init loss ≈ log(256) holds on a *fake* batch only — §4.5.2, §4.5.4

- **Where:** line 280 ("near `log(256) ≈ 5.54` nats"), line 330 ("on a
  fake batch … again sits near `log(256) ≈ 5.54`"), line 437 ("loss
  starts near `log(256) = 5.54`").
- **Problem:** the exact 5.545 holds when the pooled state is neutral
  (a fake batch). On the **real** 576-dim backbone the fused prefix is
  not neutral, so the fresh loss lands modestly above it. Measured on
  this repo's CPU run: AR head ≈ **6.1 nats**, parallel head ≈ **9.9
  nats** (its random MLP first layer is not near-uniform). Both are in
  the uniform *scale* band (far below the ~11–13 a confidently-wrong
  head reports), but neither is 5.545.
- **Fix:** keep 5.545 as the *fake-batch* sanity value (the text at
  line 330 already says "on a fake batch"), but where the check is
  described against the real backbone (§4.5.2 line 280, §4.6 line 437)
  add "modestly above `log(256)` on the real fused prefix." The unit
  test `tests/test_autoregressive_action_head.py` asserts the loss is
  within 2.0 of `log(256)` rather than equal to it, for this reason.

## 8. [bug] LeRobot does NOT assert chunks stay in-episode — it pads — §4.6.1, Listing 4.12

- **Where:** line 403: "`delta_timestamps` … asserts at load time that
  no chunk crosses an episode boundary: ask for `H = 16` on a shorter
  episode and construction raises immediately."
- **Problem:** false for lerobot 0.5.1. Boundary chunks do **not**
  raise. lerobot clamps the out-of-range frame index (repeating the
  last action) and returns a companion `action_is_pad` boolean mask
  marking the repeated steps. Nothing raises at load time.
- **Fix:** the training loss must **mask** on `action_is_pad`. The AR
  head's `forward(prefix, bins, pad_mask=...)` and the parallel head's
  `loss(..., pad_mask=...)` drop padded positions from the mean;
  `train._prepare_batch` expands the `[B, H]` frame mask to token level
  (`repeat_interleave` by `ACTION_DIM`). The manuscript should replace
  the "asserts / raises immediately" sentence with the pad-and-mask
  behavior.

## 9. [note] Chunked loader returns `observation.state` as `[B, 1, 6]` — §4.6

- **Where:** implied by Listing 4.12 / the training loop (Listing 4.13
  feeds `batch` straight to `fusion.encode_prefix`).
- **Problem:** lerobot 0.5.1 squeezes the single-offset **video** keys
  to `[B, 3, H, W]` but keeps the chunk dim on **tabular** keys, so
  `observation.state` arrives `[B, 1, 6]`, not `[B, 6]`. Fed as-is, the
  ch3 state encoder sees a stray chunk axis.
- **Fix:** `squeeze(1)` the state before the backbone
  (`train._prepare_batch` does this and asserts the `[B, 1, 6]` shape).

## 10. [note] Worked quantization example is correct for [−1, 1] — §4.3 (line 133)

- **Where:** line 133: `0.347` rad over `[−1.0, 1.0]`, `B = 256` →
  `0.674 → 172.5 → bin 172`, center ≈ `0.348` rad, error ≈ `0.001`.
- **Status:** **arithmetically correct** as printed for the `[−1, 1]`
  range (re-derived: bin 172, center 0.34766 rad, error 0.00066 rad).
  An *earlier* draft used a `[−π, π]` framing where 0.347 lands in bin
  142 (center ≈ `0.356` rad, error ≈ `0.009` rad) while still quoting
  the `[−1, 1]` numbers — that mismatch is what the CLAUDE.md
  "known follow-up" note flags. The current v3 line 133 is consistent.
- **Fix:** none needed for v3 line 133. `tests/test_action_tokenizer.py
  ::test_worked_numeric_example` derives both ranges from the formula so
  the prose cannot silently drift again. (The notebook reproduces the
  `[−1, 1]` example on a dedicated tokenizer to match the book exactly,
  since the *real* dataset's per-joint ranges are raw SO-101 units, not
  `[−1, 1]`.)

## 11. [note] Single-camera gap is EVAL-ONLY (sim), not a training issue — Listing 4.14, §4.6

- **Training is correctly two-view.** The real dataset
  `lerobot/svla_so101_pickplace` genuinely ships **two** cameras —
  `observation.images.up` and `observation.images.side`, each
  480×640×3 — and ch3's `UnifiedEmbeddingBackbone` consumes both
  (`NUM_CAMERAS = 2`, `IMAGE_TOKENS = 392 = 2 × 196`). Behavior cloning
  in this chapter trains on both views as intended; there is **no**
  camera problem at training time, and the notebook's dataset-viz cell
  plays back both cameras side by side.
- **Where the gap is:** line 453 (Listing 4.14), the closed-loop
  rollout **only**. The ManiSkill sim env `PickCubeSO100-v1` exposes a
  **single** `base_camera` (128×128), so the two-view backbone has no
  second view at eval time. The rollout adapter
  (`rollout.maniskill_obs_adapter`) duplicates the one sim view into
  both camera slots — an **eval-only** sim-vs-dataset domain gap, not a
  training or backbone defect.
- **Fix:** note the single-view caveat where the *rollout* is
  introduced (not where training is described), and treat sim success
  rates as a lower bound relative to the two-view training
  distribution. If a two-camera SO-100 sim config becomes available,
  the gap closes with no code change to training.

## 12. [note] transformers 5.x loads SmolLM2 in bfloat16 — affects every forward

- **Where:** not stated in the manuscript; affects all forward-pass
  listings (4.10, 4.13, 4.14, 4.15).
- **Problem:** `AutoModel.from_pretrained("…SmolLM2-135M")` under
  transformers 5.x loads the checkpoint's native **bfloat16** (4.x
  defaulted to float32), while SigLIP loads float32 — so the first
  cross-stream matmul in ch3's forward raises a dtype mismatch.
- **Fix:** load the backbone with an explicit float32 (`.float()`), as
  every training/eval path and the notebook do. Cross-entropy also
  upcasts logits to float32 before the loss (bf16 softmax is too coarse
  at the `~log(256)` scale). Worth a one-line dtype note in §4.5.

## 13. [bug] Closed-loop-on-`PickCubeSO100` eval is domain-mismatched — §4.6.4, §4.8

- **Where:** §4.6.4 (Listing 4.14) runs the policy **closed-loop** in
  the ManiSkill `PickCubeSO100-v1` sim, and §4.8 claims "roughly
  **60–80% closed-loop success**" for the discrete head.
- **Problem:** the policy is trained on the *real* `svla_so101_pickplace`
  dataset (SO-101, two cameras, **absolute** joint-target actions) but
  `PickCubeSO100-v1` is a different domain (SO-100, one camera, a
  **delta**-action controller `pd_joint_delta_pos`). The absolute-vs-delta
  action-space gap alone breaks the rollout, so the shipped code would
  reproduce near-0%, not 60–80% — and the failure would be attributable
  to the train/eval domain gap, not the discrete-BC method the chapter
  teaches. This is a cross-chapter contract issue (the env and its
  control mode are established in Ch 2 and ratified in Ch 3), not a
  Ch 4-local choice.
- **Fix:** the repo makes **open-loop the primary eval** — held-out
  validation loss (`rollout.evaluate_open_loop`), predicted-vs-expert
  action traces (`diagnostics.plot_action_traces`), and the
  multimodality diagnostics (Figures 4.8–4.10) — and treats the
  ManiSkill closed-loop rollout as **optional/aspirational**, clearly
  caveated and GPU-gated, with **no success number claimed**. The
  manuscript should drop the "60–80% closed-loop success" figure (or
  reframe it as aspirational, pending a matched-domain sim) and point to
  `docs/EVAL_INCONSISTENCY.md`, which documents every mismatch and the
  author's decision (Option 1). A matched-domain closed-loop eval is
  deferred to the ch2/ch3 owners.

## 14. [note] Weight decay 0.05 vs SmolVLA's ~0 — §4.6.1, Table 4.3

- **Where:** Table 4.3 (recipe) and Listing 4.13: `weight_decay=0.05`.
- **Observation:** LeRobot's official SmolVLA policy config sets
  `optimizer_weight_decay = 1e-10` (effectively none), while betas
  `(0.9, 0.95)` and head LR `1e-4` match ours exactly (validated against
  `huggingface/lerobot` `policies/smolvla/configuration_smolvla.py`, see
  `docs/HF_VALIDATION.md`). 0.05 is a standard AdamW value and not wrong,
  but it is the single largest hyperparameter gap from the production
  reference.
- **Fix:** author decision — justify 0.05 in a sentence, or lower it
  toward the SmolVLA setting. No code change made; Table 4.3 is the
  source of truth the code follows.

## 15. [note] Backbone freeze strategy differs from SmolVLA — §4.5.6

- **Where:** §4.5.6 (freeze epoch 0, then unfreeze SmolLM at LR 1e-5).
- **Observation:** SmolVLA **permanently** freezes the vision encoder
  *and* the whole VLM (`freeze_vision_encoder=True`,
  `train_expert_only=True`) and trains only a separate action expert.
  Ours reuses SmolLM *as* the decoder (OpenVLA-style), so it must be
  trained — hence freeze-then-unfreeze, which the manuscript's §4.5.6
  PITFALL motivates. This is an architecture-driven difference, not an
  error, but the contrast (production VLAs keep the VLM frozen behind a
  separate expert) is worth a sentence so a reader who compares to
  SmolVLA isn't confused. See `docs/HF_VALIDATION.md`.
- **Fix:** author decision — add one clarifying sentence; no code change.
