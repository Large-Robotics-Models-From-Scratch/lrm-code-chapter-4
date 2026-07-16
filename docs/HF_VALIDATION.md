# Validation against LeRobot's official SmolVLA training path

**Date:** 2026-07-15 · **Ch 4 owner:** Vatsal
**References cross-checked** (raw, `main` branch, 2026-07-15):
- `huggingface/lerobot` `src/lerobot/scripts/lerobot_train.py` + `optim/`, `datasets/factory.py`, `policies/smolvla/*`
- `huggingface/notebooks` `lerobot/training-smolvla.ipynb` + docs `huggingface.co/docs/lerobot/smolvla`

Our repo trains a **from-scratch discrete** BC head (OpenVLA-style reserved-vocab tokens
decoded through SmolLM). SmolVLA is a **continuous flow-matching** policy with a separate
action expert. Where they legitimately differ, it's method-driven; the mechanics that should
match, match.

## What matches (validated ✓)

| Aspect | Ours | LeRobot / SmolVLA | Verdict |
|---|---|---|---|
| Optimizer | AdamW | AdamW (`AdamWConfig`) | ✓ |
| Betas | (0.9, 0.95) | (0.9, 0.95) | ✓ exact |
| Head LR | 1e-4 | `optimizer_lr = 1e-4` | ✓ exact |
| Schedule | linear warmup → cosine decay | `CosineDecayWithWarmupSchedulerConfig` | ✓ same shape |
| Scheduler step cadence | per optimizer step | "step at every batch instead of epoch" | ✓ |
| Chunk via `delta_timestamps` | `[i/fps for i in range(H)]` | `[i/ds_meta.fps for i in action_delta_indices]` | ✓ identical mechanism |
| **Pad masking in loss** | expand `action_is_pad` [B,H]→[B,H·D], zero masked, divide by unmasked count (`clamp_min(1)`) | `losses *= (~actions_is_pad); num_valid = ((~pad).sum()*D).clamp_min(1); loss = losses.sum()/num_valid` | ✓ **near-identical** |
| **Action space (so101)** | absolute joint targets, binned q01/q99 | absolute (`use_delta_joint_actions_aloha=False`; delta only for Aloha, else `NotImplementedError`) | ✓ confirms absolute |
| **Eval posture** | open-loop val loss primary; sim optional & caveated | `eval_steps` (offline loss) and `env_eval_freq` (sim) **both optional, both off by default**; comment: real-world data is evaluated **outside** train.py on the real robot | ✓ **matches our Option 1** |
| Mixed precision | bf16 autocast (fp16+GradScaler fallback) | via `accelerate` autocast; bf16/fp16 | ✓ (we hand-roll it; a from-scratch book shouldn't hide it in `accelerate`) |

The two most consequential confirmations:

1. **Absolute action space is confirmed by LeRobot's own SmolVLA config** — so101 actions are
   absolute joint positions (delta conversion is Aloha-only and unported). This backs the
   central claim in `EVAL_INCONSISTENCY.md`: feeding absolute-target actions into the sim's
   `pd_joint_delta_pos` (delta) controller is a units error, not a trainable gap.
2. **Open-loop-primary / sim-optional is LeRobot's own default**, not just a workaround. In
   `lerobot_train.py`, offline validation loss and closed-loop sim eval are **both gated off by
   default**, and the script explicitly notes real-data policies are evaluated outside it on the
   real robot. The official SmolVLA notebook does **no** eval at all (no gym, no ManiSkill).
   Our Option-1 pivot is the canonical posture.

Our empirically-derived `action_is_pad` masking (found in Task 6, implemented in Task 7)
reproduces SmolVLA's exact loss-masking formula — independent convergence on the same code.

## Where we differ (all defensible; flagged for the author)

These are candidates for `manuscript_fixes.md` / a Table 4.3 footnote — **not bugs**, but points
where the manuscript's recipe diverges from the production reference and a reader might ask why.

| Aspect | Ours (manuscript Table 4.3) | SmolVLA | Note |
|---|---|---|---|
| Weight decay | **0.05** | **1e-10** (≈ none) | Biggest hyperparameter gap. 0.05 is standard AdamW; SmolVLA runs almost none. Worth a one-line justification or reconsideration. |
| Grad-clip norm | 1.0 | 10 | Ours is more conservative for a random-init head; fine, but note it. |
| Warmup steps | 500 | 1000 | Minor; both scale with total steps. |
| Cosine floor | → 0 | → `decay_lr = 2.5e-6` | Negligible. |
| Backbone freeze | freeze SmolLM epoch 0, then **unfreeze** at 1e-5 (split LR) | **permanent** freeze of vision encoder **and** whole VLM; only the action expert + state proj train | Architecture-driven: SmolVLA has a *separate* action expert so it never trains the VLM; we reuse SmolLM *as* the decoder (OpenVLA-style), so we must train it. Our freeze-then-unfreeze is the manuscript's §4.5.6 PITFALL mitigation — SmolVLA sidesteps the same corruption risk by never unfreezing. Worth explaining the contrast in prose. |
| Action normalization | q01/q99 **quantile** binning (discrete) | **MEAN_STD** (continuous) | Method-appropriate: discrete heads want bounded quantile ranges (RT-1/OpenVLA); continuous heads want mean/std. LeRobot's `NormalizationMode` does offer `QUANTILES`, so our choice is a supported mode, correct for discrete BC. |
| Chunk length H | 16 | 50 (`chunk_size`) | Manuscript already acknowledges "π0 uses H=50." No change needed. |
| Dataset variant | `svla_so101_pickplace` | notebook uses `svla_so100_pickplace` (SO100) | Both are official LeRobot demo sets; ours is the SO101 revision. Consistent with Ch 2/3's SO101 anchor. |

## Recommended actions

1. **Strengthen `EVAL_INCONSISTENCY.md`** with the LeRobot-config confirmation that so101 actions
   are absolute (done conceptually here; cross-link).
2. **Add the weight-decay (0.05 vs ~0) and freeze-strategy divergences to `manuscript_fixes.md`**
   as author-decision items (after Task 10.6's manuscript_fixes edits land, to avoid a merge race).
3. **Option 1 is validated** — no change to the open-loop pivot; if anything, it's now backed by
   LeRobot's own default configuration, not just community practice.

## Sources
- lerobot_train.py, optim/factory.py, optim/optimizers.py, optim/schedulers.py, datasets/factory.py,
  policies/smolvla/{configuration,modeling}_smolvla.py, smolvlm_with_expert.py, configs/types.py —
  `raw.githubusercontent.com/huggingface/lerobot/main/...`
- training-smolvla.ipynb — `raw.githubusercontent.com/huggingface/notebooks/main/lerobot/`
- https://huggingface.co/docs/lerobot/smolvla
