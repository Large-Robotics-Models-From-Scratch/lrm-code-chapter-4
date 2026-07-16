# Cross-chapter eval inconsistency: real-data training vs. sim evaluation

**Author:** Ch 4 (discrete BC) owner — Vatsal
**For discussion with:** Ch 2 owner (Siddharth), Ch 3 owner (Krishnam)
**Date:** 2026-07-15
**Status:** Open question — blocks the Chapter 4 closed-loop eval story (§4.6.4, §4.8) and the companion-repo `rollout.py`.

---

## TL;DR

Chapter 4 trains a behavior-cloning policy on the **real teleoperated dataset**
`lerobot/svla_so101_pickplace` (SO-101 hardware, two cameras, absolute joint-target
actions) and then evaluates it **closed-loop inside the ManiSkill `PickCubeSO100-v1`
simulator** (SO-100 model, one camera, delta-action controller). These two are not a
matched train/eval pair, and at least one mismatch (action space: absolute targets vs.
joint deltas) breaks the rollout independently of the others. The manuscript's claim of
"~60–80% closed-loop success" (§4.8) is unlikely to reproduce, and the failures would be
attributable to the domain gap rather than to the discrete-BC method the chapter teaches.

The eval env was **established in Chapter 2** (`ch02.env.make_env`, re-exported "for
eval-rollout chapters") and **ratified in Chapter 3's ARCHITECTURE_LOG** as a settled
cross-chapter contract ("same 6-DOF action interface"). So this is a shared decision, not
a Ch 4-local choice — hence this write-up for the three of us.

The community does **not** evaluate `svla_so101_pickplace`-trained policies in ManiSkill.
`PickCubeSO100-v1` comes from a sim2real project that runs the **opposite** direction
(train in sim → deploy to real robot).

---

## 1. The three objects and where each is defined

| Object | What it is | Embodiment | Cameras | Action space | Defined in |
|---|---|---|---|---|---|
| `lerobot/svla_so101_pickplace` | Real teleoperated **dataset**, 50 ep / 11,939 frames / 30 fps | **SO-101** hardware | **2**: `observation.images.up`, `observation.images.side` (480×640×3) | **Absolute** joint targets, 6-dim (leader-arm positions) | Loaded in `ch02` dataloader; anchor dataset for Ch 2/3/4 |
| `PickCubeSO100-v1` | ManiSkill3 **sim env** | **SO-100** model | **1**: `base_camera` (128×128) | Controller-dependent; `ch02.env.make_env` default = `pd_joint_delta_pos` (**deltas**) | `ch02/env.py`; registered by `mani_skill.envs` |
| SO-101 hardware | Real robot | SO-101 | (deployment) | — | Ch 9 (sim-to-real), later |

Verified 2026-07-15 by loading the dataset (`ds.meta.features` shows both `up` and `side`
at 480×640×3) and by reading `mani_skill` env source (`PickCubeSO100-v1` exposes a single
`base_camera`).

## 2. The decision chain across chapters (who established what)

- **Chapter 2** (`lrm-code-chapter-2`):
  - `src/ch02/env.py:16-37` — `make_env()` wraps `gym.make("PickCubeSO100-v1", ...)` with
    **`control_mode="pd_joint_delta_pos"`** (joint-space **deltas**) as the default.
  - `src/ch02/__init__.py:6` — "`make_env` is re-exported **for eval-rollout chapters**."
  - `README.md:5` — "the carrier embodiment is the **SO-100 arm in sim** transitioning to
    **SO-101 hardware** in the later deployment chapters."
  - Chapter 2 **also** builds the `svla_so101_pickplace` LeRobot loader — so the SO-100-sim /
    SO-101-dataset pairing originates here.
- **Chapter 3** (`lrm-code-chapter-3`):
  - `ARCHITECTURE_LOG.md:37` — "**Sim env vs dataset distinction**: `PickCubeSO100-v1` is the
    SIM env (ManiSkill3 ships SO-100); `svla_so101_pickplace` is the DATASET (SO-101 real
    teleoperated); SO-101 hardware in Ch 9. **Same 6-DOF action interface.**"
  - `ARCHITECTURE_LOG.md:91` — "`from ch02.env import make_env  # optional, for eval rollouts`."
  - `UnifiedEmbeddingBackbone` consumes **two** cameras (`vla_backbone.py`: `NUM_CAMERAS = 2`,
    `IMAGE_TOKENS = 392 = 2 × 196`), i.e. training is correctly two-view from the real dataset.
- **Chapter 4** (this repo / manuscript `chapter_4_v3.md`):
  - Listing 4.14 (§4.6.4) — `gym.make("PickCubeSO100-v1", obs_mode="rgb")`, 50-seed rollout,
    `info["success"]`.
  - §4.8 — claims "roughly **60–80% closed-loop success**" for the discrete head, vs "~95% for
    a continuous head like Diffusion Policy."
  - Companion `src/ch04/rollout.py` — `evaluate()` calls `gym.make(env_id, obs_mode="rgb")`
    and does **not** set `control_mode` (so it inherits the env's registered default, which is
    not guaranteed to be the dataset's absolute-target space).

The line that papers over the problem is Ch 3's **"same 6-DOF action interface."** It is true
that both are 6-dimensional. It is not true that they are the same *space* or the same
*rendering distribution*, which is what a closed-loop rollout requires.

## 3. The concrete mismatches, ranked

### 3.1 Action space: absolute targets vs. joint deltas — **breaks eval by itself**
- The dataset `action` is **absolute joint-target positions** (LeRobot SO-100/101 teleop
  records the leader arm's joint positions). The Ch 4 `ActionTokenizer` bins each dimension
  between its `q01`/`q99` **absolute** range and decodes to a bin **center** — an absolute
  target.
- `ch02.env.make_env` default controller is **`pd_joint_delta_pos`** — the env interprets an
  action as a **delta** to add to the current joint position.
- Feeding an absolute-target command (e.g. "shoulder = 1.2 rad") into a delta controller
  ("add 1.2 rad to the current shoulder angle") is not a domain gap you can train through —
  it is a units error. The arm will fling.
- `rollout.py` not setting `control_mode` makes it worse: the result depends on whatever
  `PickCubeSO100-v1` registers as its default, which we have not pinned to the dataset's space.
- **This one is fully in our control to get right or to declare unbridgeable**, independent of
  vision.

### 3.2 Visual domain gap: real camera frames vs. SAPIEN renders
- The policy's vision encoder (frozen SigLIP from Ch 3) is fed **real** `up`/`side` camera
  pixels in training and would be fed **SAPIEN-rendered** `base_camera` pixels at eval.
- Bridging this is the entire job of sim2real domain randomization; Ch 4 does none. A BC policy
  trained only on real frames has no reason to produce sensible actions on sim frames.

### 3.3 Embodiment: SO-100 (sim) vs SO-101 (dataset)
- Different hardware revision: joint ranges, gripper geometry, calibration. The `q01/q99`
  bins fit to SO-101 data do not correspond to SO-100's joint limits in sim.

### 3.4 Cameras: 1 (sim) vs 2 (dataset) — **eval-only**
- Backbone expects two views (392 image tokens). `PickCubeSO100-v1` gives one `base_camera`.
  The repo's `maniskill_obs_adapter` honestly **duplicates** the single view into both slots
  (`rollout.py`, documented) — a stopgap, not a fix. **Training is unaffected** (real dataset
  ships both cameras); this gap exists only in the sim rollout.

### 3.5 Task definition
- Sim `PickCube` success = cube within 2.5 cm of a **state-provided goal** and robot static.
  The dataset task is a teleoperated pick-and-**place** with no such goal channel. Even a
  perfect imitation of the demos is not scored against the same success predicate.

## 4. How the community actually evaluates this dataset

- **Open-loop + real-robot is the norm.** Public policies trained on `svla_so101_pickplace`
  (e.g. SmolVLA `jhou/smolvla_pickplace`, pi0 `leesangoh/pi0_pickplace`) report **validation
  loss on held-out SO-101 episodes** and **qualitative rollout videos**, or run **closed-loop
  on the real robot** via `lerobot-record` / `lerobot-eval`. None evaluate in ManiSkill.
- **`PickCubeSO100-v1` runs the opposite direction.** It originates in Stone Tao's
  [`lerobot-sim2real`](https://github.com/StoneT2000/lerobot-sim2real): **train an RL/visual
  policy *in* the PickCubeSO100 sim (with domain randomization), then deploy zero-shot to the
  real SO100.** Sim is the training domain and real is deployment — the reverse of Ch 4, where
  real is training and sim is eval. Using it as a BC eval target for real-data policies is
  off-label.

## 5. Why it matters for the book

- The manuscript states a specific number (§4.8: 60–80% closed-loop success). If a reader runs
  the shipped `rollout.py` on `PickCubeSO100-v1` with a policy trained on the real dataset, they
  will almost certainly see near-0% — and, worse, will conclude the **discrete-BC method** is
  broken, when the real cause is the train/eval domain gap. That undermines the chapter's core
  pedagogical claim (that discrete BC *works*, and its comparison against MSE/continuous heads).
- The book's own constraint (readers have **no hardware**) rules out the community's real-robot
  eval, which is exactly why a sim or open-loop path is attractive — but it has to be a *matched*
  one.

## 6. Options (for the three of us to choose)

1. **Open-loop primary + honest sim caveat.** Make held-out **validation loss** + the chapter's
   **bimodal / joint-coordination diagnostics** (Figs 4.8–4.10, already implemented) + **qualitative
   rollout videos on held-out dataset episodes** the primary eval. Keep the ManiSkill cell as
   clearly-labeled optional/aspirational with the domain-gap caveat. No closed-loop success
   number claimed. Smallest change; matches community practice; fully Colab-runnable.
2. **Matched sim (train + eval in the same domain).** Adopt the `lerobot-sim2real` setup: train
   the discrete head on **ManiSkill PickCubeSO100 demos** and eval in the **same** sim, so
   closed-loop success is meaningful. Self-consistent numbers, but abandons the real SO-101
   dataset that Ch 2/3/4 are built around — a large, cross-chapter change.
3. **Keep the sim eval, close what's closeable.** At minimum fix the **action-space/control-mode**
   mismatch (§3.1) and camera mapping, add SO-101 support to the sim if feasible, and **report
   whatever success rate actually results** (likely low, due to §3.2/§3.3). Highest effort; still
   fights the visual domain gap; risks publishing a discouraging number.

My lean: **Option 1** for Ch 4 as written, because it is honest, community-standard, and doesn't
require re-architecting Ch 2/3. Option 2 is the "right" long-term answer if the book wants real
closed-loop success numbers, but it's a book-wide decision.

## 7. Specific questions for the Ch 2 and Ch 3 owners

**For Sid (Ch 2):**
- Was `PickCubeSO100-v1` ever intended as a *closed-loop eval target for policies trained on the
  real `svla_so101_pickplace` dataset*, or only as (a) a sandbox to teach the sim/Gym API and (b)
  a home for the scripted-agent performance floor?
- What `control_mode` do you intend eval-rollout chapters to use? `make_env` defaults to
  `pd_joint_delta_pos` (deltas); the dataset actions are absolute targets. Should there be a
  documented "eval control mode" that matches the dataset's action space (e.g. `pd_joint_pos`),
  or a documented action transform?
- Is there any expectation that SO-100 (sim) and SO-101 (dataset) joint ranges are close enough
  to share a bin range, or is that an open gap?

**For Krishnam (Ch 3):**
- The ARCHITECTURE_LOG line "same 6-DOF action interface" — did that mean same *dimensionality*,
  or same *action space/units*? Ch 4's tokenizer depends on the latter for closed-loop eval.
- Ch 3 exports `from ch02.env import make_env  # optional, for eval rollouts`. Do you expect Ch 4
  to actually close the loop in sim, or is the backbone meant to be evaluated open-loop
  (validation loss + the frozen-encoder diagnostics)?

**For all three:**
- Do we standardize on **open-loop as the book's primary policy metric** (Option 1) and reserve
  closed-loop sim for a later, matched-domain chapter — or commit now to a matched sim (Option 2)?

## 8. References

**Code (this monorepo of repos):**
- `lrm-code-chapter-2/src/ch02/env.py:16-37` — `make_env`, default `control_mode="pd_joint_delta_pos"`.
- `lrm-code-chapter-2/src/ch02/__init__.py:6` — "re-exported for eval-rollout chapters."
- `lrm-code-chapter-2/README.md:5` — SO-100 sim → SO-101 hardware framing.
- `lrm-code-chapter-3/ARCHITECTURE_LOG.md:31,37,91` — anchor dataset, sim/dataset distinction, `make_env` eval note.
- `lrm-code-chapter-3/src/ch03/vla_backbone.py` — `NUM_CAMERAS = 2`, `IMAGE_TOKENS = 392` (two-view training).
- `lrm-code-chapter-4` manuscript `chapter_4_v3.md` §4.6.4 Listing 4.14, §4.8 (60–80% claim).
- `lrm-code-chapter-4/src/ch04/rollout.py` — `evaluate()` / `maniskill_obs_adapter` (single-camera duplication).
- `lrm-code-chapter-4/src/ch04/action_tokenizer.py` — absolute q01/q99 binning.

**External:**
- LeRobot dataset: https://huggingface.co/datasets/lerobot/svla_so101_pickplace (two cameras, 30 fps).
- Community policies (open-loop / real-robot eval): https://huggingface.co/jhou/smolvla_pickplace , https://huggingface.co/leesangoh/pi0_pickplace
- `PickCubeSO100` origin (sim→real, opposite direction): https://github.com/StoneT2000/lerobot-sim2real
- ManiSkill LeRobot/SO100 support: https://github.com/haosulab/ManiSkill/discussions/1079
- ManiSkill `PickCube` success predicate (cube within 2.5 cm of goal, robot static): https://maniskill.readthedocs.io/
