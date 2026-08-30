# Chapter 4: Discrete Behavior Cloning

Working code for Chapter 4 of *Build a Large Robot Model (From
Scratch)*. The implementation follows the current
[Chapter 4 manuscript](https://docs.google.com/document/d/1UIVpB6hTNta-9RRYwaHuFSvmFnjy__NnYrs1zn_fQDw/edit)
and the live Chapter 3 hand-off contract; older local Chapter 4 drafts are
not design authorities.

The repository turns Chapter 3's fused VLA representation into an
SO-101 action policy trained from Chapter 2's demonstrations. It includes
the manuscript's three action-head designs:

- `FactorizedActionHead`: a one-shot product-of-marginals baseline.
- `AutoregressiveActionHead`: exact left-to-right conditioning with a
  teacher-forced training path, separate action embeddings, and KV-cached
  generation.
- `ParallelDecodeActionHead`: the manuscript's one-pass training and
  evaluation path, with bidirectional attention inside the action grid.

## Repository map

```text
src/ch04/
├── action_tokenizer.py          # Q01/Q99 uniform per-control bins
├── factorized_action_head.py    # Listing 4.2
├── autoregressive_action_head.py# Listings 4.3–4.4
├── parallel_action_head.py      # Listing 4.5
├── backbone_adapter.py          # helpers for Ch3's two-stage API
├── data.py                      # episode splits, chunks + prepare_batch
├── losses.py                    # grid shaping + masked smoothed CE
├── train.py                     # two-LR training + held-out checkpoints
├── decoding.py                  # temperature/top-p + open-loop MAE
├── execution.py                 # the three section 4.7.2 schedules
├── so101.py                     # export + safety-gated chunk replay
├── diagnostics.py               # figure helpers (pure, array in / fig out)
├── analysis.py                  # drivers that run a head to make those arrays
├── exercises.py                 # MSE collapse + the section 4.2.2 mixture
├── cli.py                       # `ch04-train`
└── figures.py                   # `ch04-figures`
notebooks/ch04.ipynb             # complete Colab walkthrough
```

## Local setup

Use Python 3.12 or 3.13 and install the sibling chapter packages first:

```bash
python3.12 -m venv .venv
```

```bash
pip install -e "../lrm-code-chapter-2[data]" && pip install -e ../lrm-code-chapter-3 && pip install -e ".[dev,data]"
```

Run the fast suite:

```bash
pytest -m "not integration"
```

The opt-in suite exercises the live Chapter 2 and Chapter 3 hand-offs and
trains every head for two real steps:

```bash
pytest -m integration
```

## Training the action heads

`ch04-train` fits any head, or all three in sequence, each on its own
freshly initialized Chapter 3 backbone so the comparison is fair:

```bash
ch04-train --head all --steps 20000 --batch-size 32
```

```bash
ch04-train --head parallel --steps 20000 --checkpoint-dir checkpoints
```

For a full run with live loss curves:

```bash
ch04-train --head all --steps 20000 \
  --checkpoint-dir checkpoints \
  --tensorboard-dir runs/ch04
```

```bash
tensorboard --logdir runs/ch04
```

Each run writes `checkpoints/<head>/summary.json` with the configuration,
wall clock, held-out token cross-entropy, loss history, and open-loop MAE,
plus `latest.pt`, `best.pt`, and a combined
`checkpoints/training_curves.png`. TensorBoard records `loss/train`,
`loss/held_out`, overall and per-control token accuracy, overall and
per-control MAE, `entropy/train`, and the head/backbone learning rates.
A checkpoint carries the whole float32
backbone and runs to roughly 1.9 GB, so permanent snapshots are opt-in via
`--snapshot-steps 5000 20000`. `--resume-from` restores the model,
optimizer, scheduler, and step count, and refuses a checkpoint whose
configuration or tokenizer bounds differ.

The reporting follows the useful part of OpenVLA's training loop:
categorical loss is paired with exact action-token accuracy and decoded
continuous L1 error, rather than treated as sufficient on its own. See
OpenVLA's [official metric implementation](https://github.com/openvla/openvla/blob/main/prismatic/training/strategies/base_strategy.py).
Chapter 4 reports these by control dimension as small multiples so six
joint curves never obscure one another.

`accuracy` means exact argmax-bin agreement over non-padded action cells.
`mae_in_std` decodes the predicted bin centre in z-score-normalized action
space and averages absolute error; 1.0 is one training-set standard
deviation. `mae_raw_by_control` denormalizes both prediction and target
before measuring each joint in the dataset's native command units. The
headline open-loop MAE is the per-cell raw error divided by that control's
training standard deviation and then averaged. These are imitation-fit
metrics, not physical task-success rates. The train/held-out metrics read
the logits used by the loss, so later autoregressive cells are
teacher-forced. The separately reported open-loop MAE calls each head's
actual inference path and does not provide future expert bins.

## Regenerating the figures

`ch04-figures` reloads a checkpoint — including *its* tokenizer bounds and
normalization statistics rather than refitting them — and writes every
figure the chapter attributes to code:

```bash
ch04-figures checkpoints/parallel/best.pt --head parallel --output-dir figures
```

| Figure | Helper | What it shows |
| --- | --- | --- |
| 4.4 | `diagnostics.plot_bimodal_comparison` | MSE collapse against a two-component mixture |
| 4.8 | `analysis.neighborhood_softmax_figure` | listing 4.9's held-out softmax cluster |
| 4.9 | `diagnostics.plot_joint_logit_panels` | expert and direct logit-implied pair mass, separate panels |
| §4.6.2 | `diagnostics.plot_temporal_traces` | sampled bins across a chunk, per head |
| 4.10 | `diagnostics.plot_execution_schedules` | the three section 4.7.2 schedules |
| 4.11 | `diagnostics.plot_open_loop_episode` | expert against decoded commands |
| — | `diagnostics.plot_training_curves` | train/held-out CE and token accuracy |
| — | `diagnostics.plot_per_joint_metrics` | per-control train/held-out accuracy and MAE |
| — | `diagnostics.plot_head_comparison` | final metric per head |

All of them share one look. `ch04.style.use_manuscript_style()` sets the
matplotlib defaults, and `ch04.style.HEAD_STYLES` reserves a colour, dash
pattern, and marker per action head so a reader can carry one head's
identity across every figure — and so the three stay separable in
grayscale print. The Colab calls it once in its setup cell; `ch04-figures`
calls it for you.

Section 4.6.1 requires that a reported softmax figure name its
checkpoint, anchor index, neighbour count, and seed. The caption is
generated from those arguments, so it cannot drift from the run.

## The Colab

The [Chapter 4 Colab](notebooks/ch04.ipynb) installs all three chapter
repositories, demonstrates MSE collapse against a fitted mixture, fits the
tokenizer, and constructs episode-disjoint LeRobot action chunks. A shared
experiment runner then trains, visualizes, samples, and evaluates the
factorized, autoregressive, and parallel heads in order, each from its own
fresh Chapter 3 backbone, before producing the section 4.6 and 4.7 figures.

Setup installs the public Chapter 2, 3, and 4 packages directly from
GitHub and prints pip's full diagnostic on failure. Two Colab-specific
hazards are handled there:

- LeRobot can make pip replace Colab's preinstalled Torch while leaving an
  ABI-incompatible optional TorchAudio wheel behind. Transformers detects
  TorchAudio by package presence and imports that binary on the way to
  SigLIP, so `from ch03 import VLABackbone` fails with an undefined-symbol
  `OSError`. The setup probes the wheel in a child process and removes it
  only when it is broken; none of the three chapter pipelines use audio.
- `lrm-ch04` stays at version 0.1.0 while the branch is under development,
  so a reused runtime can keep a stale wheel with the same version number.
Setup force-reinstalls just that package, verifies the API it needs
imports, and drops cached chapter modules.

### Full-scale Colab run

1. Open [the Chapter 4 notebook](notebooks/ch04.ipynb) in Colab and choose
   **Runtime > Change runtime type > GPU**. A fresh runtime is recommended
   when changing package revisions.
2. Run the setup and data cells once. In **Choose a run mode**, set
   `RUN_MODE = 'full'`. The two supported modes are intentionally fixed:
   `sanity` runs 10 steps per head and prints every loss; `full` runs
   exactly 20,000 steps per head, logs every 25 steps, validates and
   refreshes restartable checkpoints every 1,000 steps, and gives every
   head the same budget.
3. Run sections 4.5.1 through 4.5.3 in order. Do not reuse one head's
   backbone for another. The **Training curves, side by side** cell saves
   `/content/ch04-checkpoints/training_curves.png` and embeds TensorBoard
   from `/content/ch04-checkpoints/tensorboard`.
4. Keep the browser tab connected. Colab storage is ephemeral: download
   `best.pt`, `latest.pt`, the TensorBoard directory, and the static curve
   before resetting the runtime. Each full-policy checkpoint is roughly
   1.9 GB because it includes the float32 backbone.
5. Only interpret the comparison figures from `full` mode. Sanity-mode
   figures verify shapes and code paths, not convergence.

The joint-mismatch figure reads categorical probability mass directly
from the logits over held-out frames. It does not estimate a density from
sample counts. The expert distribution has its own panel, which avoids
covering either distribution with overlaid points. For the
autoregressive head, the later cell's logits are teacher-forced on the
demonstrated preceding bins; this is a conditional diagnostic rather than
an exhaustive enumeration of all 256-valued prefixes.

## SO-101 chunk playback and episode replay

The Colab's **Export one chunk for a calibrated SO-101** cell writes
`so101_chunk.npz`. Download it to the computer physically connected to a
calibrated follower. Previewing is the default and does not connect:

```bash
ch04-so101-replay so101_chunk.npz \
  --port /dev/tty.usbmodemXXXX
```

After checking the six named commands, clearing the workspace, and making
an emergency stop available, opt into one 16-step playback:

```bash
ch04-so101-replay so101_chunk.npz \
  --port /dev/tty.usbmodemXXXX \
  --robot-id my_follower \
  --max-relative-target 5 \
  --execute
```

The file contains raw dataset command units in the SO-101 feature order,
plus the training-data min/max. The command refuses malformed or
out-of-range files, asks for an `EXECUTE` confirmation, uses LeRobot's
relative-target cap, sends at the dataset's 30 Hz rate, and disconnects in
a `finally` block. Re-check the workspace and run the same command again
to replay the chunk.

To replay a known demonstration episode instead of a policy chunk, use
LeRobot's official dataset replay path:

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=my_follower \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --dataset.episode=0
```

See LeRobot's [real-robot imitation-learning guide](https://huggingface.co/docs/lerobot/en/il_robots)
and [SO-101 setup/calibration guide](https://huggingface.co/docs/lerobot/en/so101).
This repository pins LeRobot 0.5.1, where calibrated follower control is
exposed through `SO101Follower.send_action` and dataset episodes through
`lerobot-replay`.

## Data and model contracts

- Actions are z-score normalized with Chapter 2 statistics before the
  tokenizer is fit or called. `data.as_tensor_stats` coerces both of
  Chapter 2's statistics formats (LeRobot's NumPy `meta/stats.json` and
  Chapter 2's own torch computation) into one `float32` structure.
- The tokenizer clips each dimension to q01/q99, divides it into 256
  bins, and returns NumPy `int64` bin ids. It is NumPy-only.
- SmolLM2 remains at its native 49,152-row vocabulary. The optional AR
  head uses a separate 256-entry action embedding table indexed by bin id.
- One label is `[H, D] = [16, 6]`. The shipped parallel head keeps this
  shape and returns logits `[B, H, D, bins]`; only the optional AR branch
  flattens the grid into 96 scalar action tokens. Padding is expanded over
  the six controls and excluded from the loss before averaging.
- Chapter 3 receives raw `[0,1]` images `[B,2,3,H,W]`, padded native text
  ids, a text attention mask, and normalized state `[B,6]`. It owns image
  resizing, direct multimodal concatenation, and compact position ids.
- The six-value proprioceptive vector becomes one observation token. The
  parallel head follows SmolVLA's vector layout: one action position per
  future timestep, with a readout covering all controls. Chapter 4 makes
  that readout categorical (`D x bins`) instead of reproducing SmolVLA's
  continuous flow-matching objective.
- The parallel head extends Chapter 3's observation prefix with `H` learned
  576-wide action positions and contextualizes the suffix with a custom
  bidirectional action mask. Constructing it switches the shared backbone
  to eager attention, which is the only implementation guaranteed to accept
  a 4-D additive mask across the supported `transformers` range.
- Training keeps SigLIP frozen while updating the Chapter 3 projection,
  state encoder, language backbone, and action head at separate learning
  rates. SmolLM2 loads in `bfloat16`, whose eight-bit mantissa rounds a
  1e-5-scale AdamW update to zero, so `train_action_head` promotes the
  trainable parameters to `float32` master weights first
  (`upcast_backbone=False` opts out). Without it roughly 2% of trunk
  elements move per step instead of all of them; with it a checkpoint is
  about 1.9 GB rather than 1 GB.
- Checkpoints include the full policy, normalization statistics,
  tokenizer bounds, optimizer/scheduler state, and held-out loss.
- Tokenizer decoding returns normalized actions. `denormalize_from_stats`
  converts those back to raw dataset units for open-loop comparison.

The exported-chunk command is a bounded hardware smoke test, not a
physical-robot deployment recipe. It executes one already-decoded chunk
without refreshing camera/state observations. Section 4.7.4's
closed-loop rollout remains a manuscript `<TODO>`; the `sim` extra is
reserved for it, and task success still requires repeated closed-loop
episodes with a stated success criterion.
