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
├── execution.py                 # chunk schedule + temporal ensemble
├── diagnostics.py               # section 4.6 plots and mismatch metric
└── exercises.py                 # multimodal MSE-collapse exercise
notebooks/ch04.ipynb             # complete Colab walkthrough
```

## Local setup

Use Python 3.12 or 3.13 and install the sibling chapter packages first:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e "../lrm-code-chapter-2[data]"
pip install -e ../lrm-code-chapter-3
pip install -e ".[dev,data]"
```

Run the fast suite:

```bash
pytest -m "not integration"
ruff check src tests
```

The [Chapter 4 Colab](notebooks/ch04.ipynb) installs all three chapter
repositories, demonstrates MSE collapse, fits the tokenizer, and constructs
episode-disjoint LeRobot action chunks. A shared experiment runner then
trains, visualizes, samples, and evaluates the factorized, autoregressive,
and parallel heads in order. Each design starts from a fresh Chapter 3
backbone, and decoded held-out chunks are compared in the dataset's raw
action units.

The Colab setup installs the public Chapter 2, 3, and 4 packages directly
from GitHub. It prints pip's full diagnostic if installation fails. LeRobot
can cause pip to replace Colab's preinstalled Torch while leaving an
ABI-incompatible optional TorchAudio wheel behind; the setup probes that
wheel in a child process and removes it only when it is broken. None of the
three chapter pipelines use audio.

## Data and model contracts

- Actions are z-score normalized with Chapter 2 statistics before the
  tokenizer is fit or called.
- The tokenizer clips each dimension to q01/q99, divides it into 256
  bins, and returns NumPy `int64` bin ids.
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
  bidirectional action mask.
- Training keeps SigLIP frozen while updating the Chapter 3 projection,
  state encoder, language backbone, and action head at separate learning
  rates. Checkpoints include the full policy, normalization statistics,
  tokenizer bounds, optimizer/scheduler state, and held-out loss.
- Tokenizer decoding returns normalized actions. `denormalize_from_stats`
  converts those back to raw dataset units for open-loop comparison.

The open-loop notebook is not a physical-robot deployment recipe. The
dataset's command units and any simulator or hardware control mode must
be converted and safety-checked explicitly before execution.
