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

Use Python 3.12 and install the sibling chapter packages first:

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
repositories, demonstrates MSE collapse, fits the tokenizer, runs every
head, constructs episode-disjoint LeRobot action chunks, trains the shipped
parallel policy for a configurable step count, and evaluates decoded held-out
chunks in the dataset's raw action units.

If a chapter repository is private, create a fine-grained GitHub personal
access token with read access to that repository, add it under **Colab >
Secrets** as `GITHUB_TOKEN`, and enable notebook access for the secret. The
setup cell sends it through Git's process environment, never places it in
the clone URL, and does not print it. For organization-owned repositories,
authorize the token for SSO if the organization requires it.

If a clone is interrupted and leaves an incomplete checkout, use **Runtime
> Disconnect and delete runtime**, reconnect, and rerun the setup cell. The
cell retries shallow clones three times and prints Git's stderr.

## Data and model contracts

- Actions are z-score normalized with Chapter 2 statistics before the
  tokenizer is fit or called.
- The tokenizer clips each dimension to q01/q99, divides it into 256
  bins, and returns NumPy `int64` bin ids.
- SmolLM2 remains at its native 49,152-row vocabulary. The optional AR
  head uses a separate 256-entry action embedding table indexed by bin id.
- One label is `[H, D] = [16, 6]`, flattened timestep-major to 96 action
  tokens. Padding is repeated across the six controls and excluded from
  the loss before averaging.
- Chapter 3 receives raw `[0,1]` images `[B,2,3,H,W]`, padded native text
  ids, a text attention mask, and normalized state `[B,6]`. It owns image
  resizing, direct multimodal concatenation, and compact position ids.
- The parallel head extends Chapter 3's observation prefix with one learned
  576-wide slot per action-grid cell and contextualizes the complete block
  with a custom bidirectional action mask.
- Training keeps SigLIP frozen while updating the Chapter 3 projection,
  state encoder, language backbone, and action head at separate learning
  rates. Checkpoints include the full policy, normalization statistics,
  tokenizer bounds, optimizer/scheduler state, and held-out loss.
- Tokenizer decoding returns normalized actions. `denormalize_from_stats`
  converts those back to raw dataset units for open-loop comparison.

The open-loop notebook is not a physical-robot deployment recipe. The
dataset's command units and any simulator or hardware control mode must
be converted and safety-checked explicitly before execution.
