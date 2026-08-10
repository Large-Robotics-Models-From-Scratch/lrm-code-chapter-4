# Chapter 4: Discrete Behavior Cloning

Working code for Chapter 4 of *Build a Large Robot Model (From
Scratch)*. The implementation follows the current manuscript in
`lrm-book/chapter_4/manuscript/chapter_4.md`; older Chapter 4 plans are
not design authorities.

The repository turns Chapter 3's fused VLA representation into an
SO-101 action policy trained from Chapter 2's demonstrations. It includes
the manuscript's three action-head designs:

- `FactorizedActionHead`: a one-shot product-of-marginals baseline.
- `AutoregressiveActionHead`: exact left-to-right conditioning with a
  teacher-forced training path and a clear uncached reference decoder.
- `ParallelDecodeActionHead`: the manuscript's one-pass training and
  evaluation path, with bidirectional attention inside the action grid.

## Repository map

```text
src/ch04/
├── action_tokenizer.py          # Q01/Q99 uniform bins + LM vocab reuse
├── factorized_action_head.py    # Listing 4.2
├── autoregressive_action_head.py# Listing 4.3
├── parallel_action_head.py      # Listing 4.4
├── backbone_adapter.py          # pre-transformer Chapter 3 splice
├── data.py                      # episode splits, chunks + prepare_batch
├── losses.py                    # grid shaping + masked smoothed CE
├── train.py                     # all-head training + compact checkpoints
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
head, constructs episode-disjoint LeRobot action chunks, trains all three
heads for configurable step counts, and decodes a held-out chunk back to
the dataset's raw action units.

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
- Bins reuse native SmolLM2 ids `48896..49151`; no language-model
  vocabulary rows are added.
- One label is `[H, D] = [16, 6]`, flattened timestep-major to 96 action
  tokens. Padding is repeated across the six controls and excluded from
  the loss before averaging.
- Chapter 3 receives images `[B, 2, 3, 224, 224]`, sequence ids, and
  normalized state `[B, 6]`, and uses hidden width 576.
- Prepared instructions have a fixed 64-token budget. Padding is masked and
  assigned compact position ids, keeping action-token positions independent
  of the other instruction lengths in a batch.
- Tokenizer decoding returns normalized actions. `denormalize_from_stats`
  converts those back to raw dataset units for open-loop comparison.

The open-loop notebook is not a physical-robot deployment recipe. The
dataset's command units and any simulator or hardware control mode must
be converted and safety-checked explicitly before execution.
