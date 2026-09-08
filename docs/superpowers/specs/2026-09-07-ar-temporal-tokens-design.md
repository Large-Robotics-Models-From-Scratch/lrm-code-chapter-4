# Autoregressive head: SmolVLA-style temporal tokenization

Date: 2026-09-07
Branch: `ch04-ar-temporal-tokens`
Status: approved design, ready for implementation planning

## Problem

Chapter 4 ships three discrete action heads over the same `H x D` target grid
(`H = 16` timesteps, `D = 6` controls, `K = 256` bins). Two of them tokenize the
grid at different granularities:

- `ParallelDecodeActionHead` appends **16** learned slots, one per future
  timestep, and reads the whole control vector out of each slot with
  `Linear(d_embed, D * K)`. This is SmolVLA's action-suffix granularity, and the
  manuscript's section 4.4.2 prose says so explicitly.
- `AutoregressiveActionHead` appends **96** tokens, one per `(timestep, control)`
  cell, and reads a single control out of each with `Linear(d_embed, K)`.

Now that the autoregressive (AR) head is the shipped path, the mismatch is a
problem. Head-to-head numbers in section 4.6 are not apples-to-apples: the two
heads differ in attention pattern *and* in sequence length, so any latency,
FLOP, or quality gap confounds the two. This change makes the AR head use the
same temporal granularity as the parallel head, isolating the causal-versus-
bidirectional comparison as the only difference between them.

## Decisions taken

Three choices were settled before design, and each closes off alternatives that
would otherwise be reasonable.

**Pure SmolVLA granularity, accepting the modeling loss.** At AR step `t` the
head emits all six controls from one hidden state, so the six controls of a
single timestep become conditionally independent given the observation and all
earlier timesteps. Rejected alternatives: a chained intra-timestep readout that
would preserve same-timestep coupling at the cost of an extra component to
explain; a `token_granularity` flag keeping both paths alive; and inverting the
change by moving the parallel head to 96 slots.

**Sum of per-control embeddings.** One `nn.Embedding(D * K, d_embed)` table with
a per-control offset of `arange(D) * K`; the six lookups for a timestep are
summed into that timestep's input embedding. Control identity is carried by the
table itself, so no positional trick is needed and nothing constrains `d_embed`
to be divisible by `D`. Rejected: concatenating six `d_embed // D` embeddings
(hard-requires divisibility, gives each control only 96 dimensions) and
embedding-then-projecting through `Linear(D * d_embed, d_embed)` (~2M extra
parameters for a mixing step the backbone can already perform).

**Repo-wide consequences in scope, retraining out of scope.** Everything in the
repository that assumed 96 serial steps is corrected here, including
re-measuring the section 4.6 cost numbers. Retraining the AR head so its
reported held-out metrics match the new architecture needs the Colab A100
recipe, which is a user-driven run, and is deliberately left for a follow-up.

## Consequence: what this costs the chapter

This is the part a reader of the diff must not miss.

Section 4.4.3 currently claims the AR head can "condition on the wrist bin
already chosen for that timestep." After this change that sentence is false.
Figure 4.9's headline result -- sampled `(gripper, wrist)` pairs at a single
timestep, where the AR head "stays on the demonstrated diagonal" while the
parallel head "sits between" -- rests on exactly the same-timestep coupling this
change removes. With temporal tokens, AR and parallel are *identically*
independent within a timestep, and that panel stops distinguishing them.

What survives is the temporal axis. The AR head still conditions each timestep
on the bins actually realized at every earlier timestep, which the parallel head
never does. The sentence immediately after Figure 4.9 already describes that
comparison ("one dimension's sampled bins across consecutive timesteps"), so the
figure must be re-anchored on it rather than on the control axis.

This is a real narrative cost, accepted deliberately in exchange for a
comparison between heads that is not confounded by sequence length.

## Design

### The head

`AutoregressiveActionHead` keeps its public contract unchanged. Every consumer
dispatches on duck-typing -- `train.action_head_logits` looks for
`teacher_forced_logits`, `decoding.action_head_bins` looks for `generate` -- so
no call site changes.

| | before | after |
| --- | --- | --- |
| appended sequence positions | 96 | **16** |
| input embedding | `Embedding(K, d_embed)`, one bin per position | `Embedding(D * K, d_embed)`, six offset lookups summed per position |
| readout | `Linear(d_embed, K)` | `Linear(d_embed, D * K)`, viewed as `[..., D, K]` |
| serial decode steps | 96 | **16** |
| `teacher_forced_logits` output | `[B, H, D, K]` | `[B, H, D, K]` (unchanged) |
| `generate` output | `[B, H, D]` | `[B, H, D]` (unchanged) |

**Teacher forcing.** Inputs are the timestep embeddings for `t = 0 .. H-2`. The
last prefix hidden state predicts `t = 0`, and each supplied timestep embedding
predicts the next, so the slice `hidden[:, prefix_len - 1:]` yields exactly `H`
positions. Position IDs come from `extend_position_ids(prefix_positions,
horizon - 1)`. The 2-D boolean attention mask and the backbone's default causal
behavior are unchanged.

**Generation.** `generate` runs `H` cached steps. Each step reads out
`[B, D, K]` and selects six bins through the existing `decoding.sample_logits`,
which already handles arbitrary leading axes -- replacing the current inline
argmax/multinomial branch. The selected bins are embedded and fed back as the
next timestep token.

**Target handling.** `_flatten_targets` becomes `_grid_targets` and returns
`[B, H, D]`, still accepting a flattened `[B, H*D]` input for compatibility with
existing callers. `self.grid` is retained as the target-cell count used for
validation; the appended sequence length is now `self.horizon`.

The result is structurally the mirror of `ParallelDecodeActionHead`: same 16
positions, same `D * K` readout, differing only in causal-versus-bidirectional
attention and in whether the appended positions carry learned slots or realized
action embeddings. That symmetry is the point of the change.

### Downstream corrections

- `diagnostics.py` -- `plot_quality_compute_tradeoff` default `decode_steps`
  entry `"autoregressive": 96` becomes `16`.
- `analysis.py` -- the docstring quoting **1.11x FLOPs / 17x GPU latency** is
  measured against the 96-token head and is now wrong. Re-measure on the real
  Chapter 3 backbone and rewrite with the measured values. Docstrings at the
  joint-logit and generated-pair helpers that describe cell-level causal
  conditioning must be narrowed to timestep-level.
- `train.py` -- "an autoregressive rollout costs `H * D` serial steps per batch"
  becomes `H`; the teacher-forcing docstrings describing per-cell expert
  prefixes become per-timestep.
- `decoding.py` -- the `sample_action_grids` docstring claiming later draws
  condition on earlier bins "in the same grid" must say earlier *timesteps*.
- `README.md`, `CLAUDE.md`, `notebooks/ch04.ipynb` -- any statement of 96 serial
  steps or per-cell conditioning.
- `docs/manuscript_fixes.md` -- **create** this file (it exists on the older
  `ch4-v2-build` branch but not on `main`) and record the manuscript
  consequences there rather than editing the Google Doc: section 4.4.3 prose, Listings 4.4 and 4.5, every "96
  dependent steps" claim, the Figure 4.9 re-anchoring described above, and the
  pre-existing Listing 4.3 bug in which the printed code allocates
  `horizon * action_dim` slots with `Linear(d_embed, n_bins)`, contradicting
  both its own prose and the shipped `ParallelDecodeActionHead`.

### Measurement

Re-run `measure_inference_latency` and `measure_inference_flops` against the
real Chapter 3 `VLABackbone` (SmolLM2-135M, 576 wide, 399-token prefix) on MPS
with `HF_HUB_OFFLINE=1`, batch 1, synchronizing on MPS as well as CUDA. Record
one-shot, bidirectional, and the new temporal AR head. Expect the AR serial
depth to fall from 96 to 16 and its latency penalty to shrink by roughly that
factor; the point of measuring is that "roughly" is not good enough for a number
printed in the book.

### Testing

Test-driven: the behavioral tests below are written before the head is changed.

New tests in `tests/test_attention_semantics.py` and `tests/test_heads.py`:

1. The AR head appends **16** positions to the prefix, not 96.
2. `teacher_forced_logits` still returns `[B, H, D, K]`; `generate` still returns
   `[B, H, D]` with every bin in `[0, K)`.
3. **Temporal causality holds.** Logits at timestep `t` are invariant to changes
   in target bins at timesteps `> t`, and *do* change when bins at timesteps
   `< t` change.
4. **Intra-timestep independence is pinned.** Logits for control `d` at timestep
   `t` are invariant to the other controls' target bins at that same timestep.
   This test documents the accepted modeling loss so a future reader cannot
   mistake it for a bug.
5. Greedy `generate` agrees with the teacher-forced argmax when the teacher's
   bins are the model's own greedy choices.

Existing suites that must stay green: `test_heads`, `test_attention_semantics`,
`test_integration_pipeline`, `test_data_losses_decoding`, `test_analysis_figures`,
`test_notebook`, `test_train_execution`.

## Explicitly out of scope

- Retraining the AR head and refreshing its held-out metrics.
- Editing the v11 or v12 Google Docs. The manuscript consequences are recorded
  in `docs/manuscript_fixes.md` only.
- Any change to `ParallelDecodeActionHead` or `OneshotActionHead`.
- The section 4.6 evaluation-axis work in progress on
  `ch04-eval-axis-and-rollout-curves`, which touches `diagnostics.py` and the
  notebook. That branch and this one will conflict in
  `plot_quality_compute_tradeoff`; whichever merges second resolves it.
