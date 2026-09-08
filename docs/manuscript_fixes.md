# Manuscript fixes owed by the AR temporal-tokenization change

Branch `ch04-ar-temporal-tokens` changed `AutoregressiveActionHead` to append
one token per future timestep instead of one per `(timestep, control)` cell.
The code and the v11 manuscript now disagree in the places below.

Nothing here has been applied to the Google Docs. v11 is
`1UIVpB6hTNta-9RRYwaHuFSvmFnjy__NnYrs1zn_fQDw` (stable, do not edit); v12
review markup is `1S7N9270akNTzT-Ui2Pe1R1ZxGLfyvIzHujsbs7VBVjk`, using the
green-addition / red-strikethrough-removal / yellow-edit convention.

## 1. Section 4.4 -- the autoregressive factorization

The displayed factorization runs the product over both axes with each cell
conditioned on `a_<(t,d)`. It should condition on earlier *timesteps*: the
product over `d` within a timestep is now independent given `s` and `a_<t`.

"With H = 16 and D = 6, the autoregressive (AR) head makes 96 dependent
predictions at inference" -- now **16** dependent predictions, each emitting all
six controls.

## 2. Section 4.4.3 -- prose that is now false

> "It flattens the H x D grid in time-major order [...] and predicts those
> scalar bin IDs sequentially. When it predicts the gripper, for example, it can
> condition on the wrist bin already chosen for that timestep."

The second sentence must go. The head no longer conditions on anything chosen
within the same timestep. Replacement framing: the head appends one token per
timestep whose embedding is the sum of that timestep's six control-bin
embeddings, and reads all six control distributions out of one hidden state.

> "At inference each bin is chosen, fed back in, and only then is the next one
> predicted: 96 dependent steps."

Now 16 dependent steps, each feeding back a whole control vector.

Also worth adding here: the AR head and the bidirectional head now append
*identical* suffixes, which is what makes the section 4.6 comparison clean.
That is the argument that justifies the change.

## 3. Listings 4.4 and 4.5

Both listings must be regenerated from
`src/ch04/autoregressive_action_head.py`. The substantive edits:

- `self.grid = horizon * action_dim` is no longer the sequence length. It
  survives only as a target-cell count; serial depth is `self.horizon`, exposed
  as the `serial_steps` property.
- `nn.Embedding(n_bins, d_embed)` becomes `nn.Embedding(action_dim * n_bins,
  d_embed)`, with a registered `control_offsets` buffer of `arange(D) * n_bins`.
- `nn.Linear(d_embed, n_bins)` becomes `nn.Linear(d_embed, action_dim *
  n_bins)`, unflattened to `[..., D, n_bins]`.
- Teacher forcing embeds `target_bins[:, :-1]` as a `[B, H-1, D]` grid summed
  over the control axis, and extends positions by `horizon - 1`, not `grid - 1`.
- `generate` loops `range(self.horizon)` and selects `[B, D]` bins per step.
- Annotation #A ("Record the H x D scalar sequence length used by the
  autoregressive branch") is wrong twice over and must be rewritten.
- Annotation #F ("Restore the generated scalar sequence to the [B, H, D] action
  grid") no longer applies: `generate` stacks `[B, D]` steps directly into
  `[B, H, D]` with no reshape.

## 4. Figure 4.9 -- must be re-anchored

This is the largest editorial consequence.

The figure caption reads: sampled `(gripper, wrist)` pairs on a state whose
demonstrations contain only (close, lift) and (open, lower), where "the
autoregressive head stays on the demonstrated diagonal" and "the parallel
decoding head sits between."

Gripper and wrist are two controls **at the same timestep**. After this change
the AR head decodes them from one hidden state, so it is exactly as independent
across that pair as the parallel head. The panel can no longer show the claimed
separation, and re-running it would produce two indistinguishable panels.

The chapter already names the replacement, in the sentence immediately after the
figure: "The same comparison run along the temporal axis, plotting one
dimension's sampled bins across consecutive timesteps." That comparison still
holds and still separates the heads, because AR conditions each timestep on the
timesteps it already sampled. Figure 4.9 should be rebuilt on it -- one control,
consecutive timesteps -- and the trailing sentence dropped or inverted.

`analysis.joint_mismatch_samples` still runs and its docstring now says this
explicitly, so the figure code is not broken, only the axis choice.

The replacement figure needs no new code: `diagnostics.plot_temporal_traces`
already renders sampled bins for one control across a chunk, one panel per
head, annotated with mean step-to-step change. It was written for exactly the
sentence that follows the figure and is unaffected by this change.

The section 4.4.1 "Where independence breaks: joint mismatch" discussion should
survive but narrow its claim: the one-shot head's failure spans both axes; the
AR head repairs the temporal axis only. Chapter 5's flow matching is still the
thing that models the chunk jointly.

## 5. Section 4.6 -- measured numbers

Re-measured on the real Chapter 3 backbone (SmolLM2-135M, 576 wide, 405-token
observation prefix, batch 1, Apple MPS), median of 7 timed repeats:

| head | latency (ms) | GFLOPs | serial steps |
| --- | --- | --- | --- |
| one-shot | 119 | 165.695 | 1 |
| bidirectional parallel | 124 | 170.021 | 1 |
| autoregressive | 370 | 169.324 | 16 |

AR now costs **3.0x** the bidirectional head's GPU latency (was 17x with 96
tokens) and **0.996x** its FLOPs.

The FLOPs point in the chapter gets *stronger*, not weaker. Previously FLOPs
compressed a 17x latency gap to 1.11x. Now the arithmetic actually **inverts the
ranking**, scoring the 3x-slower head as marginally the cheaper of the two.
Cached decoding recomputes no projections and skips the masked half of the
causal attention matrix, and the shared 405-token prefill dominates both totals.
Latency remains the correct axis for the section 4.6 figure.

Any prose describing AR's latency penalty as overwhelming should be softened:
3x is a real cost but not the 17x the 96-token design implied.

## 6. Pre-existing, unrelated to this change

Listing 4.3 (`ParallelDecodeActionHead`) prints
`self.slots = nn.Parameter(0.02 * torch.randn(self.grid, d_embed))` with
`self.grid = horizon * action_dim` and `self.readout = nn.Linear(d_embed,
n_bins)` -- 96 slots with a single-control readout. Annotation #H says "Return
logits for all 96 cells."

This contradicts the listing's own preceding paragraph ("adopts SmolVLA's action
suffix granularity - H sequence positions, one per future timestep [...] i.e.
D=1") and contradicts the shipped code, which has always used 16 slots and
`Linear(d_embed, action_dim * n_bins)`. The listing needs regenerating from
`src/ch04/parallel_action_head.py` regardless of this branch.

## Not addressed here

The AR head's trained checkpoints and every held-out metric derived from them
are invalidated by the architecture change; retraining needs the Colab A100
recipe and is a separate task.
