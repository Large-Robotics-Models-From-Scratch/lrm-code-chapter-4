# Chapter 4 project guidance

The sole Chapter 4 design authority is the v13 manuscript copy:

`https://docs.google.com/document/d/1oSaT2sR271OWgaZEJhyUx4DG6Knu0GbOV0vZJcqjEmo/edit`

The local Markdown manuscript can lag this document and is not an
architecture authority.

Use the Chapter 2 and Chapter 3 repositories only for their live callable
APIs and tensor contracts. Do not infer Chapter 4 requirements from old
plans, earlier README text, or historical architecture notes.

Implementation invariants derived from the manuscript:

- SO-101: 6 controls, 16-step chunks, 256 bins.
- Chapter 2 z-score-normalized actions are the tokenizer domain.
- Action bins reuse SmolLM2 native ids 48896 through 49151.
- `ActionTokenizer` remains NumPy-only.
- The three public heads are `OneshotActionHead`,
  `AutoregressiveActionHead`, and `ParallelDecodeActionHead`.
- `AutoregressiveActionHead` is the shipped head and the default CLI path.
- The AR head is the Listings 4.4 through 4.8 main path.
- `ParallelDecodeActionHead` is the Listing 4.3 comparison baseline. It
  uses 16 timestep slots and six categorical readouts per slot.
- Both append 16 timestep positions: SmolVLA action-suffix granularity, so
  head-to-head cost numbers are not confounded by suffix length. AR serial
  depth is `H` (16), not `H * D`; read it from `head.serial_steps`, never
  from `head.grid`, which counts target cells.
- The AR head conditions across time only. A timestep's six controls are
  decoded from one hidden state and are conditionally independent, so a
  same-timestep control pair does not separate AR from the parallel head.
- Tests assert shapes, dtypes, grid order, padding, and vocabulary reuse.
