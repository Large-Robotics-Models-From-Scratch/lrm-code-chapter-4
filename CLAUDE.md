# Chapter 4 project guidance

The sole Chapter 4 design authority is:

`../lrm-book/chapter_4/manuscript/chapter_4.md`

Use the Chapter 2 and Chapter 3 repositories only for their live callable
APIs and tensor contracts. Do not infer Chapter 4 requirements from old
plans, earlier README text, or historical architecture notes.

Implementation invariants derived from the manuscript:

- SO-101: 6 controls, 16-step chunks, 256 bins.
- Chapter 2 z-score-normalized actions are the tokenizer domain.
- Action bins reuse SmolLM2 native ids 48896 through 49151.
- `ActionTokenizer` remains NumPy-only.
- The three public heads are `FactorizedActionHead`,
  `AutoregressiveActionHead`, and `ParallelDecodeActionHead`.
- The parallel head is the Listings 4.6 through 4.8 main path.
- Tests assert shapes, dtypes, grid order, padding, and vocabulary reuse.
