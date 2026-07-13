<!-- Synced copy of ../lrm-book/chapter_4/chapter_4_structure_and_plan.md.
     Corrected to the Ch 3 v5 contract (UnifiedEmbeddingBackbone, 576 hidden,
     two cameras, 256 shared reserved ids 48896-49151). Keep both in sync. -->

# Chapter 4: Discrete Behavior Cloning — Structure & Content Plan

## Archetype

**Primary:** Coding-heavy (Raschka *Build a Large Language Model from Scratch* model, per `book_learnings.md` §2).
**Secondary inflections:** Math-heavy patterns from Lambert (`book_learnings.md` §3) for §4.3 (categorical distribution); systems patterns from Wang/Szeto (`book_learnings.md` §4) for §4.2's tokenizer triptych.

This chapter is the book's first major implementation chapter. The reader writes real PyTorch code, runs it, inspects expected outputs, and watches a policy converge. We follow the "let's run it" micro-loop (describe → listing → run → output → interpret → bridge) 4–8 times per section.

## Source documents

* Research dossier — `chapter_4/resources/claude_research_summary.md`
* Supplementary synthesis — `chapter_4/resources/gemini_research_synthesis.md` (cross-embodiment material + adjacent-bin label smoothing)
* Body-chapter craft — `book_learnings.md`
* Proposal section outline — `proposal/proposal.md` §4.1–4.6 (we expand to 8 sections; deviation justified inline)
* Style — `writing_instructions/writing_instructions.md` (Manning Executive page, 7pt Roboto Mono listings, Heading 4 listing captions, Heading 6 figure captions)
* Audience — `mqr/mqr.md` (intermediate Python, basic DL/PyTorch, no robotics background, single consumer GPU)

---

## Deviation from the proposal

The proposal lists six sections (4.1 Tokenization and Binning → 4.2 The Categorical Distribution → 4.3 Implementing the Tokenizer → 4.4 Attaching the Action Head → 4.5 In Action → 4.6 Summary). We expand to **eight** sections:

1. Add a new **§4.1 "Why Discrete? The Multimodal Regression Trap"** in front. The proposal jumps straight to binning without first selling *why* MSE regression catastrophically fails on multimodal demos. Without this, the chapter reads as "do this because RT-1 does it" instead of "do this because the alternative provably collapses to the obstacle." The reader's "aha" moment lives here.
2. Split **§4.7 "Limits and the Bridge to Continuous Control"** from **§4.8 "Summary"** — matching Chapter 1's pattern where Summary is its own terminal section. §4.7 carries the structural-ceiling argument that motivates Chapter 5; §4.8 is a pure bulleted recap.

The proposal's 4.1 (Tokenization), 4.2 (Categorical Distribution), 4.3 (Implementing Tokenizer), 4.4 (Attaching Head), 4.5 (In Action), 4.6 (Summary) map to our **§4.2, §4.3, §4.4, §4.5, §4.6, §4.8** respectively.

---

## Chapter Opening

### "This chapter covers" block (5 bullets, gerund/verb-phrase form per `book_learnings.md` §1.1)

* Understanding why standard regression fails when expert demonstrations are multimodal — and why classification fixes it
* Designing an action tokenizer that turns continuous joint targets into a vocabulary the Transformer can predict
* Surveying the tokenizer design space (RT-1 uniform bins, RT-2/OpenVLA vocabulary reuse, BeT/VQ-BeT, FAST, cross-embodiment Cartesian)
* Building two action heads on top of the Chapter 3 backbone — a parallel baseline and the autoregressive head we ship — and seeing why per-dimension independence forces the move to autoregressive decoding
* Training the autoregressive head with cross-entropy on a LeRobot demonstration dataset, decoding action tokens left-to-right with a causal mask and a KV-cache
* Watching a per-bin softmax distribution converge from uniform noise to sharp, multimodal peaks — and seeing why decode-time sampling matters

### Hook paragraphs (3 paragraphs)

* **Paragraph 1 — Recap and pain point.** In Chapter 3 the reader spliced a SigLIP visual encoder (two cameras) and a state token into the SmolLM2 language backbone's own stream and got a sequence of 576-dim contextualized hidden states for every (images, instruction, state) triple. A real robot, however, doesn't want hidden states — it wants seven joint targets, fifty times a second. The obvious bridge is to bolt a linear regressor onto the backbone and minimize mean squared error against expert demos. This works on a toy problem for exactly as long as it takes to encounter a state where two equally valid expert behaviors exist. Then it fails — silently, catastrophically, and reproducibly.
* **Paragraph 2 — The thesis.** The fix is the same intellectual move that powers ChatGPT: stop predicting a *number* and start predicting a *distribution*, then decode it the way a language model decodes text — one token at a time, each conditioned on the last. We discretize each joint's range into bins, assign each bin a token id, and decode the action chunk autoregressively with cross-entropy. The Transformer is now doing exactly what it already does well — predict the next token — and a multimodal demo set produces a softmax with multiple peaks instead of a collapsed mean. RT-2 and OpenVLA rest on this single observation. This chapter implements it from scratch.
* **Paragraph 3 — What you will build.** By the end of the chapter, you will have a working `ActionTokenizer`, a `ParallelActionHead` (the baseline we build to feel its limits) and the `AutoregressiveActionHead` we actually ship, a training loop running on the SO-101 pick-and-place dataset, a closed-loop evaluation harness in the `PickCubeSO100` simulator, and a series of diagnostic plots that show *why* the autoregressive categorical policy works and where it still hurts. You will also have a clear list of structural ceilings — quantization error, decode-time mode collapse, sequential-decode latency, compounding covariate shift, causal confusion — that Chapter 5 dismantles by replacing the discrete bottleneck with continuous flow matching.

### Section-by-section preview paragraph

Section 4.1 demonstrates the multimodal regression failure on a one-dimensional toy problem before any robotics enters the picture. Section 4.2 introduces action tokenization and tours the design space — uniform binning, vocabulary reuse, learned codebooks, frequency-space compression, and cross-embodiment Cartesian spaces. Section 4.3 derives the cross-entropy objective and the bias–variance tradeoffs of binning, including label smoothing and the compounding-error horizon. Section 4.4 implements the tokenizer in four incremental code listings, ending with full LeRobot statistics integration. Section 4.5 builds the parallel head as a baseline, watches it sample incoherent joint combinations, and then builds the autoregressive head that fixes them — reserved-vocabulary action tokens decoded left-to-right through the Chapter 3 unified-embedding backbone — closing with training discipline (freezing, learning-rate splits, teacher forcing). Section 4.6 runs the full training loop, surfaces the entropy-collapse heatmap, shows autoregressive decoding recovering joint coordination the parallel head loses, and stress-tests the policy on a bimodal PushT state. Section 4.7 names the structural limits of discrete BC and bridges to Chapter 5's flow matching. Section 4.8 summarizes.

### Roadmap figure (Figure 4.1)

The book-wide stages diagram established in Chapter 1 (Figure 1.7), reproduced here with the **Chapter 4 stage highlighted** — the "Hands (Action Decoder)" component of the architecture overview, sitting between Chapter 3's backbone and Chapter 5's continuous head. Caption interprets where Chapter 4 sits in the larger arc.

### State-restoration ritual (first numbered section's opening)

Per `book_learnings.md` §1.1, every body chapter restores state from the prior chapter before adding new state. Section 4.2 opens with two short code snippets: (a) re-instantiating the Chapter 3 `UnifiedEmbeddingBackbone` (explicit `dtype=torch.float32` — transformers 5.x loads SmolLM2 in bfloat16 by default) and (b) calling `backbone(images, sequence_ids, state)` to confirm we still get a `[B, 392 + L + 1, 576]` tensor. No new content — just a sanity-check ritual so the reader can re-anchor.

---

## Section 4.1: Why Discrete? The Multimodal Regression Trap

**Purpose:** Sell the central idea of the chapter — predicting a *distribution*, not a *number* — before introducing any robotics machinery. The reader leaves this section convinced that MSE regression is broken for behavior cloning, not because someone said so but because they just saw it fail in front of them.

### 4.1.1 The naïve bridge: linear regression onto motor commands

* Sketch the obvious first attempt — bolt a `Linear(d_embed, action_dim)` head onto the Chapter 3 backbone and minimize `||f(o) − a||²` against expert demonstrations.
* State the supervised-learning view of behavior cloning: dataset `D = {(o_i, a_i)}` from a human demonstrator, fit `π_θ(a | o)`. This is "just supervised learning with observations as features and actions as labels."
* Note the equivalence: MSE is maximum likelihood under an isotropic Gaussian assumption. The optimum is `E[a | o]`.

### 4.1.2 The multimodal trap on a one-dimensional toy

* Describe a synthetic setup the reader can run in a notebook (no robot needed yet): a 1-D mixture-of-Gaussians target `p(a | o)` with two equally weighted modes at `a = −1` and `a = +1`.
* Train a tiny MLP with MSE. Plot the predictions. They cluster around `a = 0` — the mean of the two modes, which is the *lowest-density* region of the true distribution.
* Reframe: the model is doing exactly what we asked. We asked for the conditional mean, and the conditional mean of two valid actions is a third, invalid action. The model is correct; the loss is wrong.
* Brief note that MAE (L1) recovers the conditional median, which for symmetric bimodal distributions also lies between modes.

**Listing 4.1 — Toy bimodal regression collapse.** ~25 lines: generate a 1-D bimodal dataset, train a 2-layer MLP under `nn.MSELoss`, scatter-plot expert actions vs. predicted actions. Expected output: predictions concentrated near zero. The reader runs this once and never forgets it.

**Figure 4.2: The Multimodal Trap**

* **Left panel.** Scatter of synthetic expert actions (two clean clusters at `a = −1` and `a = +1`) with the MSE regressor's prediction line slicing through the empty middle.
* **Right panel.** Same data, y-axis discretized into 16 bins, shown as a bimodal histogram with two clean peaks. Above each panel: the loss formula being optimized.
* **Caption (2–3 sentences):** "Mean squared error minimizes the conditional mean, which for a bimodal expert distribution lies in the lowest-density region between modes. A categorical loss makes no such assumption: the same data trained against cross-entropy preserves both modes simultaneously."

### 4.1.3 Why this is fatal for robots

* Translate the toy failure to a physical setting. An obstacle on the table — the expert sometimes goes left, sometimes right, both succeed. MSE-trained policy averages the two demonstrations and drives the arm into the obstacle.
* Cite the canonical PushT failure (Chi et al., Diffusion Policy, RSS 2023) and the "left or right" cartoon (Levine, CS285). Show one frame of the bisecting-path failure.
* Generalize: expert behavior is multimodal *by default*. Multiple humans demonstrate; the same human demonstrates differently across episodes; symmetric tasks have multiple solutions. The unimodal Gaussian assumption is the exception, not the rule.

### 4.1.4 The classification reframe

* State the fix in one sentence: discretize each action dimension into `B` bins, treat each bin as a class, train with cross-entropy.
* Show the softmax formula:

  `p_θ(a_d = b | o) = exp(z_{d,b}(o)) / Σ_{b'} exp(z_{d,b'}(o))`

  Note that cross-entropy is *scale-invariant across bins* — a bin one position off costs the same as one a hundred positions off. This is the property that frees the model from mode-averaging.
* Forward-reference: §4.3 derives the cross-entropy objective formally and discusses what we lose by giving up an ordinal prior on action magnitude.

**Callout: PITFALL — Why MSE feels right and isn't.**

* MSE is the standard regression loss everywhere else in ML, and the reader's instinct is to start there.
* On *unimodal* problems (e.g., a single joint with one trajectory per state) MSE works fine — which is why the failure stays hidden until the data set scales or the task admits multiple solutions.
* The right test before choosing MSE is: "Could two skilled humans complete this task differently from the same state?" If yes, MSE collapses the modes.

**Transition:** "If actions are tokens, what's the vocabulary? Let's design the alphabet."

---

## Section 4.2: Action Tokenization — Speaking the Robot's Language

**Purpose:** Establish action tokenization as the central abstraction of the chapter, build the simplest possible scheme (uniform binning) deeply, then survey alternatives so the reader knows what they're choosing against.

### 4.2.1 State restoration: loading the Chapter 3 backbone

* Short code block re-instantiating the trained `UnifiedEmbeddingBackbone`, confirming output shape `[B, 392 + L + 1, 576]`.
* One paragraph mapping the chapter's task: bridge these 576-dim hidden states to a per-joint action prediction.

### 4.2.2 Uniform per-dimension binning — the RT-1 recipe

* Define *action tokenization*: a fixed, invertible map `q: R^D → Z^D` that converts a continuous action vector into a tuple of integer bin ids.
* The uniform per-dimension scheme: for each dimension `i`, divide its empirical range `[a_min^i, a_max^i]` into `B` equal bins. Bin id `t_i = floor((a_i − a_min^i) / (a_max^i − a_min^i) · B)`.
* Decode: the bin's *center*, `â_i = a_min^i + (t_i + 0.5) · (a_max^i − a_min^i) / B`.
* Pick `B = 256` (RT-1's choice): each bin fits in one byte, the softmax stays cheap, sub-mm resolution on typical Cartesian ranges. We use this throughout the chapter and revisit the choice in §4.3.2.
* **Worked numeric example** in prose: a single joint with range `[−π, π]`, value `0.347 rad`, `B=256` → bin id `142` → decoded center `0.349 rad` → quantization error `0.002 rad` (≈0.1°). Note that the reader will see this exact example again in Listing 4.3.

**Figure 4.3: The Action Tokenization Pipeline (intuition + worked numeric)**

* **Top panel — intuition (MotionLM-style, after Seff et al., ICCV 2023).** A vocabulary grid of quantized motion primitives drawn as arrows (each cell = one bin = one discrete joint-delta direction/magnitude), with two cells highlighted. To its left, a robot executing a reach, and below, the *action token sequence* `[t_142, t_88, …]` laid out over chunk timesteps `t=1 … H`. This is the figure's payload: continuous motion becomes a sequence of tokens drawn from a fixed vocabulary, and the left-to-right order *is* the autoregressive decode order of §4.5.4. We borrow MotionLM's visual language because it makes the "actions are a token stream" claim immediate; their figure tokenizes two cars' trajectories, ours tokenizes one arm's joint chunk.
* **Bottom panel — worked numeric.** Five horizontal boxes left-to-right: raw `0.347 rad` → normalize-to-`[0,1]` → multiply by `B=256` → `floor` → clamp to `[0, B−1]` → token id `142`. Below each box, the numeric value at that stage. Inverse arrow above the chain: `142` → bin-center lookup → decoded `0.349 rad`.
* **Caption:** "Top: an action chunk is a sequence of tokens drawn from a fixed vocabulary of quantized motion primitives, decoded left to right (the autoregressive order of section 4.5). Bottom: uniform per-dimension binning converts one continuous joint target into one integer token; the decode step returns the bin center, so the round-trip error is bounded by half a bin width."

### 4.2.3 Vocabulary reuse — the RT-2 / OpenVLA trick

* The next step beyond standalone bins: instead of inventing 256 new tokens, *overwrite* existing tokens in the language model's vocabulary.
* RT-2-PaLI-X reuses tokens 0–999 (PaLI-X has dedicated integer embeddings).
* RT-2-PaLM-E identifies the 256 *least-used* tokens and overwrites them as `<act_0>…<act_255>`.
* OpenVLA overwrites the last 256 tokens of Llama-2's 32K BPE vocab. Each action timestep is 7 tokens (6-DoF end-effector + gripper). Reported eval speed ~6 Hz on A100.
* Why this matters: the language model head can be reused verbatim. One cross-entropy objective trains both language and action tokens. The model is, literally, a polyglot.
* **Decode-time constrained sampling:** at inference, mask logits over non-action token ids to `−∞` so the policy can only emit valid action tokens. Forward-reference §4.6.4.

**Figure 4.4: Vocabulary Alignment Triptych**

* Three side-by-side stacked-cell strips representing token vocabularies:
  * **(a) RT-1:** 256 standalone action cells, no language tokens.
  * **(b) RT-2 (PaLI-X):** Long horizontal vocab strip with the integer-token region highlighted.
  * **(c) OpenVLA:** Llama-2's 32K vocab strip with the *last* 256 cells highlighted (the overwritten low-frequency tokens).
* **Caption:** "Three strategies for giving a robot an action vocabulary. RT-1 uses a dedicated 256-token vocabulary disconnected from any language model. RT-2 reuses pre-existing integer tokens. OpenVLA overwrites the least-frequent tokens of a pretrained language model, preserving the LM's text-generation ability for the other 99% of its vocabulary."

### 4.2.4 Cross-embodiment tokenization — actions as a lingua franca

*Material drawn from `gemini_research_synthesis.md` §2.B; folded in here because tokenization is what makes cross-embodiment possible.*

* The Open X-Embodiment finding (Padalkar et al., 2023): one tokenization scheme + one Transformer can control radically different robot bodies (a Google arm, a Franka, a quadruped) as long as actions are *normalized per-dataset* and expressed in a *common action space*.
* The common space is typically end-effector Cartesian: 3-D translation, 3-D rotation (axis-angle), 1-D gripper. This bypasses each robot's specific joint geometry — the model doesn't need to know whether the arm has 6 or 7 joints, only "where should the hand be next?"
* Per-dataset normalization (Q01/Q99 statistics per robot, covered in §4.4.4) ensures that a "maximum positive move" token maps to whatever range that particular robot can physically execute.
* Emergent positive transfer: training on a 970k-episode mixture lifts performance on under-represented robot types. The model learns "manipulation," not "this particular arm."
* Scoping note: **we don't train cross-embodiment in this book** — PushT and ALOHA are enough. But tokenization is the mechanism that makes such scaling possible, which is why we mention it now.

**Figure 4.5: Cross-Embodiment via Per-Dataset Normalization** *(MotionLM-style shared-vocabulary layout)*

* **Right — one shared vocabulary panel.** The same arrow-grid vocabulary from Figure 4.3, drawn once, dashed-boxed and labeled "Vocabulary" (mirrors MotionLM's shared-vocabulary panel — where they share one vocab across two *agents*, we share one vocab across two *embodiments*).
* **Center — one shared token sequence** "..., t_42, t_88, t_153, ..." drawn from that vocabulary.
* **Left — two divergent decoders.** Two arrows from the shared sequence pointing to two robot diagrams, each with its own per-dataset normalization lookup:
  * **Top:** an SO-100 6-DoF tabletop arm; stats `{q01_SO100, q99_SO100}`; decoded action: ±2 cm move.
  * **Bottom:** a Franka Panda 7-DoF arm; stats `{q01_Franka, q99_Franka}`; decoded action: ±15 cm move.
* **Caption:** "One vocabulary, two bodies. The same action token decodes to different physical motions on different robots, because each dataset's normalization statistics rescale the bins to that robot's reachable range. This is how a single autoregressive VLA can be trained jointly on data from many embodiments — and how knowledge transfers from data-rich robots to data-poor ones."

### 4.2.5 The wider design space — a guided tour

A side-by-side comparison table (Wang/Szeto triptych pattern, `book_learnings.md` §4.3), so the reader sees where uniform binning sits among modern alternatives. Each row gets one paragraph of body text, *not* an implementation.

**Table 4.1: Action Tokenizer Families**

| Method | Discrete unit | Vocab size | Captures inter-dimension correlation? | Inference speed | When to use |
|---|---|---|---|---|---|
| RT-1 uniform bins | per-dim bin | 256/dim | no | very fast (parallel) | teaching, single-step prediction, the chapter's choice |
| RT-2 / OpenVLA vocab reuse | LM token id | 256 reserved | no | slow (autoregressive) | grounding in pretrained language priors |
| BeT (k-means + offset) | cluster id + residual | K ≈ 64 + R^D | yes | fast | small multimodal datasets |
| VQ-BeT (residual VQ-VAE) | codebook id | 16×16 = 256 | yes | fast | PushT, kitchen-style benchmarks |
| FAST / FAST+ (DCT + BPE) | BPE token | ~1024 | yes (chunk-level) | very fast (few tokens per chunk) | universal cross-embodiment, π0-FAST |
| OAT (coarse-to-fine) | hierarchical register | small | yes | tunable | latency-critical edge deployment |
| ACT (CVAE chunk) | continuous latent | n/a | yes | one-shot | bimanual fine motor, ALOHA |

* Cover each row in 1 paragraph: who proposed it, what problem it solves vs. uniform binning, what it costs.
* **Honest scoping disclaimer (`book_learnings.md` §1.4):** "We implement the RT-1 / OpenVLA family in detail. The other rows are surveyed so you know the design space — full implementations live in the cited papers and reference repos linked in Further Reading."

**Callout: DEFINITION — Action chunking.** Predicting `H` future actions from one observation instead of one action per step. Smooths motion, attenuates compounding error. We adopt it from §4.4 onward (default `H=16`).

**Transition:** "Tokenization is half the story. The other half is the loss function that turns a sequence of bin ids into a trained policy — cross-entropy. Why does cross-entropy do something MSE provably cannot? That's the next section."

---

## Section 4.3: The Mathematics of the Categorical Distribution

**Purpose:** Give the reader a clean, MQR-appropriate derivation of cross-entropy in this setting, name the key tradeoffs (quantization vs. sequence length, class imbalance, compounding error), and surface advanced loss variants without losing them.

This section is the chapter's math-heavy block. Per `book_learnings.md` §3, it opens with a NOTATION callout, derives one transformation per paragraph, and closes every derivation with an "in plain English" recap.

### 4.3.1 Notation callout (top of section)

* `o` — observation (image + instruction + proprioception), encoded by the Chapter 3 backbone to `h ∈ R^d`.
* `a ∈ R^D` — continuous action (`D` joints).
* `q: R^D → {0, …, B−1}^D` — the tokenizer from §4.2.
* `t_d = q(a_d)` — the bin id for action dimension `d`.
* `z_{d,b}(h)` — logit for dimension `d`, bin `b`. Produced by the action head, with shape `[B_batch, D, B]`.
* `p_θ(a_d = b | o)` — softmax over the bin axis.
* `H(p)` — entropy of a categorical distribution, in nats.

### 4.3.2 The cross-entropy objective, derived

Two paragraphs of derivation, one transformation per paragraph, each transformation *named* (per `book_learnings.md` §3.2).

* **Autoregressive factorization (the chapter's objective; RT-2 / OpenVLA).** Apply the chain rule of probability exactly, with no independence assumption — predict each token given everything decoded before it (earlier joints in this timestep and all earlier timesteps), flattening the chunk to `H·D` tokens:

  `p_θ(a | o) = ∏_t ∏_d p_θ(a_{t,d} | o, a_{<(t,d)})`

  This is the *exact* factorization of the joint distribution — no terms dropped — so it can represent any correlation across joints and across time. Plain-English recap: the head is a tiny language model over the action chunk; this is the objective we train. The cost is `H·D` sequential decode steps (§4.5.4–§4.5.5 make them cheap with a KV-cache and amortize them over the chunk).
* **Per-dimension factorization (the parallel-head foil).** *Assume* conditional independence across action dimensions given `h`:

  `p_θ(a | o) = ∏_d p_θ(a_d | o)`

  The negative log-likelihood becomes a sum of independent per-dim cross-entropies, computed in one parallel forward. Plain-English recap: each joint gets its own softmax, but the head can no longer represent "if joint 1 closes the gripper, joint 2 should lift." We build this first (§4.5.2) precisely to watch it fail on coordinated joints (§4.5.3), which is what motivates the autoregressive objective above.
* **As N → ∞, NLL minimization recovers the true conditional `p*(a | o)`** for the autoregressive factorization (the parallel factorization recovers only the per-dim marginals). The true conditional may be arbitrarily multimodal *and* correlated across joints. This is the formal version of §4.1's "predict a distribution, not a number" claim, extended to "and decode it coherently."

**Callout: DEEP DIVE — Why is cross-entropy scale-invariant across bins?**

* `L_CE = −log p_θ(b*)` depends only on the predicted probability of the *correct* bin, not on the distance from any other bin.
* This is the property that prevents mode collapse: a model is not penalized for putting probability mass at bin 200 *while* the target is bin 50, the way MSE would penalize predicting `a = 200` when the target is `a = 50`.
* The cost: cross-entropy has no ordinal prior. Predicting bin 51 when the target is bin 50 is treated the same as predicting bin 200. §4.3.5 (label smoothing) and §4.3.7 (Fourier head) discuss remedies.

### 4.3.3 The quantization–sequence-length dilemma

* The bin width gives a quantization error variance of `(a_max − a_min)² / (12 B²)` (standard uniform-quantizer result). Halving the error means doubling `B`.
* The token budget for an action chunk of horizon `H` across `D` dims is `L = D · H` tokens per inference call.
* Two design pressures pull in opposite directions:
  * For accuracy: large `B` (fine bins), short `H` (cheaper rollouts) — but short `H` means more decisions per second, exacerbating compounding error.
  * For latency at 50 Hz: small `L` — but small `L` means coarse bins or short chunks.
* **Plain-English recap:** "Discrete BC's pain point is fundamental: you can have fine resolution or short token sequences, but rarely both. FAST (§4.2.5) is the field's most successful attack on this dilemma."

**Bulleted "what vanishes / what survives" recap (Lambert pattern, `book_learnings.md` §3.2):**

* **What survives:** the categorical's ability to represent arbitrary multimodal shapes; the per-dim independence assumption (in the parallel head).
* **What vanishes:** ordinal information about bin distance; the ability to predict values *between* bin centers.

### 4.3.4 Per-dimension independence: when it bites

* The parallel head models `p(a_1, …, a_D | o) = ∏_d p(a_d | o)`. If two joints must move *together* (a coordinated grasp) the head can place high probability on `(close-gripper, lift-up)` *and* `(open-gripper, drop)` independently — and then nothing prevents it from sampling `(close-gripper, drop)` at decode time, which is the worst of both.
* In practice this is rare on simple tasks but appears on contact-rich bimanual ALOHA work.
* Mitigation in this chapter: short chunks + argmax decode for the demo policy; flag autoregressive heads (§4.5) and VQ-BeT (§4.2.5) as the structural fixes.

### 4.3.5 Label smoothing on *adjacent* bins

*Material from `gemini_research_synthesis.md` §3.B; this is the simplest ordinal-aware loss tweak.*

* Standard label smoothing spreads probability mass uniformly across all bins (e.g., `0.95` on the target, `0.05/(B−1)` on every other bin). For an ordinal vocabulary this is wasteful — bin 200 is no more wrong than bin 51 when the target is bin 50, which is exactly the *opposite* of physical truth.
* **Adjacent-bin label smoothing:** put smoothing mass on bins `[b*−k, b*+k]` only, with a small fixed weight (e.g., 0.05 split across the neighbors). This restores a weak ordinal prior without rebuilding the loss from scratch.
* Show the modified target distribution as a one-line equation, point at where it slots into the cross-entropy in §4.6's training listing.
* **Plain-English recap:** "We're telling the loss that being off by one bin is much better than being off by 100. Cross-entropy doesn't know this by default; one extra line of code teaches it."

### 4.3.6 Compounding error and the horizon argument

* The Ross & Bagnell observation (DAgger, 2011): under bounded per-step 0/1 error `ε`, vanilla behavior cloning's regret over horizon `T` is `O(T² ε)`. DAgger reduces this to `O(T ε)` by querying the expert on the learner's distribution.
* The sharper result (Foster et al., *Is Behavior Cloning All You Need?*, NeurIPS 2024): under **log-loss** (i.e., cross-entropy on discrete actions), behavior cloning enjoys horizon-independent `O(ε)` regret in many problems, with worst-case `Ω(min(H, 1/(1−γ))² · ε)`.
* This is a real argument *for* the categorical approach the chapter has been building. Cross-entropy *is* log-loss; we've been getting horizon-favorable regret for free.
* Practical implications:
  * **Chunking is the right move.** Reduces decisions from `T` to `T/H`.
  * **Temporal ensembling** (predict overlapping chunks, weight-average per-timestep) absorbs jitter from sparse decisions.
* **Plain-English recap:** "Discrete actions aren't only good for multimodality — under the right loss, they also blunt one of imitation learning's oldest theoretical curses."

### 4.3.7 Beyond standard cross-entropy (brief tour)

* **Focal loss** (Lin et al., 2017): down-weights easy bins (e.g., the "zero velocity" bin that dominates teleoperation data). One paragraph, one equation, used by BeT.
* **Fourier head** (Gillman et al., NeurIPS 2024): parametrize logits as a truncated Fourier series in bin position. Smooth-by-construction softmax, still multimodal. Drop-in for `Linear(d_embed, B)`. One paragraph, no implementation.
* **Implicit BC** (Florence et al., CoRL 2021): energy-based, recovers actions via argmin. Sibling alternative; mention and move on.

**Callout: PITFALL — Class imbalance from teleop data.** Roughly 50% of teleop timesteps have a stationary gripper. Vanilla cross-entropy puts most gradient mass on the "zero" bin and the network learns to output zero. Diagnostic: log a histogram of bin frequencies. If the modal bin exceeds ~20% of all tokens, switch to focal loss (γ=2). We surface this in the training-diagnostics section (§4.6).

### 4.3.8 Decode-time sampling — the loss is only half the story

Brief preview (full coverage in §4.6.4): even when training succeeds and the softmax is bimodal, naïve `argmax` decoding collapses it to a single mode. This is the canonical decode-time bug.

**Transition:** "We have a target distribution and a loss. Let's build the machinery that produces them — starting with the tokenizer."

---

## Section 4.4: Implementing the Tokenizer

**Purpose:** Build `ActionTokenizer` end-to-end through the "Raschka micro-loop" (`book_learnings.md` §2.2) — four progressive listings, each followed by a "let's run it" block, each adding one capability. The reader can copy-paste the final class and use it for the rest of the chapter.

### 4.4.1 Skeleton — the class interface

* Define the API first: `encode(action: np.ndarray) → ids` and `decode(ids: np.ndarray) → action`, plus constructor accepting per-dimension `lo`, `hi`, `n_bins`.
* **Listing 4.2 — `ActionTokenizer` skeleton.** ~15 lines, just the class signature, `__init__`, and `NotImplementedError` stubs. Annotation labels `#A` (constructor stores stats), `#B` (encode signature), `#C` (decode signature).
* Quick `print(tok)` check to confirm it instantiates.

### 4.4.2 Single-joint encode — the worked numeric

* **Listing 4.3 — Encode a single joint.** ~10 lines, scalar path. Bounds-clip, normalize to `[0, 1]`, multiply by `n_bins`, floor, integer-clip to `[0, n_bins−1]`.
* "Let's run it." Input `0.347` with range `[−π, π]`, `B=256`. Expected output: `142`. The same example from Figure 4.3.
* Decode the result: bin 142 → center `0.349`. Quantization error: `0.002 rad`.

**Callout: TIP — Off-by-one at the upper edge.** A common bug is forgetting the upper clip — an action exactly equal to `a_max` floors to `B` (out of range). One line of `np.clip(..., 0, B−1)` fixes it. The unit test is one line; write it.

### 4.4.3 Vectorized encode/decode — the production form

* **Listing 4.4 — Vectorized encode/decode over `[..., D]`.** ~20 lines. Use NumPy broadcasting so `lo`, `hi`, `edges`, `centers` are all `[D]`-shaped and a single call handles any batch.
* "Let's run it." Encode a `[8, 7]` batch of fake joint targets, print the resulting `[8, 7]` integer tensor. Decode, print round-trip error per dimension.
* Tensor-shape prose, per `book_learnings.md` §2.4: "The result has shape `[batch_size, D]` where each entry is in `{0, …, B−1}`."

### 4.4.4 LeRobot statistics integration

* The real workflow doesn't hardcode `lo` and `hi` — it reads them from a dataset's `meta/stats.json`. Walk through the three normalization strategies LeRobot ships:

  | Strategy | Use for |
  |---|---|
  | `MEAN_STD` | proprioception, image pixels (with ImageNet stats) |
  | `MIN_MAX` | last resort, sensitive to outliers |
  | **`Q01_Q99`** | **default for actions** — robust to teleop outliers |

* Q01/Q99 wastes ~2% of bin range on tails but produces uniform bin utilization on the body of the distribution — exactly the property cross-entropy benefits from.
* **Listing 4.5 — `ActionTokenizer` with LeRobot stats.** ~25 lines: open `lerobot/pusht/meta/stats.json`, extract `q01`/`q99` for the action key, construct the tokenizer, encode the first 32 frames of the dataset.

**Callout: DEFINITION — LeRobotDataset format.** A LeRobotDataset folder contains `meta/*.json` (schema, episode lengths, instructions, normalization stats), `data/chunk-XXX/episode_NNNNNN.parquet` (per-frame state and action), and `videos/chunk-XXX/observation.images.<cam>/episode_NNNNNN.mp4` (H.264-encoded images decoded on the fly). The MP4 storage is what makes ALOHA's 50 demos × 400 frames × 2 cameras fit in <1 GB.

### 4.4.5 OpenVLA-style vocabulary reuse (sidebar listing, optional path)

* Brief listing showing how to overwrite the last 256 tokens of a HuggingFace tokenizer. Not required for the rest of the chapter; included so the curious reader can see the OpenVLA trick in code.
* **Listing 4.6 (sidebar) — Vocabulary-reuse tokenizer.** ~30 lines, the OpenVLA reference pattern from `claude_research_summary.md` §5.3.
* Note that this listing is the only one with `transformers` as a dependency.

### 4.4.6 Sanity-check pass

* Encode + decode the entire PushT action set, plot the round-trip-error histogram. Expect near-zero mean, max error ≈ half a bin width.
* **Reader-trust pattern (`book_learnings.md` §2.5):** "If your error histogram has a long tail, you've probably mis-set `lo`/`hi`. The most common cause is using `min`/`max` instead of `q01`/`q99` on a dataset with outliers."

**Exercise 4.1 — Bin sweep.** Re-tokenize ALOHA cube actions at `B ∈ {64, 256, 1024}`. Report per-joint reconstruction error and disk size of the resulting integer-encoded dataset. Pick a sweet spot and justify. Solution in Appendix.

**Transition:** "The tokenizer turns continuous actions into bin ids. The action head learns to predict those ids from the Chapter 3 embedding. That's the next layer up."

---

## Section 4.5: Attaching the Action Head

**Purpose:** Build two heads. First the parallel linear head, as a baseline, so the reader can feel its one structural weakness directly; then the autoregressive head — the chapter's shipped artifact — that fixes it by decoding action tokens left-to-right through the Chapter 3 unified-embedding backbone. Lay out the training discipline (freezing, learning-rate splits, teacher forcing) that matters for hooking onto a pretrained backbone.

### 4.5.1 Three head archetypes

Side-by-side comparison so the reader sees the choice (Wang/Szeto triptych pattern). We build the first two — the parallel head as a foil, the autoregressive head as the real thing.

**Table 4.2: Discrete Action Head Architectures**

| Head | Mechanism | Latency (per chunk) | Captures inter-dim correlation? | Used by | This book |
|---|---|---|---|---|---|
| Linear / MLP parallel | `Linear(d_embed → H · D · B)`, reshape, per-dim softmax | very low (1 forward) | no | RT-1, simple BC | **baseline foil — §4.5.2** |
| Autoregressive | reserved-vocab action tokens decoded `H · D` steps through the causal pretrained backbone | moderate (`H · D` cached steps, once per chunk) | yes | RT-2, OpenVLA, MotionLM | **shipped — §4.5.4** |
| Decoupled / parallel-decode | pretrained kinematic decoder modulated by VLM features | low | partial | PD-VLA, recent research | survey only |

**Figure 4.6: Action Head Architectures (Three-Way Comparison)**

* Three vertical panels with the *same* incoming context from the Chapter 3 backbone splitting into three head designs.
  * **(a) Parallel linear (foil).** Final state token → `Linear(d, H·D·B)` → reshape `[H, D, B]` → independent softmax per `(t, d)`. One forward, no cross-token conditioning.
  * **(b) Autoregressive (shipped).** Reserved-vocab action-token embeddings appended to the fused sequence; the causal pretrained backbone decodes `H·D` tokens one at a time, each attending to the perception prefix and all earlier action tokens.
  * **(c) Decoupled.** Fusion context modulates (via FiLM or cross-attention) a separately pretrained "kinematics decoder."
* Annotations below each panel: approximate parameter count; latency for `D=6, H=16` (parallel = 1 forward; AR = 96 cached steps once per chunk); whether joint coordination is representable.
* **Caption:** "Three places to attach a discrete action head. The parallel linear head is fast and easy to train but assumes per-dimension independence — it cannot represent that two joints must move together. The autoregressive head decodes action tokens left to right through the same causal transformer, conditioning each token on the ones before it, so joint coordination is representable; its cost is sequential decode, amortized over the chunk with a KV-cache. Decoupled heads pretrain a kinematics-only component to bypass that bottleneck. We build the parallel head as a baseline and ship the autoregressive head."

### 4.5.2 The parallel head as a baseline — `ParallelActionHead`

* **Listing 4.7 — `ParallelActionHead`.** ~25 lines. A two-layer MLP (GELU) that reads the backbone's final state token and projects `d_embed → action_dim · chunk_len · n_bins`, reshapes to `[batch, H, D, B]`, exposes `forward`, `loss`, and `predict`.
* Annotation labels: `#A` store `(D, H, B)`; `#B` two-layer MLP with GELU; `#C` reshape so the bin axis is last; `#D` `F.cross_entropy`; `#E` `predict` returns decoded action via `bin_centers[ids]`.
* "Let's run it." Instantiate on a `[4, 576]` fake backbone summary, forward → assert shape `[4, 16, 6, 256]`. Compute loss against random integer targets, assert near `log 256 ≈ 5.54` (untrained = uniform).
* Sanity-check sidebar: "An untrained categorical head should output exactly `log(B)` nats of entropy on average. If not, your initialization is biased."
* **Cross-entropy loss with adjacent-bin smoothing (Listing 4.8).** ~15 lines; mass `1 − ε` on the correct bin, `ε/2` on each neighbor (clamped at the boundary). Connect to §4.3.4's math.

**Callout: DEEP DIVE — Gaussian mixtures, the continuous cousin.** The parallel head fixes within-dimension multimodality with discrete bins. A mixture density network (MDN) does the same continuously — each dimension predicts `K` Gaussian components (means, variances, mixing weights) trained by NLL. We do *not* use it as the foil, for three reasons: (a) a *per-dimension* MDN has the identical inter-dimension independence flaw we are about to expose in §4.5.3, so it would not strengthen the argument for autoregression; (b) it is continuous density estimation, which is Chapter 5's territory (flow matching) and would blur the discrete/continuous boundary the two chapters are organized around; (c) MDNs are fiddly to train (variance collapse, log-σ clamping, mode dropping). A *joint* multivariate GMM (full covariance) or a CVAE — i.e., ACT — is the parallel route that *would* capture inter-joint correlation; we survey those in Table 4.1 rather than build them. Forward-pointer: Chapter 5 makes the continuous-multimodal head the main event.

### 4.5.3 Where the parallel head breaks — inter-dimension coupling

* This is the chapter's Act-3 hinge. The parallel head predicts each joint's marginal correctly, but factorizes the joint distribution as a product of marginals. On a coordinated motion it can place high marginal probability on `close-gripper` (joint 1) and on `drop` (joint 6) — each correct in isolation — and then sample the incoherent combination `(close-gripper, drop)`: a closed, empty gripper held over the bin.
* Demonstrate it concretely: take a held-out state where the demonstrations require a coordinated `(grasp, lift)`. Sample the trained parallel head many times; plot the joint histogram over (gripper, wrist) bins. Show mass leaking into the two off-diagonal "impossible" quadrants.
* State the fix in one sentence — condition each joint on the joints already decoded, i.e., the autoregressive factorization from §4.3.1 — and turn the page to build it.

### 4.5.4 The autoregressive head — `AutoregressiveActionHead` (the shipped head)

* **The mechanism, adapted to our stack.** Like OpenVLA, our Chapter 3 backbone fuses inside the pretrained LM itself: SigLIP tokens and the state token are spliced into SmolLM2's own 576-dim stream (`UnifiedEmbeddingBackbone`). So "reuse the LM head" becomes: reserve the last 256 ids of SmolLM2's vocab as `<act_0..act_255>` (the reservation lives in Ch4's tokenizer, §4.4.5, not Ch3 source), embed action tokens, append them to the fused sequence `[img | lang | state | a_1 … a_{H·D}]`, run the existing causal backbone, and read each action position's logits through a 576→256 action head. One shared cross-entropy objective over the action positions.
* **Token order.** Per-timestep then per-dim: `a^1_1..a^1_D, a^2_1..a^2_D, …` (RT-2 / OpenVLA order), `H·D` tokens.
* **Training: teacher forcing.** Feed the ground-truth shifted action tokens as decoder input; the causal mask guarantees position `i` never sees token `i` or later. One forward computes the loss over all `H·D` positions in parallel at train time (the sequential cost is inference-only). Flag exposure bias / scheduled sampling as the known AR gotcha.
* **Listing 4.9 — `AutoregressiveActionHead` (training/teacher-forcing path).** ~30 lines: embed reserved action tokens, append to fused sequence, causal forward, gather action-position logits, `F.cross_entropy` against shifted targets.
* Annotation labels: `#A` reserved-vocab embedding lookup; `#B` append to fused prefix; `#C` causal mask over the whole sequence (Ch3's mask, unchanged); `#D` gather the `H·D` action positions; `#E` cross-entropy against teacher-forced targets.
* "Let's run it." Forward on a `[4, …]` batch → assert action-logit shape `[4, H·D, 256]`; loss near `log 256` untrained.

### 4.5.5 Inference — KV-cache, constrained decoding, and the latency budget

* **The decode loop (Listing 4.10).** Greedy/sampled left-to-right decode of `H·D` tokens with a KV-cache so each step is one cached forward, not a full re-forward. Constrained decoding: mask logits outside the reserved 256 action ids to `−∞` before sampling. Temperature + top-p carry over from §4.6.4.
* **The latency budget — why AR is fine here.** Chunking decouples inference rate from control rate. We run inference once per chunk and execute all `H` steps open-loop: `H·D = 96` cached decode steps happen at ~3 Hz inference while the controller ships 50 Hz commands. Spell out the arithmetic and contrast with the naive "96 full forwards per timestep at 50 Hz" reading that made AR look impossible. The residual costs (sequential decode even with a cache; the discrete quantization bottleneck) are what Chapter 5 removes.
* **Action chunking (the DEFINITION from §4.2 lands here too).** Default `H=16` (320 ms at 50 Hz); production VLAs (π0) use `H=50`. Execute the chunk, then re-observe (or overlap and temporally ensemble, §4.6.4).

### 4.5.6 Training discipline — freezing, learning rates, warmup, teacher forcing

Per the research dossier, the rules of thumb that matter enormously for a healthy run (the first three apply to both heads; teacher forcing is autoregressive-specific):

* **Freeze the Chapter 3 backbone for epoch 1.** The action head starts at random init; its gradients are huge and noisy. Backprop into the backbone in this state will corrupt good vision/language features. Freeze → head learns the action vocabulary → unfreeze.
* **Split learning rates after unfreezing.** Head LR `1e-4`, backbone LR `1e-5`. This is the standard fine-tuning idiom; it preserves Chapter 3's representations while still letting them shift toward action-relevant features.
* **500-step linear warmup + cosine decay to zero.** Stops the AdamW updater from over-shooting in the first dozen steps when gradient norms are largest.
* **Teacher forcing (autoregressive head only).** Train on ground-truth shifted action tokens, not the model's own samples, so all `H·D` positions are supervised in one forward. Note the train/inference mismatch this creates (exposure bias): at inference the head conditions on its *own* prior tokens, which can drift. Mention scheduled sampling as the standard mitigation; in practice short chunks plus the closed-loop eval of §4.6 keep it in check. This is the one extra discipline the autoregressive head needs over the parallel head.

**Callout: PITFALL — Backbone corruption from unfrozen training.** A symptom: training loss drops smoothly but closed-loop success rate drops, then climbs back. The head was learning while the backbone was being scrambled by its gradients. Always freeze the first epoch.

**Exercise 4.2 — Loss surgery.** Swap cross-entropy → focal loss (γ=2.0). Train 5,000 steps. Report recall on bins appearing in <0.1% of training data (e.g., the gripper-transition bins). Solution in Appendix.

**Transition:** "We have a tokenizer, a head, and a training discipline. Time to run it."

---

## Section 4.6: In Action — Training and Watching Convergence

**Purpose:** Pull it all together — load data, train, evaluate in closed-loop, and (the chapter's visual payoff) watch the softmax converge from uniform noise to multimodal peaks.

### 4.6.1 The training recipe

* **Table 4.3: Chapter 4 Training Recipe.**

  | Item | Value |
  |---|---|
  | Optimizer | AdamW, β = (0.9, 0.95), wd = 0.05 |
  | Head LR | 1e-4 |
  | Backbone LR (after epoch 1) | 1e-5 |
  | Schedule | linear warmup 500 → cosine decay to 0 |
  | Gradient clip | global norm 1.0 |
  | Precision | bf16 autocast (Ada / Hopper / A100) |
  | Gradient accumulation | 2–4 → effective batch 32–64 |
  | Batch — PushT (1 cam, 96²) | 64 micro |
  | Batch — ALOHA (2 cam, 480×640, H=16) | 8 micro × 4 accum |
  | Chunk length `H` | 8 (PushT) / 16 (ALOHA) |
  | Loss | cross-entropy (+ optional adjacent-bin smoothing ε=0.05) |
  | Training time (PushT, single 4090) | ~6–12 h over 20–40k steps |

* **Listing 4.9 — Data loader with `delta_timestamps`.** ~20 lines. Constructs a `LeRobotDataset` for `lerobot/pusht`, passes `delta_timestamps={"action": [t/10.0 for t in range(8)]}` for an 8-step chunk. Shows that `delta_timestamps` is the LeRobot mechanism that does temporal stacking and asserts episode-boundary integrity.

### 4.6.2 The training loop

* **Listing 4.10 — Training loop.** ~30 lines. AdamW + scheduler + bf16 autocast + gradient accumulation + freeze→unfreeze switch at epoch 1. The single most concrete artifact of the chapter.
* "Let's run it." Show the first few logged lines: step 0 loss ≈ 5.54, step 500 ≈ 3.2, step 5000 ≈ 1.8.
* Note: the GitHub repo contains the full script with all bells and whistles; the listing shows the essentials.

**Callout: NOTE — Hardware expectations.** On a single RTX 4090, PushT runs comfortably at the batch sizes above. On Google Colab T4 (16 GB), drop microbatch to 32 and use bf16 (the T4 supports it via Turing emulation but is slow — expect 4× wall-clock vs. 4090). ALOHA's larger images may not fit on T4; see Appendix for memory-tight settings.

### 4.6.3 The convergence figure — the chapter's payoff

This is the single image the reader should remember from the chapter (the "fold this onto your wall" mental-model figure from `book_learnings.md` §2.1).

**Figure 4.7: Entropy Collapse Heatmap**

* A 2-D heatmap: X-axis = training step (0 → 20k, log-spaced ticks), Y-axis = 256 bin ids for one representative joint, color = predicted probability on a held-out state.
* At step 0: uniform horizontal band (entropy ≈ `log 256 ≈ 5.54` nats).
* By step 1k: a single broad peak in the middle (mode-averaging — exactly the MSE failure).
* By step 5k: the peak splits into two ridges (the bimodal target).
* By step 20k: two sharp ridges, very low cross-ridge density.
* **Caption (3–4 sentences):** "The softmax of a categorical action head, as it learns. The uniform band at step 0 corresponds to maximum entropy — every bin is equally likely. By step 5,000 the model has discovered the two valid push directions for this PushT state. By step 20,000 the ridges are sharp and well-separated. This visualization is the most direct demonstration in the chapter that the policy is representing multimodality rather than collapsing it."

### 4.6.4 Closed-loop evaluation — and the decode-time trap

* Open-loop loss is a *bad* proxy for closed-loop success. A 5% open-loop drop can correspond to a 30% closed-loop drop. Always run rollouts.
* **Listing 4.11 — Closed-loop rollout on PushT.** ~15 lines. Reset env, call `policy.select_action(obs)` in a loop, accumulate reward, count terminations with `t-iou ≥ 0.95`. Run 50 seeds, report mean ± stderr.
* **The four decode-time sampling strategies** and when to use each:

  | Strategy | Formula | When to use |
  |---|---|---|
  | `argmax` | `b̂ = argmax_b p_b` | benchmark determinism; **collapses multimodality** |
  | Temperature | `p^{(τ)}_b ∝ exp(z_b/τ)`, sample | `τ ∈ [0.5, 1.0]` default |
  | Top-p (nucleus) | smallest set with cumulative `p ≥ p_target` | prevents junk-bin kinematic spikes |
  | Expected value | `â = Σ_b p_b · c_b` | smooth, sub-bin; **dangerous on bimodal — averages modes!** |

* **Default recipe for the rest of the book:** temperature `τ=1.0` + top-p `0.95`. Argmax only for ablations. Expected-value only for dimensions known to be unimodal (e.g., grip force at a given state).

**Callout: PITFALL — Expected-value decoding looks free, isn't.** It feels like a smoothing trick that gets sub-bin resolution at no cost. On bimodal states it reconstructs *exactly* the bisecting path the chapter's whole machinery was built to avoid. If §4.1's toy failure surprised you, this is the same failure mode wearing a different hat.

### 4.6.5 The bimodal stress test — the chapter's "the trick works" moment

* Pick a held-out PushT state where two valid push directions exist.
* Plot the predicted softmax over the X-coordinate of the next target push, side-by-side with the same plot from an MSE-trained baseline.

**Figure 4.8: Bimodal Convergence (Categorical vs. MSE)**

* Two side-by-side density plots of the predicted action distribution on a single PushT bimodal state.
* **Left (MSE baseline):** a single Gaussian centered between the two true modes — the collision-causing bisecting prediction.
* **Right (categorical head, ours):** two clean peaks at the two valid push X-coordinates.
* Below each plot: the path the policy would execute if sampled. Left: into the T-block. Right: around it.
* **Caption:** "On a held-out bimodal state, the MSE policy collapses to the conditional mean — exactly the action that drives the arm into the obstacle. The categorical policy preserves both modes; sampling from it produces a coherent path around either side of the obstacle. This is the failure-and-fix that drove the chapter."

### 4.6.6 Diagnostic dashboard — what to log

A short checklist of things to log every run, beyond loss curves:

* Mean predicted entropy per dimension (expect drop from `log B ≈ 5.54` to ~2–3 nats).
* Bin-frequency histogram on the training set (catches class imbalance; if the modal bin exceeds 20%, switch to focal loss per §4.3.7).
* Snapshot of the full softmax on 4 canary states at steps {0, 1k, 5k, 20k, end}.
* Closed-loop success rate every 5,000 steps, 50 seeds (the only metric that actually matters).

**Exercise 4.3 — Multimodal stress test.** Train discrete BC on PushT for 50,000 steps. Compute success rate over 100 rollouts. Visualize a state where `argmax` lands in the saddle between two valid push directions — *this is the hook for Chapter 5.* Solution and screenshots in Appendix.

**Exercise 4.4 — Entropy diagnostic.** On a held-out episode, plot per-timestep softmax entropy per joint. Identify *decision points* (high entropy) versus *committed motions* (low entropy). Solution in Appendix.

**Exercise 4.5 — Vocab alignment.** Implement RT-2's "overwrite 256 least-used tokens" trick on a SmolLM tokenizer. Verify that the overwritten strings are genuinely rare (count their occurrences in a small text corpus). Solution in Appendix.

**Transition:** "The policy works. It also has ceilings — structural ones the categorical loss cannot fix. Chapter 5 dismantles them. Here's what's still wrong, and where we go next."

---

## Section 4.7: The Limits of Discrete BC — and the Bridge to Chapter 5

**Purpose:** Name the structural ceilings of the chapter's approach honestly. This sets up Chapter 5 (flow matching) and Chapter 7 (RL refinement), both of which the proposal commits to.

### 4.7.1 The four structural ceilings

* **Quantization vs. sequence-length dilemma (revisited).** Fine manipulation at 50 Hz wants `B ≫ 256` per dimension *and* short token sequences. Uniform binning cannot give you both. This is the irreconcilable tradeoff that motivates FAST and continuous heads (Chapter 5).
* **Mode collapse at decode time.** Even when the softmax is bimodal, naïve `argmax` or expected-value decoding collapses the modes. Sampling restores them at the cost of variance. Chapter 5 sidesteps the choice — flow matching samples coherent trajectories natively.
* **Compounding covariate shift.** Even with log-loss horizon advantages (§4.3.6), behavior cloning is open-loop with respect to the learner's induced state distribution. Chunking + temporal ensembling mitigates; only *online refinement* fixes. That's Chapter 7.
* **Causal confusion.** With a high-capacity multimodal backbone, the policy can latch onto spurious correlations — *consequences* of the expert action that happen to be visible in the observation, rather than *causes*. (Classic example: the brake-light indicator. Robotic example: the gripper's own visual silhouette in a wrist camera.) Information-bottleneck regularization and observation ablation help. Chapter 6's staged curriculum adds another layer of regularization.

### 4.7.2 The cost we *did* pay — sequential decode

* The autoregressive head is not free. It decodes `H · D` tokens in sequence; a KV-cache makes each step cheap and chunking amortizes the whole decode over `H` executed steps (§4.5.5), which is why it fits our control budget — but the decode is still inherently serial, and OpenVLA's ~6 Hz on an A100 (running per-step rather than per-chunk) is the cautionary number for anyone who skips the chunking amortization.
* The deeper, unavoidable cost is the discrete bottleneck itself: quantization error and the quantization-versus-sequence-length dilemma (§4.3.2) survive no matter how we decode. Both the serial decode and the discrete bottleneck are exactly what Chapter 5's flow-matching head removes, while keeping the inter-joint coherence the autoregressive head bought us. There is no free lunch; this is the bill.

### 4.7.3 What Chapter 5 fixes

* **Keep:** the multimodal output, the backbone, the data pipeline, the closed-loop harness.
* **Replace:** the discrete categorical bottleneck with a continuous **flow-matching** head that learns a vector field pushing Gaussian noise toward the demonstration manifold in ~10 ODE steps.
* **Gain:** smoothness, sub-bin resolution, true 50 Hz inference, no decode-time mode-collapse trap.
* **Lose:** the ability to reuse the LM head verbatim; we trade the "actions are tokens" pedagogy for a continuous generative model.

### 4.7.4 What this chapter is good for, even given the ceilings

* The categorical head is the right starting point for *every* VLA project for three reasons: (a) it shares an objective with the LM head, so the codepath stays unified; (b) it tolerates quantization-to-INT8 cleanly for edge deployment (Chapter 10); (c) it makes multimodality visible — you can *see* the bimodal softmax and reason about it.
* Forward-references:
  * **Chapter 5:** continuous flow-matching head.
  * **Chapter 6:** the action head is the natural first LoRA target — small, task-specific, swappable.
  * **Chapter 7:** RL refinement pushes the policy past the BC ceiling.
  * **Chapter 10:** categorical heads quantize to INT8 cleanly; we revisit deployment.

**Exercise 4.6 — Stretch.** Replace uniform binning with k-means (K=256) fit on the training data. Compare reconstruction error and closed-loop success rate against the uniform baseline. Discuss when data-adaptive binning helps and when it overfits. Solution in Appendix.

**Transition:** "Here's the recap."

---

## Section 4.8: Summary

Bulleted, claim-style sentences per `book_learnings.md` §1.5. Each bullet is one complete sentence stating a takeaway. The final bullet forward-references Chapter 5.

* Behavior cloning is supervised learning on `(observation, action)` pairs; the choice of loss determines whether the policy can represent multimodal expert behavior.
* Mean squared error optimizes the conditional mean. On bimodal demonstrations the conditional mean is the empty space between modes, producing the collision-causing "bisecting path" failure.
* Discretizing each action dimension into `B` bins and training with cross-entropy reframes behavior cloning as classification. The resulting softmax can represent arbitrary multimodal action distributions.
* Action tokenization is the central abstraction: a fixed, invertible map from continuous actions to integer bin ids that lets the Transformer treat motion as another token stream.
* The tokenizer design space spans uniform binning (RT-1), vocabulary reuse from a pretrained language model (RT-2, OpenVLA), data-adaptive codebooks (BeT, VQ-BeT), frequency-space compression (FAST), and Cartesian end-effector spaces for cross-embodiment training.
* Cross-entropy is scale-invariant across bins — it does not penalize being "far off" more than "slightly off" — which is exactly the property that frees the model from mode-averaging.
* The quantization–sequence-length dilemma is the central design tradeoff of discrete behavior cloning: fine bins and short chunks conflict, and FAST exists to attack both at once.
* Under log-loss, behavior cloning enjoys horizon-favorable regret bounds (Foster et al., 2024), and chunking with temporal ensembling further attenuates compounding error.
* A parallel linear head exposes the cost of per-dimension independence: correct marginals, incoherent joint samples. The autoregressive head — reserved-vocab action tokens decoded left-to-right through the Chapter 3 unified-embedding backbone with a causal mask and a KV-cache — fixes it, and trained with bf16 AdamW for ~20,000 steps on a consumer GPU produces a working policy on the SO-101 pick-and-place task.
* Autoregressive decoding is the honest "actions are tokens" claim — the same left-to-right, causally-masked next-token prediction the language model already does — and it makes the Chapter 3 causal fusion the correct design rather than an over-restriction.
* The single most important diagnostic is closed-loop success rate; open-loop validation loss can drop while success rate worsens, because compounding error and decode-time sampling do not appear in offline metrics.
* Decode-time sampling matters as much as training: `argmax` and expected-value decoding collapse modes; temperature + top-p preserves them.
* Discrete behavior cloning has four structural ceilings — quantization vs. sequence-length, decode-time mode collapse, compounding covariate shift, causal confusion — which set the agenda for the next several chapters.
* **Chapter 5** dismantles the first two ceilings by replacing the categorical bottleneck with continuous flow matching, producing smoother motion at higher control rates without the quantization tradeoff.

---

## Figure Summary

| Figure | Description | Type |
|---|---|---|
| Figure 4.1 | Book-wide roadmap, Chapter 4 stage highlighted | Roadmap (reused from Ch 1, Ch 4 stage emphasized) |
| Figure 4.2 | The Multimodal Trap — MSE bisecting path vs. categorical bimodal peaks | Concept comparison |
| Figure 4.3 | The Action Tokenization Pipeline — MotionLM-style vocabulary + token sequence (intuition) over the worked numeric (0.347 rad → bin 142 → 0.349 rad) | Intuition + flow with numbers |
| Figure 4.4 | Vocabulary Alignment Triptych — RT-1 / RT-2 / OpenVLA token strips | Comparison stacks |
| Figure 4.5 | Cross-Embodiment via Per-Dataset Normalization — one shared vocabulary, two robots (MotionLM-style layout) | Architecture diagram |
| Figure 4.6 | Action Head Architectures — Parallel / AR (shipped) / Decoupled, with latency annotations | Architecture triptych |
| Figure 4.7 | Entropy Collapse Heatmap — softmax convergence over training steps | Training-dynamics heatmap |
| Figure 4.8 | Bimodal Convergence on PushT — MSE single Gaussian vs. categorical two peaks | Within-dimension payoff |
| Figure 4.9 | Joint Coordination — parallel head off-diagonal leakage vs. autoregressive on-diagonal samples | Across-dimension payoff (Act 3) |

## Listing Summary

| Listing | Title | Section | Purpose |
|---|---|---|---|
| Listing 4.1 | Toy bimodal regression collapse | 4.1.2 | Motivates the chapter — shows MSE failing on a 1-D mixture |
| Listing 4.2 | Re-instantiating the Chapter 3 backbone | 4.2.1 | State-restoration ritual |
| Listing 4.3 | `ActionTokenizer` skeleton | 4.4.1 | Class signature and stubs |
| Listing 4.4 | Encode a single joint | 4.4.2 | Scalar path with the worked numeric example |
| Listing 4.5 | Vectorized encode/decode | 4.4.3 | Production form over `[..., D]` |
| Listing 4.6 | `ActionTokenizer` with LeRobot stats | 4.4.4 | Q01/Q99 statistics integration |
| Listing 4.7 | Vocabulary-reuse tokenizer | 4.4.5 | Reserved-vocab action-token map the AR head consumes |
| Listing 4.8 | `ParallelActionHead` | 4.5.2 | The baseline foil |
| Listing 4.9 | Cross-entropy with adjacent-bin smoothing | 4.5.3 | Ordinal-aware loss variant (shared by both heads) |
| Listing 4.10 | `AutoregressiveActionHead` (teacher-forced) | 4.5.5 | The shipped head — append-and-decode through the pretrained backbone |
| Listing 4.11 | Autoregressive decode (KV-cache + constrained sampling) | 4.5.6 | Inference loop |
| Listing 4.12 | LeRobotDataset loader with `delta_timestamps` | 4.6.1 | Action chunking data pipeline |
| Listing 4.13 | Training loop | 4.6.2 | AdamW + cosine + bf16 + freeze→unfreeze + teacher forcing |
| Listing 4.14 | Closed-loop rollout | 4.6.4 | PickCubeSO100 evaluation harness |

## Table Summary

| Table | Section | Purpose |
|---|---|---|
| Table 4.1 | 4.2.5 | Action tokenizer families — design-space comparison |
| Table 4.2 | 4.5.1 | Discrete action head architectures — parallel (foil) vs. AR (shipped) vs. decoupled |
| Table 4.3 | 4.6.1 | Training recipe — optimizer, LRs, batch sizes, hardware |

## Callout Box Summary

Using the unified taxonomy from `book_learnings.md` §5.

| Callout | Section | Type | Purpose |
|---|---|---|---|
| Why MSE feels right and isn't | 4.1.4 | PITFALL | The reader's instinct is to start with MSE; this names the trap |
| Why cross-entropy is scale-invariant across bins | 4.3.2 | DEEP DIVE | Mathematical "why it works" for the curious reader |
| Class imbalance from teleop data | 4.3.7 | PITFALL | When to switch from CE to focal loss |
| Off-by-one at the upper edge | 4.4.2 | TIP | One-line bug, one-line fix, one-line test |
| LeRobotDataset format | 4.4.4 | DEFINITION | Robotics-format vocabulary for ML readers |
| Backbone corruption from unfrozen training | 4.5.5 | PITFALL | The symptom-and-fix for a common silent failure |
| Hardware expectations | 4.6.2 | NOTE | Calibrates Colab T4 vs. RTX 4090 vs. A100 |
| Expected-value decoding looks free, isn't | 4.6.4 | PITFALL | The decode-time version of §4.1's failure |
| Action chunking | 4.2 | DEFINITION | Term used throughout the rest of the chapter |
| Notation | 4.3.1 | NOTATION | Symbol legend for the math-heavy section |

## Exercise Summary

| Exercise | Section | Difficulty | Skill |
|---|---|---|---|
| Exercise 4.1 — Bin sweep | 4.4.6 | Light | Re-tokenize at `B ∈ {64, 256, 1024}`, report tradeoffs |
| Exercise 4.2 — Loss surgery | 4.5.5 | Light | Swap CE → focal loss, measure rare-bin recall |
| Exercise 4.3 — Multimodal stress test | 4.6.6 | Medium | Train on PushT, visualize bimodal failure on a saddle state |
| Exercise 4.4 — Entropy diagnostic | 4.6.6 | Light | Plot per-timestep entropy, find decision points |
| Exercise 4.5 — Vocab alignment | 4.6.6 | Medium | Implement RT-2's "overwrite least-used tokens" trick |
| Exercise 4.6 — k-means binning (stretch) | 4.7.4 | Stretch | Data-adaptive binning vs. uniform — when does it help? |

All exercises are placed inline at natural pause points (per `book_learnings.md` §2.6), not collected at chapter end. Solutions live in Appendix.

---

## Cross-Chapter Connections

* **From Chapter 3 →.** Reuse the backbone built in Ch 3. Its pretrained backbone is causal end-to-end; the autoregressive head makes that mask the *correct* design (action tokens attend causally to the perception prefix and to earlier action tokens), so no Chapter 3 rework is needed. The earlier suggestion to relax Ch 3 to a bidirectional/prefix-LM mask is withdrawn — it was premised on the parallel head. Chapter 4 appends reserved-vocab action tokens to the fused sequence and decodes through that same transformer.
* **To Chapter 5.** End on PushT *representing* multimodality, then losing it at decode time. Ch 5 keeps the multimodal output, drops the discrete bottleneck via continuous flow matching.
* **To Chapter 6 (LoRA).** The action head is the natural first LoRA target — small, task-specific, swappable across tasks while keeping the backbone shared.
* **To Chapter 7 (RL).** The compounding-error ceiling (§4.7.1) motivates online refinement with REINFORCE / PPO / GRPO.
* **To Chapter 10 (Deployment).** Categorical heads quantize to INT8 cleanly because softmax outputs are bounded; we revisit this as a deployment win.

## Honest-Scoping Disclaimers (≥2× per chapter, per `book_learnings.md` §1.4)

* §4.1 opener: "*From scratch* in this chapter means the tokenizer and the head — we reuse Chapter 3's backbone, not Llama-2 pretraining."
* §4.2.5: "We implement the RT-1 / OpenVLA family in detail. The other rows of Table 4.1 are surveyed so you know the design space — full implementations live in the cited papers."
* §4.5.1: "We build the parallel linear head as a baseline and the autoregressive head as the shipped policy. The decoupled head appears in the comparison table; its implementation is referenced, not reproduced."
* §4.6: "Closed-loop SR sits at 60–80% on PushT — Diffusion Policy reaches ~95%, motivating Chapter 5."

## "Further Reading" list (chapter-end, ≈10–12 bullets)

Per `book_learnings.md` §7, a short list of papers and repos we want the reader to know about but didn't cite inline:

* RT-1 — Brohan et al., 2022, [arXiv:2212.06817](https://arxiv.org/abs/2212.06817)
* RT-2 — Brohan et al., 2023, [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
* OpenVLA — Kim, Pertsch, Karamcheti et al., 2024, [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
* MotionLM — Seff et al., ICCV 2023, [arXiv:2309.16534](https://arxiv.org/abs/2309.16534) (autoregressive discrete motion-token modeling; source of the Figure 4.3/4.5 shared-vocabulary visual idiom)
* BeT — Shafiullah et al., NeurIPS 2022, [arXiv:2206.11251](https://arxiv.org/abs/2206.11251)
* VQ-BeT — Lee et al., ICML 2024, [arXiv:2403.03181](https://arxiv.org/abs/2403.03181)
* FAST — Pertsch et al., 2025, [arXiv:2501.09747](https://arxiv.org/abs/2501.09747)
* ACT — Zhao et al., RSS 2023, [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
* Diffusion Policy — Chi et al., RSS 2023, [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
* Open X-Embodiment — Padalkar et al., 2023, [arXiv:2310.08864](https://arxiv.org/abs/2310.08864)
* Foster et al., *Is Behavior Cloning All You Need?*, NeurIPS 2024, [arXiv:2407.15007](https://arxiv.org/abs/2407.15007)
* Fourier Head — Gillman et al., NeurIPS 2024, [arXiv:2410.22269](https://arxiv.org/abs/2410.22269)
* LeRobot — [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
* OpenVLA — [github.com/openvla/openvla](https://github.com/openvla/openvla) (see `prismatic/vla/action_tokenizer.py`)

---

## Manning Style Reminders for the Drafter

From `writing_instructions/writing_instructions.md`:

* Listings use **7pt Roboto Mono**, captions in **Heading 4** style. Each `#A`/`#B` annotation gets its own line in **Arial**.
* Annotated code lines must be ≤ **55 characters** wide.
* Figures inserted **In line** (never "Move with text"). Captions in **Heading 6** style.
* Table captions in **Heading 5** style, placed **above** the table.
* Bullet lists use bullet points (•), not dashes.
* Inline equations typed directly (italic), not via the equation editor; block equations use the equation editor (Times New Roman, 8pt).
* No tabs inside headings.

## MQR Reminders for the Drafter

From `mqr/mqr.md`:

* Reader has **basic** PyTorch and DL; assume `nn.Module`, `forward`, `loss.backward()`, `optimizer.step()` are familiar. Do *not* assume familiarity with `nn.functional.cross_entropy`'s broadcasting rules — show shapes.
* Reader has **no robotics background**. Every robotics term (joint, end-effector, gripper, proprioception, DoF, teleoperation) gets a short DEFINITION callout the first time it appears.
* Reader has **a single consumer GPU or Colab**. All listings must run in <24 GB VRAM. Hardware-dependent numbers (training time, FPS) must be annotated with the hardware.
* Reader is here to **build**, not to read a literature survey. Survey content goes in tables and Further Reading; the main flow stays implementation-first.

---

## Estimated Length and Word Count

* **Estimated length:** 28–32 pages (Manning Executive format).
* **Estimated word count:** 11,000–13,000 words.
* **Figure count:** 9 figures + 4 tables (above the typical Ch 1 figure count, reflecting the coding-heavy archetype's `~1 figure per 2–3 pages` density target from `book_learnings.md` §6.1). The added figure (4.9, joint coordination) is the Act-3 autoregressive payoff; the added table (4.4, decode-time sampling) was already in the manuscript.
* **Listing count:** 14 listings (8–30 lines each, per Raschka conventions; longer scripts in the repo with link). The autoregressive pivot adds the AR head (4.10) and the KV-cached decode loop (4.11).
* **Callout count:** 10 callouts spanning PITFALL / TIP / DEFINITION / NOTE / NOTATION / DEEP DIVE.
* **Exercise count:** 6 inline exercises with solutions in Appendix.
