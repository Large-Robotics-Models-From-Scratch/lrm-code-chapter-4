"""Discrete policy + decode-time sampling -- the deployment surface.

Training was a single parallel forward: the whole target chunk went in
teacher-forced, one causal pass produced every position's logits at
once (SS4.5.4). Inference cannot cheat that way -- each action token is
sampled and only then can it inform the next -- so decode is a *serial*
loop over the ``H * D`` tokens of a chunk. The trick that keeps every
step cheap is the KV cache: the fused observation prefix and every
already-sampled token are attended to through cached keys/values, so
step ``t`` runs the language model on a single new position instead of
re-reading the whole prefix (SS4.5.5). We decode one full chunk, buffer
its ``H`` actions, and replay them open-loop -- re-planning only every
``H`` control steps, not every step.

``sample_bin`` is the decode-time knob (Table 4.4, SS4.6.4): four ways
to turn a bin's logits into a chosen bin (or a continuous action, for
the expected-value strategy). ``DiscretePolicy`` wraps the fusion
adapter, autoregressive head, and action tokenizer behind the
manuscript's ``reset()`` / ``select_action(obs)`` interface
(Listing 4.14).
"""

from __future__ import annotations

import numpy as np
import torch

_STRATEGIES = (
    "argmax",
    "temperature",
    "temperature_top_p",
    "expected_value",
)


def sample_bin(
    logits: torch.Tensor,
    strategy: str = "temperature_top_p",
    temperature: float = 1.0,
    top_p: float = 0.95,
    centers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Turn per-bin logits into a chosen bin (Table 4.4, SS4.6.4).

    ``logits`` is ``[B, n_bins]``. The return type *depends on the
    strategy* and the caller must handle both:

    * ``"argmax"`` / ``"temperature"`` / ``"temperature_top_p"`` return
      ``[B]`` int64 bin ids.
    * ``"expected_value"`` returns ``[B]`` float actions (the softmax
      mean of the bin centers), so there is no bin id to feed back --
      the policy rounds it to the nearest bin for the token it appends.

    Strategies:

    * ``argmax`` -- the greedy bin, ``argmax_b``. Deterministic.
    * ``temperature`` -- ``softmax(logits / T)`` then a multinomial
      draw. ``T`` is applied *before* the softmax; ``T -> 0`` sharpens
      toward the argmax and we fall back to a plain argmax at/near zero
      to avoid a divide-by-zero.
    * ``temperature_top_p`` -- temperature, then nucleus filtering: sort
      descending, keep the smallest prefix whose cumulative mass is
      ``>= top_p``, renormalize that set, sample. The default policy
      strategy (``T = 1.0``, ``p = 0.95``).
    * ``expected_value`` -- ``sum_b softmax(logits)_b * centers_b`` (no
      temperature), the distribution's mean along this dimension's bin
      centers. PITFALL: on a *bimodal* posterior (e.g. "go left or go
      right") the mean lands in the empty valley between the modes -- a
      command the policy never intended. Safe only when the per-bin
      distribution is unimodal; prefer a sampling strategy otherwise.

    ``centers`` is the ``[n_bins]`` bin-center vector for the *current*
    action dimension and is required only by ``expected_value``.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; "
            f"expected one of {_STRATEGIES}"
        )
    logits = logits.float()

    if strategy == "argmax":
        return torch.argmax(logits, dim=-1).to(torch.int64)

    if strategy == "expected_value":
        if centers is None:
            raise ValueError(
                "strategy 'expected_value' needs the current "
                "dimension's bin centers [n_bins]"
            )
        probs = torch.softmax(logits, dim=-1)
        centers = centers.to(probs.dtype).reshape(1, -1)
        return (probs * centers).sum(dim=-1)

    # temperature and temperature_top_p share the tempered softmax.
    if temperature <= 1e-6:
        # T -> 0 is the argmax limit; short-circuit to avoid div-by-0.
        return torch.argmax(logits, dim=-1).to(torch.int64)
    probs = torch.softmax(logits / temperature, dim=-1)

    if strategy == "temperature_top_p":
        probs = _nucleus(probs, top_p)

    return torch.multinomial(probs, num_samples=1).squeeze(-1).to(
        torch.int64
    )


def _nucleus(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out the tail outside the smallest ``>= top_p`` mass set.

    Sort each row descending, find the smallest prefix whose cumulative
    mass reaches ``top_p``, keep it (always at least the top bin), zero
    the rest, and renormalize so the kept set sums to 1.
    """
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Keep every bin up to and including the one that first crosses
    # top_p: shift the "already >= top_p" flag right so the crossing bin
    # itself stays. The top bin is always kept (column 0 forced False).
    remove_sorted = cumsum - sorted_probs >= top_p
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(-1, sorted_idx, remove_sorted)
    kept = probs.masked_fill(remove, 0.0)
    return kept / kept.sum(dim=-1, keepdim=True)


class DiscretePolicy:
    """Autoregressive discrete policy for deployment (Listing 4.14).

    Wraps the fusion adapter, the autoregressive action head, and the
    action tokenizer behind the manuscript's control interface:
    ``reset()`` before an episode, then ``select_action(obs)`` once per
    control step. Actions are decoded a full chunk at a time and
    replayed open-loop for ``H`` steps (SS4.5.5), so the expensive
    KV-cached decode fires only every ``H`` calls.

    ``obs`` is the batch dict ``encode_prefix`` expects (single env,
    ``B = 1``): ``observation.images.up`` / ``...side`` ``[1, 3, H, W]``
    in ``[0, 1]``, ``observation.state`` ``[1, 6]``, and ``task``
    ``list[str]`` of length 1. Task 9 supplies the ManiSkill ->
    batch-dict adapter; here the caller passes the dict directly (or a
    state-only ``[6]`` / ``[1, 6]`` array is accepted for convenience).
    """

    def __init__(
        self,
        fusion,
        head,
        tokenizer,
        chunk_h: int = 16,
        action_dim: int = 6,
        temperature: float = 1.0,
        top_p: float = 0.95,
        strategy: str = "temperature_top_p",
        device: str = "cpu",
    ) -> None:
        self.fusion = fusion
        self.head = head
        self.tokenizer = tokenizer
        self.chunk_h = chunk_h
        self.action_dim = action_dim
        self.temperature = temperature
        self.top_p = top_p
        self.strategy = strategy
        self.device = device
        # Per-dim bin centers as a tensor for expected_value; the
        # tokenizer stays pure NumPy (edge-CPU contract), so convert
        # once here rather than importing torch into it.
        self._centers = torch.as_tensor(
            np.asarray(tokenizer.centers), dtype=torch.float32,
            device=device,
        )
        self._buffer: list[np.ndarray] = []

    def reset(self) -> None:
        """Clear the chunk buffer (and any cached decode state).

        Call at episode start: the next ``select_action`` re-plans from
        the fresh observation instead of replaying a stale chunk. The KV
        cache is decode-local (built and dropped inside ``_decode_chunk``
        each chunk), so there is nothing else to clear.
        """
        self._buffer = []

    def select_action(self, obs) -> np.ndarray:
        """Return one ``[action_dim]`` action (Listing 4.14).

        When the buffer is empty, decode a fresh chunk (one KV-cached
        pass over ``H * D`` tokens) and fill the buffer with its ``H``
        actions; otherwise replay. Always pop and return the next
        buffered action -- open-loop within the chunk (SS4.5.5).
        """
        if not self._buffer:
            chunk = self._decode_chunk(obs)  # [H, D] float actions
            self._buffer = [row for row in chunk]
        action = self._buffer.pop(0)
        return np.asarray(action, dtype=np.float64)

    @torch.no_grad()
    def _decode_chunk(self, obs) -> np.ndarray:
        """KV-cached serial decode of one chunk -> ``[H, D]`` actions.

        Seed the causal stream with the BOS embedding after the fused
        prefix (one ``use_cache=True`` forward), then step ``H * D``
        times: read out the last hidden -> per-bin logits -> sample a
        bin -> append that token's embedding and advance the cache by
        one position. Decode order matches training's teacher-forcing
        flatten (token index ``= h * D + d``, ``chunk_data`` reshapes
        ``[B, H, D] -> [B, H*D]`` row-major), so the collected ids
        reshape back to ``[H, D]`` and ``tokenizer.decode`` returns the
        continuous chunk.

        Note: the manuscript's non-action-vocab masking (SS4.5.5) is
        moot here -- ``head.readout`` is an ``n_bins``-way linear, so it
        can only ever emit a valid bin; there is no text/image id to
        mask out. (manuscript_fixes item.)
        """
        fusion = self.fusion
        head = self.head
        horizon = self.chunk_h * self.action_dim

        batch = self._obs_to_batch(obs)
        prefix = fusion.encode_prefix(batch)
        # Seed: BOS embed right after the prefix, prime the cache.
        bos = torch.full(
            (1, 1), head.bos_id, dtype=torch.long, device=prefix.device
        )
        step_emb = fusion.embed(bos).to(prefix.dtype)
        seq = torch.cat([prefix, step_emb], dim=1)
        hidden, past = fusion.forward(seq, use_cache=True)

        bin_ids: list[int] = []
        actions: list[float] = []
        for step in range(horizon):
            dim = step % self.action_dim
            logits = head.readout(hidden[:, -1, :])  # [1, n_bins]
            if self.strategy == "expected_value":
                value = sample_bin(
                    logits, strategy="expected_value",
                    centers=self._centers[dim],
                )
                actions.append(float(value.item()))
                # Feed back the nearest bin's token: the model was
                # trained on discrete ids, so round the continuous mean
                # onto the grid for the next step's context.
                bin_id = self._nearest_bin(float(value.item()), dim)
            else:
                bin_id = int(
                    sample_bin(
                        logits, strategy=self.strategy,
                        temperature=self.temperature, top_p=self.top_p,
                    ).item()
                )
            bin_ids.append(bin_id)
            if step == horizon - 1:
                break
            tok = torch.full(
                (1, 1), head.act_token_base + bin_id,
                dtype=torch.long, device=prefix.device,
            )
            step_emb = fusion.embed(tok).to(prefix.dtype)
            hidden, past = fusion.forward(
                step_emb, past_key_values=past, use_cache=True
            )

        bins = np.array(bin_ids, dtype=np.int64).reshape(
            self.chunk_h, self.action_dim
        )
        if self.strategy == "expected_value":
            # Continuous means already computed per position; reshape
            # them to [H, D] instead of decoding bin centers.
            return np.array(actions, dtype=np.float64).reshape(
                self.chunk_h, self.action_dim
            )
        return np.asarray(self.tokenizer.decode(bins), dtype=np.float64)

    def _nearest_bin(self, value: float, dim: int) -> int:
        """Snap a continuous value to its nearest bin id for ``dim``."""
        centers = self._centers[dim]
        idx = int(torch.argmin(torch.abs(centers - value)).item())
        return idx

    def _obs_to_batch(self, obs) -> dict:
        """Coerce ``obs`` into the batch dict ``encode_prefix`` wants.

        Accepts the full batch dict as-is (the common path; Task 9's
        ManiSkill adapter produces exactly this). No transformation is
        applied beyond validating the required keys are present.
        """
        if isinstance(obs, dict):
            required = (
                "observation.images.up",
                "observation.images.side",
                "observation.state",
                "task",
            )
            missing = [k for k in required if k not in obs]
            if missing:
                raise ValueError(
                    f"obs dict missing keys {missing}; expected "
                    f"{required} (single-env batch, B=1)"
                )
            return obs
        raise TypeError(
            "obs must be the batch dict encode_prefix expects "
            "(observation.images.up/side, observation.state, task); "
            "Task 9 supplies the ManiSkill adapter"
        )
