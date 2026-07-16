"""Evaluation: open-loop val loss (primary) + sim rollout (optional).

Chapter 4 evaluates the discrete BC policy **open-loop** as its primary,
honest metric, and treats the closed-loop ManiSkill rollout as an
optional, domain-caveated extra. The reason is a train/eval domain
mismatch, documented in full in ``docs/EVAL_INCONSISTENCY.md``: the
policy is trained on the *real* ``svla_so101_pickplace`` dataset (SO-101,
two cameras, absolute joint-target actions) but ``PickCubeSO100-v1`` is a
different domain (SO-100, one camera, a delta-action controller). The
absolute-vs-delta action-space gap alone breaks the rollout, so a low
sim success rate would reflect that gap, not the discrete-BC method the
chapter teaches. Until a matched-domain sim exists (a ch2/ch3 decision),
open-loop is how we report results here.

``evaluate_open_loop`` (primary) scores the head's teacher-forced,
pad-masked cross-entropy on a **held-out** loader with no gradient. It
mirrors the training loss exactly, but on data the policy never trained
on. It is an *optimistic* proxy -- open-loop feeds the policy
ground-truth states, so it never sees its own compounding errors (that
is exactly what closed-loop would test) -- but it is a real, reproducible
number, unlike the mismatched sim. Pair it with the trace figure
(``diagnostics.plot_action_traces``) and the bimodal/coordination
diagnostics (Figures 4.8-4.10) for the full open-loop story.

``evaluate`` (optional) runs the manuscript's closed-loop rollout loop --
reset per seed, ``policy.reset()``, step until termination or
truncation, collect ``info["success"]`` -- and returns the success rate
plus the per-seed flags. It is kept for a future matched-domain sim, not
as a headline metric. ``maniskill_obs_adapter`` is the bridge from
ManiSkill's rgb observation dict to the batch dict
``FusionAdapter.encode_prefix`` wants.

The obs mapping (verified against ManiSkill 3.0.1 `PickCubeSO100-v1`):

* ManiSkill rgb obs is
  ``{"agent": {"qpos": [N, 6], "qvel": ...}, "extra": {...},
  "sensor_param": {...}, "sensor_data": {"base_camera":
  {"rgb": [N, H, W, 3] uint8}}}``.
* PickCubeSO100 exposes a **single** camera (``base_camera``, 128x128),
  but the Chapter 3 backbone (and this dataset's SO-101 recording) use
  **two** (``up`` + ``side``). The adapter fills both slots from the one
  available camera -- the honest mapping given the env provides one
  view. When a future env exposes named ``up``/``side`` cameras, extend
  ``_pick_cameras`` to select them by name.
* ManiSkill returns HWC uint8 in ``[0, 255]``; ``encode_prefix`` (via
  ch3's ``preprocess_image``) wants CHW float in ``[0, 1]``, so the
  adapter permutes and divides by 255.
* ``observation.state`` is the agent's ``qpos`` ``[N, 6]`` (5 arm
  joints + gripper), cast to float.
* PickCube carries no language instruction, so ``task`` is supplied by
  the caller (the SO-101 dataset's shared instruction string).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def evaluate_open_loop(
    head,
    fusion,
    tokenizer,
    loader,
    n_batches=None,
    device="cpu",
):
    """Held-out open-loop validation loss -- the primary Ch 4 metric.

    Iterate the (held-out) ``loader`` and score the autoregressive
    head's teacher-forced, pad-masked cross-entropy on data it never
    trained on. This is the same loss ``train`` optimizes, computed here
    with no gradient in eval mode, so it is directly comparable to the
    training curve -- just honest, because the batches are held out.

    Open-loop is an *optimistic* proxy for closed-loop success: it feeds
    the policy ground-truth states at every step, so the policy never
    sees the compounding errors a real rollout would surface. But it is a
    real, reproducible number, unlike the domain-mismatched
    ``PickCubeSO100-v1`` sim (see the module docstring and
    ``docs/EVAL_INCONSISTENCY.md``), which is why the chapter reports it
    as primary.

    Per batch: squeeze the ``[B, 1, 6]`` state to ``[B, 6]``, encode the
    fused prefix, tokenize ``action`` (``[B, H, D]``) to bins and flatten
    to ``[B, H * D]``, expand the ``[B, H]`` ``action_is_pad`` frame mask
    to token level (``repeat_interleave`` by ``D``), and run the head's
    masked CE path (``logits`` + ``cross_entropy(reduction="none")`` with
    the float32 upcast ``forward`` uses). Losses accumulate
    **token-weighted**: fully-padded batches add 0 to both the numerator
    and the token count, so they leave ``val_loss`` unchanged.

    ``n_batches`` (optional) caps how many batches are read, for a quick
    estimate on a large held-out split.

    Returns ``{"val_loss": float, "per_dim_loss": list[float],
    "n_tokens": int}`` -- the scalar token-weighted mean CE, the mean CE
    for each of the ``D`` action dimensions (token index ``% D`` selects
    the dimension), and the number of unmasked action tokens scored.
    """
    head.eval()
    fusion.eval()
    loss_sum = 0.0
    n_tokens = 0
    action_dim: int | None = None
    per_dim_sum: np.ndarray | None = None
    per_dim_count: np.ndarray | None = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if n_batches is not None and i >= n_batches:
                break
            state = batch["observation.state"]
            model_batch = dict(batch)
            model_batch["observation.state"] = state.squeeze(1).to(
                device
            )
            for key in (
                "observation.images.up",
                "observation.images.side",
            ):
                model_batch[key] = batch[key].to(device)

            action = batch["action"]  # [B, H, D]
            batch_size, horizon, dim = action.shape
            if action_dim is None:
                action_dim = dim
                per_dim_sum = np.zeros(dim, dtype=np.float64)
                per_dim_count = np.zeros(dim, dtype=np.int64)

            bins = torch.from_numpy(
                tokenizer.encode(action.numpy())
            ).long().to(device)
            target = bins.reshape(batch_size, -1)  # [B, H * D]

            pad = batch.get("action_is_pad")
            if pad is not None:
                pad_mask = pad.to(device).repeat_interleave(
                    dim, dim=1
                )  # [B, H * D]
            else:
                pad_mask = torch.zeros_like(target, dtype=torch.bool)

            prefix = fusion.encode_prefix(model_batch)
            logits = head.logits(prefix, target)  # [B, T, n_bins]
            # Mirror head.forward's masked CE exactly: float32 upcast
            # (bf16 softmax is too coarse at the ~log(256) scale), then
            # token-level cross-entropy with the pad positions zeroed.
            flat_logits = logits.reshape(-1, head.n_bins).float()
            flat_targets = target.reshape(-1)
            ce = F.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            )
            keep = (~pad_mask.reshape(-1)).to(ce.dtype)
            weighted = (ce * keep).cpu().numpy()
            keep_np = keep.cpu().numpy()

            loss_sum += float(weighted.sum())
            n_tokens += int(keep_np.sum())

            # token index t = h * D + d, so d = t % D selects the joint.
            dim_ids = np.tile(
                np.arange(horizon * dim) % dim, batch_size
            )
            for d in range(dim):
                sel = dim_ids == d
                per_dim_sum[d] += weighted[sel].sum()
                per_dim_count[d] += int(keep_np[sel].sum())

    if action_dim is None:  # empty loader
        return {"val_loss": 0.0, "per_dim_loss": [], "n_tokens": 0}

    val_loss = loss_sum / max(n_tokens, 1)
    per_dim_loss = [
        float(per_dim_sum[d] / max(int(per_dim_count[d]), 1))
        for d in range(action_dim)
    ]
    return {
        "val_loss": val_loss,
        "per_dim_loss": per_dim_loss,
        "n_tokens": n_tokens,
    }


def coerce_success(value) -> bool:
    """Coerce ManiSkill's ``info["success"]`` to a Python ``bool``.

    ManiSkill runs batched under the hood, so ``info["success"]`` for a
    single env is usually a length-1 ``torch`` tensor (``tensor([True])``),
    but it can also arrive as a NumPy array, a 0-d tensor, or a plain
    bool depending on wrappers. Reduce whatever comes in to one bool:
    tensors/arrays are flattened and treated as truthy if *any* element
    is set (a single-env batch has exactly one element anyway).
    """
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1).any().item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1).any())
    if isinstance(value, (list, tuple)):
        return bool(any(value))
    return bool(value)


def maniskill_obs_adapter(obs: dict, instruction: str) -> dict:
    """Map a ManiSkill rgb obs into the ``encode_prefix`` batch dict.

    Returns ``{"observation.images.up": [1, 3, H, W],
    "observation.images.side": [1, 3, H, W], "observation.state":
    [1, 6], "task": [instruction]}`` with images as float in ``[0, 1]``.
    See the module docstring for the full field-by-field mapping and why
    the single ManiSkill camera fills both image slots.
    """
    camera = _pick_camera(obs)
    image = _rgb_to_chw01(camera)  # [1, 3, H, W] float32
    state = _extract_state(obs)  # [1, 6] float32
    return {
        # One camera view fills both slots (env exposes one camera).
        "observation.images.up": image,
        "observation.images.side": image,
        "observation.state": state,
        "task": [instruction],
    }


def make_maniskill_obs_adapter(instruction: str):
    """Return an ``obs -> batch`` adapter bound to one instruction.

    ``evaluate`` calls ``obs_adapter(obs)`` with a single argument, so
    this factory closes over the (constant) task string. Pass the result
    as ``evaluate(..., obs_adapter=make_maniskill_obs_adapter(task))``.
    """

    def adapter(obs: dict) -> dict:
        return maniskill_obs_adapter(obs, instruction)

    return adapter


def _pick_camera(obs: dict):
    """Return the first camera's rgb tensor from a ManiSkill obs.

    Prefers ``base_camera`` (PickCubeSO100's sensor), then falls back to
    whatever single camera ``sensor_data`` carries, so a differently
    named camera still works.
    """
    sensors = obs.get("sensor_data")
    if not sensors:
        raise KeyError(
            "obs has no 'sensor_data'; is the env in obs_mode='rgb'? "
            f"got top-level keys {sorted(obs.keys())}"
        )
    name = "base_camera" if "base_camera" in sensors else next(
        iter(sensors)
    )
    cam = sensors[name]
    if "rgb" not in cam:
        raise KeyError(
            f"camera {name!r} has no 'rgb' key; got {sorted(cam)}"
        )
    return cam["rgb"]


def _rgb_to_chw01(rgb) -> torch.Tensor:
    """HWC uint8 ``[N, H, W, 3]`` -> CHW float ``[1, 3, H, W]`` in [0,1].

    Accepts a NumPy array or torch tensor, batched (``[N, H, W, 3]``) or
    single (``[H, W, 3]``); keeps only the first env and normalizes.
    """
    t = torch.as_tensor(np.asarray(rgb))
    if t.ndim == 4:
        t = t[0]  # first env -> [H, W, 3]
    if t.ndim != 3 or t.shape[-1] != 3:
        raise ValueError(
            f"expected rgb [.., H, W, 3]; got shape {tuple(t.shape)}"
        )
    chw = t.permute(2, 0, 1).to(torch.float32) / 255.0
    return chw.unsqueeze(0)  # [1, 3, H, W]


def _extract_state(obs: dict) -> torch.Tensor:
    """Agent ``qpos`` -> ``[1, 6]`` float32 state.

    ManiSkill nests proprioception under ``obs["agent"]["qpos"]``
    (``[N, 6]`` for SO100: 5 arm joints + gripper).
    """
    agent = obs.get("agent", {})
    qpos = agent.get("qpos")
    if qpos is None:
        raise KeyError(
            "obs['agent']['qpos'] missing; cannot build state vector"
        )
    t = torch.as_tensor(np.asarray(qpos), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t[:1]  # first env -> [1, 6]


def evaluate(
    policy,
    env_id: str | None = None,
    n_seeds: int = 50,
    obs_adapter=None,
    max_steps: int | None = None,
    env=None,
):
    """Run closed-loop rollouts and return ``(success_rate, per_seed)``.

    Listing 4.14 semantics: build ``gym.make(env_id, obs_mode="rgb")``,
    then for each of ``n_seeds`` seeds reset the env with that seed,
    call ``policy.reset()`` (clearing its chunk buffer), and step
    ``policy.select_action(obs)`` until the episode terminates or
    truncates, collecting ``info["success"]`` (coerced to ``bool``).
    The success rate is the mean of the per-seed flags.

    ``obs_adapter`` (optional) maps the raw ManiSkill obs to whatever
    ``policy.select_action`` expects -- pass
    ``make_maniskill_obs_adapter(instruction)`` for the real env.

    ``env`` (optional) injects a pre-built env, bypassing ``gym.make``;
    it takes precedence over ``env_id`` and is the seam the unit tests
    use to run the loop against a scripted fake without SAPIEN or a
    network. When ``evaluate`` builds the env itself (``env_id`` path),
    it closes it before returning; an injected ``env`` is left open for
    the caller to reuse and close.

    ``max_steps`` (optional) caps the steps per episode -- a safety
    valve for a policy that never triggers termination.
    """
    owns_env = env is None
    if owns_env:
        if env_id is None:
            raise ValueError(
                "provide env_id (or an env=) to evaluate against"
            )
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  (registers the env ids)

        env = gym.make(env_id, obs_mode="rgb")

    per_seed: list[bool] = []
    try:
        for seed in range(n_seeds):
            obs, _ = env.reset(seed=seed)
            policy.reset()
            info: dict = {}
            done = False
            steps = 0
            while not done:
                model_obs = obs_adapter(obs) if obs_adapter else obs
                action = policy.select_action(model_obs)
                obs, _, term, trunc, info = env.step(action)
                steps += 1
                done = bool(term) or bool(trunc)
                if max_steps is not None and steps >= max_steps:
                    break
            per_seed.append(coerce_success(info.get("success", False)))
    finally:
        if owns_env:
            env.close()

    success_rate = (
        float(np.mean(per_seed)) if per_seed else 0.0
    )
    return success_rate, per_seed
