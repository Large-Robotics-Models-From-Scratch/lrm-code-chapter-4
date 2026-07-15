"""Tests for closed-loop rollout evaluation (PR 8).

Two surfaces. ``evaluate`` (Listing 4.14) is exercised against a
*scripted* fake env and a fake policy so the rollout loop -- seed
handling, per-episode ``policy.reset()``, success collection, the
success-rate average -- is verified without a simulator or network.
``maniskill_obs_adapter`` is unit-tested against a synthetic ManiSkill
observation dict (the real ``sensor_data``/``agent`` structure recorded
in Task 0) so the ManiSkill -> ``encode_prefix`` batch glue is checked
without instantiating SAPIEN. The one genuinely end-to-end check (a
real ``gym.make`` + real policy) lives behind ``@pytest.mark.integration``
because it needs a Vulkan-capable host.
"""

import numpy as np
import pytest
import torch

from ch04.rollout import (
    coerce_success,
    evaluate,
    make_maniskill_obs_adapter,
    maniskill_obs_adapter,
)

# -- coerce_success --------------------------------------------------


def test_coerce_success_from_python_bool():
    assert coerce_success(True) is True
    assert coerce_success(False) is False


def test_coerce_success_from_scalar_tensor():
    assert coerce_success(torch.tensor(True)) is True
    assert coerce_success(torch.tensor(0)) is False


def test_coerce_success_from_batched_tensor():
    # ManiSkill returns a batched [N] success tensor (N=1 per env).
    assert coerce_success(torch.tensor([True])) is True
    assert coerce_success(torch.tensor([0])) is False


def test_coerce_success_from_numpy():
    assert coerce_success(np.array([1.0])) is True
    assert coerce_success(np.array(False)) is False


# -- scripted fake env + policy --------------------------------------


class _ScriptedEnv:
    """Deterministic env: each episode ends after ``ep_len`` steps and
    reports a pre-scripted success flag for that seed.

    ``successes`` is indexed by the seed passed to ``reset``. Success is
    returned as a batched ``torch`` tensor to mirror ManiSkill and
    exercise ``coerce_success``.
    """

    def __init__(self, successes, ep_len=3):
        self.successes = successes
        self.ep_len = ep_len
        self._t = 0
        self._seed = 0

    def reset(self, seed=0):
        self._t = 0
        self._seed = seed
        return {"obs": seed}, {}

    def step(self, action):
        self._t += 1
        term = self._t >= self.ep_len
        trunc = False
        info = {"success": torch.tensor([self.successes[self._seed]])}
        return {"obs": self._seed}, 0.0, term, trunc, info

    def close(self):
        pass


class _FakePolicy:
    """Counts ``reset`` calls and emits a fixed action each step."""

    def __init__(self, action_dim=6):
        self.action_dim = action_dim
        self.reset_calls = 0
        self.select_calls = 0

    def reset(self):
        self.reset_calls += 1

    def select_action(self, obs):
        self.select_calls += 1
        return np.zeros(self.action_dim, dtype=np.float64)


def test_evaluate_returns_exact_rate_and_per_seed():
    # 3 of 4 seeds succeed -> 0.75.
    successes = [True, False, True, True]
    env = _ScriptedEnv(successes)
    policy = _FakePolicy()
    rate, per_seed = evaluate(policy, n_seeds=4, env=env)
    assert rate == pytest.approx(0.75)
    assert per_seed == successes
    assert all(isinstance(b, bool) for b in per_seed)


def test_evaluate_calls_policy_reset_once_per_seed():
    env = _ScriptedEnv([True, True])
    policy = _FakePolicy()
    evaluate(policy, n_seeds=2, env=env)
    assert policy.reset_calls == 2


def test_evaluate_applies_obs_adapter():
    seen = []

    def adapter(obs):
        seen.append(obs)
        return obs

    env = _ScriptedEnv([True])
    policy = _FakePolicy()
    evaluate(policy, n_seeds=1, env=env, obs_adapter=adapter)
    # adapter ran on the reset obs and each stepped obs.
    assert len(seen) >= 1


def test_evaluate_respects_max_steps():
    # Env never terminates on its own; max_steps must truncate.
    class _Never(_ScriptedEnv):
        def step(self, action):
            info = {"success": torch.tensor([False])}
            return {"obs": 0}, 0.0, False, False, info

    env = _Never([False])
    policy = _FakePolicy()
    rate, per_seed = evaluate(
        policy, n_seeds=1, env=env, max_steps=5
    )
    assert policy.select_calls == 5
    assert per_seed == [False]


# -- maniskill_obs_adapter (synthetic obs) ---------------------------


def _synthetic_maniskill_obs(h=128, w=128, cam="base_camera"):
    """The recorded PickCubeSO100 rgb obs structure (Task 0)."""
    rgb = torch.randint(
        0, 256, (1, h, w, 3), dtype=torch.uint8
    )
    return {
        "agent": {
            "qpos": torch.zeros(1, 6),
            "qvel": torch.zeros(1, 6),
        },
        "extra": {},
        "sensor_param": {cam: {}},
        "sensor_data": {cam: {"rgb": rgb}},
    }


def test_obs_adapter_produces_encode_prefix_batch():
    obs = _synthetic_maniskill_obs()
    batch = maniskill_obs_adapter(obs, "pick up the cube")
    for key in (
        "observation.images.up",
        "observation.images.side",
        "observation.state",
        "task",
    ):
        assert key in batch
    up = batch["observation.images.up"]
    side = batch["observation.images.side"]
    assert up.shape == (1, 3, 128, 128)
    assert side.shape == (1, 3, 128, 128)
    assert up.dtype == torch.float32
    # values normalized into [0, 1].
    assert float(up.min()) >= 0.0 and float(up.max()) <= 1.0
    assert batch["observation.state"].shape == (1, 6)
    assert batch["task"] == ["pick up the cube"]


def test_obs_adapter_duplicates_single_camera():
    # PickCubeSO100 has one camera; both slots must be filled.
    obs = _synthetic_maniskill_obs()
    batch = maniskill_obs_adapter(obs, "task")
    assert torch.equal(
        batch["observation.images.up"],
        batch["observation.images.side"],
    )


def test_obs_adapter_handles_alternate_camera_name():
    obs = _synthetic_maniskill_obs(cam="hand_camera")
    batch = maniskill_obs_adapter(obs, "task")
    assert batch["observation.images.up"].shape == (1, 3, 128, 128)


def test_make_maniskill_obs_adapter_is_callable_factory():
    adapter = make_maniskill_obs_adapter("grab it")
    obs = _synthetic_maniskill_obs()
    batch = adapter(obs)
    assert batch["task"] == ["grab it"]


# -- integration (needs a Vulkan-capable host) -----------------------


@pytest.mark.integration
def test_maniskill_end_to_end_smoke():
    """Real ``gym.make`` + adapter + a short ``evaluate`` run.

    Untrained policy, so success is ~0 -- the point is the glue runs
    end to end and returns a bool. Skipped automatically if SAPIEN
    cannot instantiate the env (e.g. no Vulkan driver, as on macOS).
    """
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    import mani_skill.envs  # noqa: F401

    try:
        env = gym.make("PickCubeSO100-v1", obs_mode="rgb")
    except Exception as exc:  # pragma: no cover - host dependent
        pytest.skip(f"cannot instantiate ManiSkill env: {exc}")

    from ch03 import UnifiedEmbeddingBackbone

    from ch04.action_tokenizer import ActionTokenizer
    from ch04.autoregressive_action_head import (
        AutoregressiveActionHead,
    )
    from ch04.fusion_adapter import FusionAdapter
    from ch04.policy import DiscretePolicy

    backbone = UnifiedEmbeddingBackbone().float()
    fusion = FusionAdapter(backbone)
    head = AutoregressiveActionHead(fusion)
    lo = -np.ones(6)
    hi = np.ones(6)
    tok = ActionTokenizer(lo, hi)
    policy = DiscretePolicy(fusion, head, tok)

    # Verify the adapter maps a real reset obs into a valid batch.
    obs, _ = env.reset(seed=0)
    batch = maniskill_obs_adapter(obs, "pick up the cube")
    prefix = fusion.encode_prefix(batch)
    assert prefix.shape[0] == 1
    assert batch["observation.state"].shape == (1, 6)

    adapter = make_maniskill_obs_adapter("pick up the cube")
    rate, per_seed = evaluate(
        policy, env=env, n_seeds=1, obs_adapter=adapter,
        max_steps=5,
    )
    assert isinstance(rate, float)
    assert len(per_seed) == 1
    assert isinstance(per_seed[0], bool)
    env.close()
