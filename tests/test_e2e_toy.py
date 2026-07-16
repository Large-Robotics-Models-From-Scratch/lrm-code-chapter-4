"""End-to-end tests over a tiny synthetic LeRobot dataset (PR 9b).

These prove that every Chapter 4 module composes on **real
LeRobot-format data** -- not hand-built batch dicts -- without any of the
82 MB Hub download the real ``svla_so101_pickplace`` dataset needs. The
fixture (``tests/toy_dataset.py``) writes a real ``LeRobotDataset`` to a
temp dir with the same schema as the real one, so the pipeline code runs
unchanged.

The fast test (``test_e2e_toy_pipeline_fake_backbone``) wires the toy
dataset through the whole chain against the toy ``FakeBackbone`` -- no
network, no simulator, not marked ``integration`` -- and each assert pins
a real shape/contract at every stage:

    tokenizer -> chunked loader -> fusion adapter -> both action heads
    -> train -> checkpoint round-trip -> policy decode -> open-loop eval
    -> closed-loop rollout loop

Because the ``FakeBackbone`` vocab is only 64 wide, the fast test uses a
small ``n_bins`` / ``act_token_base`` (8 / 56), the same toy config the
head and train unit tests use. The second test repeats the spirit of the
pipeline against the **real** ``ch03.UnifiedEmbeddingBackbone`` at the
chapter's real 256-bin / 48896-base configuration; it is marked
``integration`` because it downloads SmolLM2 + SigLIP once.
"""

import math

import numpy as np
import pytest
import torch
from fakes import FAKE_VOCAB, FAKE_WIDTH, FakeBackbone
from toy_dataset import make_toy_lerobot_dataset

from ch04.action_tokenizer import ActionTokenizer
from ch04.autoregressive_action_head import AutoregressiveActionHead
from ch04.chunk_data import make_chunk_dataset, make_chunk_loader
from ch04.fusion_adapter import FusionAdapter
from ch04.parallel_action_head import ParallelActionHead
from ch04.policy import DiscretePolicy
from ch04.rollout import evaluate, evaluate_open_loop
from ch04.train import TrainConfig, load_checkpoint, train

CHUNK_H = 8
ACTION_DIM = 6
BATCH = 2
# The FakeBackbone vocab is only 64 wide, so the toy config reserves the
# last 8 ids as action bins (base 56), matching the head/train unit tests.
FAKE_BINS = 8
FAKE_BASE = FAKE_VOCAB - FAKE_BINS  # 56


@pytest.fixture(scope="module")
def toy_ds(tmp_path_factory):
    """Build one tiny real LeRobotDataset, shared read-only by tests."""
    root = tmp_path_factory.mktemp("toy_lerobot") / "dataset"
    repo_id, root = make_toy_lerobot_dataset(
        root, n_episodes=3, frames_per_episode=12, fps=30
    )
    return repo_id, root


def _obs_from_batch(batch, index=0):
    """Single-env obs dict for ``encode_prefix`` from a loader batch.

    Cameras come squeezed as ``[B, 3, H, W]`` and state as ``[B, 1, 6]``
    (the video/tabular asymmetry), so slice env ``index`` and rebuild the
    ``B = 1`` batch the policy/adapter expect: images ``[1, 3, H, W]`` and
    state ``[1, 6]``.
    """
    return {
        "observation.images.up": batch["observation.images.up"][
            index : index + 1
        ],
        "observation.images.side": batch["observation.images.side"][
            index : index + 1
        ],
        "observation.state": batch["observation.state"][index],  # [1, 6]
        "task": [batch["task"][index]],
    }


class _ScriptedEnv:
    """Deterministic env emitting a real obs batch dict each step.

    Unlike the ``test_rollout`` scripted env (which returns an opaque
    ``{"obs": seed}``), this returns the batch dict ``DiscretePolicy``
    expects, so the rollout loop drives the *real* policy decode -- the
    genuine end-to-end path -- without a simulator. Each episode ends
    after ``ep_len`` steps and reports a pre-scripted success flag.
    """

    def __init__(self, obs, successes, ep_len=2):
        self._obs = obs
        self.successes = successes
        self.ep_len = ep_len
        self._t = 0
        self._seed = 0

    def reset(self, seed=0):
        self._t = 0
        self._seed = seed
        return self._obs, {}

    def step(self, action):
        self._t += 1
        term = self._t >= self.ep_len
        info = {"success": torch.tensor([self.successes[self._seed]])}
        return self._obs, 0.0, term, False, info

    def close(self):
        pass


def test_e2e_toy_pipeline_fake_backbone(toy_ds, tmp_path):
    """Full pipeline on the toy dataset with the fake backbone.

    One test, every stage; each assert pins a real shape/contract so the
    chain is verified to *compose* on real LeRobot-format data.
    """
    repo_id, root = toy_ds
    torch.manual_seed(0)

    # (a) Tokenizer from the real dataset's action column. The fake vocab
    # only fits 8 bins, so build at n_bins = FAKE_BINS.
    ds = make_chunk_dataset(
        repo_id=repo_id, chunk_h=CHUNK_H, root=str(root)
    )
    tok = ActionTokenizer.from_lerobot_dataset(ds, n_bins=FAKE_BINS)
    assert tok.action_dim == ACTION_DIM
    assert tok.n_bins == FAKE_BINS
    assert np.all(np.isfinite(tok.lo)) and np.all(np.isfinite(tok.hi))
    assert np.all(tok.lo < tok.hi)  # non-degenerate per-dim range

    # (b) Chunked loader: real lerobot batch shapes.
    loader = make_chunk_loader(ds, batch_size=BATCH, num_workers=0)
    batch = next(iter(loader))
    assert batch["action"].shape == (BATCH, CHUNK_H, ACTION_DIM)
    assert "observation.images.up" in batch
    assert "observation.images.side" in batch
    # Video keys come back squeezed; tabular state keeps the chunk dim.
    assert batch["observation.images.up"].shape[:2] == (BATCH, 3)
    assert batch["observation.state"].shape == (BATCH, 1, ACTION_DIM)
    assert "action_is_pad" in batch
    assert batch["action_is_pad"].shape == (BATCH, CHUNK_H)

    # (c) Fusion adapter + both heads at the fake-matching config.
    fusion = FusionAdapter(FakeBackbone())
    head = AutoregressiveActionHead(
        fusion,
        d_embed=FAKE_WIDTH,
        n_bins=tok.n_bins,
        act_token_base=FAKE_BASE,
        bos_id=1,
    )
    parallel = ParallelActionHead(
        d_embed=FAKE_WIDTH,
        chunk_h=CHUNK_H,
        action_dim=ACTION_DIM,
        n_bins=tok.n_bins,
        hidden=32,
    )
    # The prefix encodes the fused observation from a real batch; both
    # heads must consume it.
    obs = _obs_from_batch(batch)
    prefix = fusion.encode_prefix(obs)
    assert prefix.shape[0] == 1 and prefix.shape[2] == FAKE_WIDTH
    targets = torch.randint(0, tok.n_bins, (1, CHUNK_H * ACTION_DIM))
    ar_logits = head.logits(prefix, targets)
    assert ar_logits.shape == (1, CHUNK_H * ACTION_DIM, tok.n_bins)
    par_logits = parallel(prefix.mean(dim=1))  # pool prefix -> [1, W]
    assert par_logits.shape == (
        1, CHUNK_H, ACTION_DIM, tok.n_bins
    )

    # (d) Train a few steps on the real loader; checkpoint round-trip.
    out_dir = tmp_path / "ckpts"
    cfg = TrainConfig(
        n_epochs=1,
        total_steps=6,
        warmup_steps=2,
        microbatch=BATCH,
        grad_accum=1,
        bf16=False,
        steps_per_checkpoint=6,
        log_every=1,
        out_dir=str(out_dir),
    )
    history = train(
        head, fusion, tok, loader, cfg, log_fn=lambda *_: None
    )
    assert set(history) >= {"step", "loss", "entropy"}
    assert history["loss"] and all(
        math.isfinite(x) for x in history["loss"]
    )
    ckpt = out_dir / "step_6.pt"
    assert ckpt.exists()
    step = load_checkpoint(str(ckpt), head, fusion)
    assert step == 6

    # (e) Policy decode: one KV-cached decode fills the H-step buffer.
    policy = DiscretePolicy(
        fusion, head, tok,
        chunk_h=CHUNK_H, action_dim=ACTION_DIM,
        strategy="argmax", device="cpu",
    )
    decode_calls = {"n": 0}
    real_decode = policy._decode_chunk

    def counting_decode(obs_in):
        decode_calls["n"] += 1
        return real_decode(obs_in)

    policy._decode_chunk = counting_decode
    first = policy.select_action(obs)
    assert first.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(first))
    for _ in range(CHUNK_H - 1):  # exhaust the rest of the chunk
        policy.select_action(obs)
    assert decode_calls["n"] == 1  # exactly one decode over H calls
    policy.select_action(obs)  # buffer empty -> a second decode fires
    assert decode_calls["n"] == 2

    # (f) Open-loop eval on the real loader.
    eval_loader = make_chunk_loader(ds, batch_size=BATCH, num_workers=0)
    result = evaluate_open_loop(
        head, fusion, tok, eval_loader, n_batches=2
    )
    assert math.isfinite(result["val_loss"])
    assert len(result["per_dim_loss"]) == ACTION_DIM
    assert result["n_tokens"] > 0

    # (g) Closed-loop rollout loop against a scripted env driving the
    # real policy (no simulator).
    successes = [True, False, True]
    env = _ScriptedEnv(obs, successes, ep_len=2)
    rate, per_seed = evaluate(policy, n_seeds=3, env=env)
    assert len(per_seed) == 3
    assert per_seed == successes
    assert rate == pytest.approx(2 / 3)


@pytest.mark.integration
def test_e2e_toy_pipeline_real_backbone(toy_ds, tmp_path):
    """Full-stack e2e with the real ch03 backbone over the toy dataset.

    Proves the shipped 576-wide backbone, the 256-bin / 48896-base
    tokenizer/head configuration, and the real LeRobot data schema all
    compose. Kept to a few train steps and a tiny batch because the real
    backbone forward on CPU is heavy; it downloads SmolLM2 + SigLIP once
    (cached thereafter).
    """
    from ch03 import UnifiedEmbeddingBackbone

    from ch04 import ACT_TOKEN_BASE, N_BINS

    repo_id, root = toy_ds
    torch.manual_seed(0)

    ds = make_chunk_dataset(
        repo_id=repo_id, chunk_h=CHUNK_H, root=str(root)
    )
    tok = ActionTokenizer.from_lerobot_dataset(ds)  # real 256 bins
    assert tok.n_bins == N_BINS
    assert np.all(tok.lo < tok.hi)

    loader = make_chunk_loader(ds, batch_size=1, num_workers=0)
    batch = next(iter(loader))
    assert batch["action"].shape == (1, CHUNK_H, ACTION_DIM)
    assert batch["observation.state"].shape == (1, 1, ACTION_DIM)

    backbone = UnifiedEmbeddingBackbone().float()
    fusion = FusionAdapter(backbone)
    head = AutoregressiveActionHead(
        fusion,
        d_embed=576,
        n_bins=N_BINS,
        act_token_base=ACT_TOKEN_BASE,
        bos_id=1,
    )

    # Prefix + head logits at the real config.
    obs = _obs_from_batch(batch)
    with torch.no_grad():
        prefix = fusion.encode_prefix(obs)
        assert prefix.shape[0] == 1 and prefix.shape[2] == 576
        targets = torch.randint(
            0, N_BINS, (1, CHUNK_H * ACTION_DIM)
        )
        logits = head.logits(prefix, targets)
        assert logits.shape == (1, CHUNK_H * ACTION_DIM, N_BINS)

    # A few train steps on the real data + backbone.
    out_dir = tmp_path / "ckpts_real"
    cfg = TrainConfig(
        n_epochs=1,
        total_steps=3,
        warmup_steps=1,
        microbatch=1,
        grad_accum=1,
        bf16=False,
        steps_per_checkpoint=3,
        log_every=1,
        out_dir=str(out_dir),
    )
    history = train(
        head, fusion, tok, loader, cfg, log_fn=lambda *_: None
    )
    assert history["loss"] and all(
        math.isfinite(x) for x in history["loss"]
    )
    assert (out_dir / "step_3.pt").exists()

    # Open-loop eval closes the loop on the real stack.
    eval_loader = make_chunk_loader(ds, batch_size=1, num_workers=0)
    result = evaluate_open_loop(
        head, fusion, tok, eval_loader, n_batches=2
    )
    assert math.isfinite(result["val_loss"])
    assert len(result["per_dim_loss"]) == ACTION_DIM
    assert result["n_tokens"] > 0
