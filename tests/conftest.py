"""Shared fixtures for Chapter 4 unit tests."""

import math

import matplotlib

# Figure tests must never try to open a window on CI.
matplotlib.use("Agg")
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def action_bounds():
    return -np.ones(6, dtype=np.float32), np.ones(6, dtype=np.float32)


@pytest.fixture
def fake_stats():
    return {
        "action": {
            "mean": torch.arange(6, dtype=torch.float32),
            "std": torch.full((6,), 2.0),
        },
        "observation.state": {
            "mean": torch.zeros(6),
            "std": torch.ones(6),
        },
    }


class FakeOutput:
    def __init__(self, hidden, past_key_values=None):
        self.last_hidden_state = hidden
        self.past_key_values = past_key_values


class FakeCache:
    """The minimum key/value cache the AR generation loop relies on."""

    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, keys, values):
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = torch.cat([self.keys, keys], dim=1)
            self.values = torch.cat([self.values, values], dim=1)
        return self.keys, self.values


class FakeLanguageBackbone(nn.Module):
    """A one-head attention stub that honours the mask and positions.

    A stub that ignores ``attention_mask`` makes every masking test
    vacuous: padded keys and a bidirectional action block would be
    indistinguishable from a position-wise linear. This stub therefore
    runs real scaled dot-product attention, applies either a 2-D key
    validity mask or a 4-D additive mask, adds a position embedding, and
    supports incremental decoding so one cached step is numerically
    equal to recomputing the whole prefix.
    """

    def __init__(self, width=12):
        super().__init__()
        self.width = width
        self.projection = nn.Linear(width, width)
        self.embedding = nn.Embedding(512, width)
        self.positions = nn.Embedding(512, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.last_attention_mask = None
        self.last_position_ids = None

    def get_input_embeddings(self):
        return self.embedding

    def _additive_mask(
        self, attention_mask, n_query, n_key, dtype, device
    ):
        if attention_mask is None:
            return torch.zeros(1, n_query, n_key, device=device).to(dtype)
        if attention_mask.dim() == 4:
            return attention_mask[:, 0, -n_query:, :].to(dtype)
        valid = attention_mask.bool()[:, None, :]
        bias = torch.zeros(valid.shape[0], n_query, n_key, device=device)
        bias = bias.masked_fill(~valid, -torch.inf)
        future = torch.triu(
            torch.ones(n_query, n_key, dtype=torch.bool, device=device),
            diagonal=1 + n_key - n_query,
        )
        return bias.masked_fill(future[None], -torch.inf).to(dtype)

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
    ):
        self.last_attention_mask = attention_mask
        self.last_position_ids = position_ids
        hidden = self.projection(inputs_embeds)
        if position_ids is not None:
            hidden = hidden + self.positions(position_ids)
        queries = self.query(hidden)
        keys, values = self.key(hidden), self.value(hidden)

        cache = past_key_values
        if use_cache or cache is not None:
            if cache is None:
                cache = FakeCache()
            keys, values = cache.update(keys, values)
        scores = queries @ keys.transpose(-1, -2)
        scores = scores / math.sqrt(self.width)
        scores = scores + self._additive_mask(
            attention_mask,
            queries.shape[1],
            keys.shape[1],
            scores.dtype,
            scores.device,
        )
        attended = scores.softmax(dim=-1) @ values
        return FakeOutput(hidden + attended, cache if use_cache else None)


class FakeVisionEncoder(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.width = width
        self.project = nn.Linear(width, width)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        pooled = images.mean(dim=(1, 2, 3), keepdim=False)
        return pooled[:, None, None].expand(-1, 2, self.width)


class FakeStateEncoder(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.layer = nn.Linear(6, width)

    def forward(self, state):
        return self.layer(state)[:, None]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, tasks, padding=True, return_tensors="pt"):
        rows = [
            [10 + len(word) % 7 for word in task.split()]
            for task in tasks
        ]
        width = max(len(row) for row in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class FakeBackbone(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.width = width
        self.vision_encoder = FakeVisionEncoder(width)
        self.state_encoder = FakeStateEncoder(width)
        self.language_backbone = FakeLanguageBackbone(width)
        self.tokenizer = FakeTokenizer()

    def embed_inputs(
        self, images, input_ids, state, text_attention_mask=None
    ):
        batch_size = images.shape[0]
        image = self.vision_encoder(images.flatten(0, 1)).reshape(
            batch_size, 4, self.width
        )
        text = self.language_backbone.get_input_embeddings()(input_ids)
        state_embedding = self.state_encoder(state)
        embeddings = torch.cat([image, text, state_embedding], dim=1)
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(input_ids)
        valid = torch.cat(
            [
                torch.ones(
                    batch_size, 4, dtype=torch.bool, device=images.device
                ),
                text_attention_mask.bool(),
                torch.ones(
                    batch_size, 1, dtype=torch.bool, device=images.device
                ),
            ],
            dim=1,
        )
        positions = valid.long().cumsum(1) - 1
        positions.masked_fill_(~valid, 0)
        return embeddings, valid, positions

    def contextualize(self, embeddings, attention_mask, position_ids):
        return self.language_backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state

    def forward(
        self, images, input_ids, state, text_attention_mask=None
    ):
        embeddings, valid, positions = self.embed_inputs(
            images, input_ids, state, text_attention_mask
        )
        return self.contextualize(embeddings, valid, positions)


@pytest.fixture
def fake_backbone():
    return FakeBackbone()


@pytest.fixture
def chunk_batch():
    """One LeRobot-shaped batch with a two-camera observation."""

    def build(batch_size=2, horizon=16, action_dim=6):
        torch.manual_seed(0)
        return {
            "observation.images.up": torch.rand(batch_size, 3, 20, 30),
            "observation.images.side": torch.rand(batch_size, 3, 20, 30),
            "observation.state": torch.rand(batch_size, 6),
            "action": torch.rand(batch_size, horizon, action_dim),
            "action_is_pad": torch.zeros(
                batch_size, horizon, dtype=torch.bool
            ),
            "task": ["pick up the object"] * batch_size,
        }

    return build


@pytest.fixture
def model_inputs(fake_backbone):
    images = torch.rand(2, 2, 3, 8, 8)
    input_ids = torch.tensor([[10, 11], [10, 11]])
    state = torch.rand(2, 6)
    text_attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return images, input_ids, state, text_attention_mask
