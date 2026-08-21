"""Shared fixtures for Chapter 4 unit tests."""

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


class FakeLanguageBackbone(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.embedding = nn.Embedding(512, width)
        self.last_attention_mask = None
        self.last_position_ids = None

    def get_input_embeddings(self):
        return self.embedding

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
        cache = object() if use_cache else past_key_values
        return FakeOutput(self.projection(inputs_embeds), cache)


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
def model_inputs(fake_backbone):
    images = torch.rand(2, 2, 3, 8, 8)
    input_ids = torch.tensor([[10, 11], [10, 11]])
    state = torch.rand(2, 6)
    text_attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return images, input_ids, state, text_attention_mask
