"""Shared fixtures for Chapter 4 unit tests."""

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
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class FakeLanguageBackbone(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.last_attention_mask = None
        self.last_position_ids = None

    def forward(
        self, inputs_embeds, attention_mask=None, position_ids=None
    ):
        self.last_attention_mask = attention_mask
        self.last_position_ids = position_ids
        return FakeOutput(self.projection(inputs_embeds))


class FakeVisionEncoder(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.width = width
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
    eos_token_id = 1


class FakeBackbone(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.width = width
        self.image_id = 300
        self.state_id = 301
        self.embed_tokens = nn.Embedding(49_154, width)
        self.vision_encoder = FakeVisionEncoder(width)
        self.state_encoder = FakeStateEncoder(width)
        self.language_backbone = FakeLanguageBackbone(width)
        self.tokenizer = FakeTokenizer()
        self.seen_ids = []

    def tokenize_instruction(self, instruction):
        return [10 + len(instruction) % 3, 11]

    def build_sequence_ids(self, text_ids):
        return [self.image_id] * 4 + text_ids + [self.state_id]

    def forward(self, images, sequence_ids, state):
        self.seen_ids.append(sequence_ids.detach().clone())
        from ch04.backbone_adapter import embed_inputs

        embeddings = embed_inputs(self, images, sequence_ids, state)
        return self.language_backbone(
            inputs_embeds=embeddings
        ).last_hidden_state


@pytest.fixture
def fake_backbone():
    return FakeBackbone()


@pytest.fixture
def model_inputs(fake_backbone):
    images = torch.rand(2, 2, 3, 8, 8)
    rows = [fake_backbone.build_sequence_ids([10, 11])] * 2
    sequence_ids = torch.tensor(rows)
    state = torch.rand(2, 6)
    return images, sequence_ids, state
