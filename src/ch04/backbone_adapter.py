"""Chapter 4 adapter for the Chapter 3 VLA backbone."""

from __future__ import annotations

import torch


def prefix_valid_mask(backbone, sequence_ids: torch.Tensor) -> torch.Tensor:
    """Mark real prefix tokens through each row's state placeholder.

    Variable-length rows are padded *after* the state token. The state
    position therefore identifies the last real prefix token without
    relying on a language tokenizer's potentially ambiguous pad id.
    """
    state_matches = sequence_ids == backbone.state_id
    counts = state_matches.sum(dim=1)
    if not bool((counts == 1).all()):
        raise ValueError("each row needs exactly one state placeholder")
    state_positions = state_matches.to(torch.int64).argmax(dim=1)
    positions = torch.arange(
        sequence_ids.shape[1], device=sequence_ids.device
    )
    return positions[None, :] <= state_positions[:, None]


def causal_mask_with_prefix_padding(
    prefix_valid: torch.Tensor,
    extra_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a causal mask that excludes right-padded prefix keys."""
    if prefix_valid.ndim != 2 or prefix_valid.dtype != torch.bool:
        raise ValueError("prefix_valid must be bool with shape [B, N]")
    if extra_tokens < 0:
        raise ValueError("extra_tokens must be non-negative")
    batch_size, n_prefix = prefix_valid.shape
    n_total = n_prefix + extra_tokens
    device = prefix_valid.device
    mask = torch.zeros(
        batch_size, 1, n_total, n_total, device=device, dtype=dtype
    )
    future = torch.triu(
        torch.ones(n_total, n_total, device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask.masked_fill_(future[None, None], -torch.inf)
    valid_keys = torch.cat(
        [
            prefix_valid,
            torch.ones(
                batch_size, extra_tokens, device=device, dtype=torch.bool
            ),
        ],
        dim=1,
    )
    mask.masked_fill_(~valid_keys[:, None, None, :], -torch.inf)
    return mask


def position_ids_with_prefix_padding(
    prefix_valid: torch.Tensor,
    extra_tokens: int,
) -> torch.Tensor:
    """Assign positions as if right-padded prefix tokens did not exist.

    Action tokens are physically appended after the rectangular padded
    prefix. Compact position ids keep their RoPE positions immediately after
    each row's state token, independent of other instructions in the batch.
    """
    if prefix_valid.ndim != 2 or prefix_valid.dtype != torch.bool:
        raise ValueError("prefix_valid must be bool with shape [B, N]")
    if extra_tokens < 0:
        raise ValueError("extra_tokens must be non-negative")
    valid = torch.cat(
        [
            prefix_valid,
            torch.ones(
                prefix_valid.shape[0],
                extra_tokens,
                device=prefix_valid.device,
                dtype=torch.bool,
            ),
        ],
        dim=1,
    )
    return (valid.to(torch.long).cumsum(dim=1) - 1).clamp_min_(0)


def embed_inputs(
    backbone,
    images: torch.Tensor,
    sequence_ids: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Run Chapter 3's splice and stop before SmolLM.

    The manuscript calls this operation ``backbone.embed_inputs``. The
    current Chapter 3 handoff exposes the necessary modules but not the
    combined method, so Chapter 4 owns this thin adapter.
    """
    batch_size = images.shape[0]
    patches = backbone.vision_encoder(images.flatten(0, 1))
    image_tokens = patches.reshape(batch_size, -1, patches.shape[-1])
    embeddings = backbone.embed_tokens(sequence_ids)
    image_tokens = image_tokens.to(embeddings.dtype)
    state_token = backbone.state_encoder(state).to(embeddings.dtype)

    image_mask = (sequence_ids == backbone.image_id).unsqueeze(-1)
    state_mask = (sequence_ids == backbone.state_id).unsqueeze(-1)
    expected_images = image_tokens.shape[1]
    image_counts = image_mask.sum(dim=(1, 2))
    state_counts = state_mask.sum(dim=(1, 2))
    if not bool((image_counts == expected_images).all()):
        raise ValueError(
            f"each row needs {expected_images} image placeholders"
        )
    if not bool((state_counts == 1).all()):
        raise ValueError("each row needs exactly one state placeholder")
    embeddings = embeddings.masked_scatter(image_mask, image_tokens)
    return embeddings.masked_scatter(state_mask, state_token)


def encode_prefix(
    backbone,
    images: torch.Tensor,
    sequence_ids: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Encode a padded perception prefix with a causal validity mask."""
    embeddings = embed_inputs(backbone, images, sequence_ids, state)
    valid = prefix_valid_mask(backbone, sequence_ids)
    mask = causal_mask_with_prefix_padding(valid, 0, embeddings.dtype)
    position_ids = position_ids_with_prefix_padding(valid, 0)
    return backbone.language_backbone(
        inputs_embeds=embeddings,
        attention_mask=mask,
        position_ids=position_ids,
    ).last_hidden_state


def gather_state_hidden(
    backbone,
    hidden: torch.Tensor,
    sequence_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather each row's state hidden vector from a padded prefix."""
    state_positions = (
        (sequence_ids == backbone.state_id).to(torch.int64).argmax(dim=1)
    )
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, state_positions]
