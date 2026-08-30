"""Guard the Chapter 3 -> Chapter 4 integration boundary."""

from __future__ import annotations

from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src" / "ch04"


def test_chapter4_does_not_reintroduce_retired_fusion_api():
    source = "\n".join(
        path.read_text() for path in sorted(SOURCE.glob("*.py"))
    )
    retired_names = (
        "UnifiedEmbeddingBackbone",
        "FusionAdapter",
        "masked_scatter",
        "resize_token_embeddings",
        "image_id",
        "state_id",
        "img_proj",
        "state_proj",
    )
    for name in retired_names:
        assert name not in source


def test_chapter4_leaves_image_resizing_to_chapter3():
    source = "\n".join(
        path.read_text() for path in sorted(SOURCE.glob("*.py"))
    )
    assert "F.interpolate" not in source
    assert "torch.nn.functional.interpolate" not in source
