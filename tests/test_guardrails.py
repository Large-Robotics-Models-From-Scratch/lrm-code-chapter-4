"""Cross-chapter guardrails for the ch04 source.

These tests encode contracts that are easy to violate silently:

1. The action tokenizer stays pure NumPy (no torch), so it runs at
   deployment time on an edge CPU exactly as in the book's claim.
2. No JAX / TensorFlow anywhere in the package (PyTorch only, per the
   book's locked stack).
3. Chapter 4 owns the action-token vocabulary reservation; it must never
   reach into the Chapter 3 source to mutate the backbone's embeddings.
"""

import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ch04"


def _module_text(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def test_tokenizer_is_torch_free():
    text = _module_text("action_tokenizer.py")
    assert "import torch" not in text
    assert "from torch" not in text


def test_no_jax_or_tensorflow_in_package():
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import jax" not in text, path.name
        assert "import tensorflow" not in text, path.name


def test_ch4_does_not_mutate_ch3_embeddings():
    """Vocab reservation lives in Ch 4's tokenizer, not by resizing Ch 3.

    The OpenVLA recipe reserves existing ids; it never calls
    ``resize_token_embeddings`` or ``add_tokens`` on the Chapter 3
    backbone (which keeps its native SmolLM vocab).
    """
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "resize_token_embeddings" not in text, path.name
        assert "add_tokens" not in text, path.name
