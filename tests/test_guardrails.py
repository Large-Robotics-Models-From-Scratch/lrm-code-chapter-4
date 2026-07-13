"""Cross-chapter guardrails for the ch04 source.

These tests encode contracts that are easy to violate silently:

1. The action tokenizer stays pure NumPy (no torch), so it runs at
   deployment time on an edge CPU exactly as in the book's claim.
2. No JAX / TensorFlow anywhere in the package (PyTorch only, per the
   book's locked stack).
3. Chapter 4 reserves ids inside the existing SmolLM2 vocabulary; it
   must never resize or extend it.
4. Chapter 4 consumes the Chapter 3 backbone read-only; it must never
   assign into ``ch03`` module attributes (mutation guard).
5. The reserved token range sits exactly at the top of the vocabulary.
"""

import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ch04"


def _module_text(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def test_tokenizer_is_torch_free():
    if not (SRC / "action_tokenizer.py").exists():
        pytest.skip("action_tokenizer.py lands in PR 1")
    text = _module_text("action_tokenizer.py")
    assert "import torch" not in text
    assert "from torch" not in text


def test_no_jax_or_tensorflow_in_package():
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import jax" not in text, path.name
        assert "import tensorflow" not in text, path.name


def test_ch4_does_not_resize_the_vocabulary():
    """Vocab reservation lives in Ch 4's tokenizer, not by resizing.

    The OpenVLA recipe reserves existing ids; it never calls
    ``resize_token_embeddings`` or ``add_tokens`` on the Chapter 3
    backbone (which keeps its native 49,152-id SmolLM2 vocab).
    """
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "resize_token_embeddings" not in text, path.name
        assert "add_tokens" not in text, path.name


def test_ch4_never_mutates_ch3_module_attributes():
    """Ch 4 reads the Ch 3 backbone; it never assigns into it.

    Flags ``ch03.something = ...`` (including dotted chains) and
    ``setattr(ch03...`` in any ch04 source file. ``==`` comparisons
    and keyword arguments do not match.
    """
    assign = re.compile(r"(?<![\w.])ch03\.[\w.]+\s*=(?!=)")
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not assign.search(text), path.name
        assert "setattr(ch03" not in text, path.name


def test_reserved_range_tops_out_the_vocab():
    from ch04 import ACT_TOKEN_BASE, N_BINS, SMOLLM_VOCAB

    assert ACT_TOKEN_BASE + N_BINS == SMOLLM_VOCAB
    assert SMOLLM_VOCAB == 49152
