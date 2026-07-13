"""Smoke tests for the ch04 package.

Verifies the package installs, imports cleanly, and exposes the
chapter constants downstream modules and tests build on. Real
per-module coverage lands in tests/test_action_tokenizer.py (PR 1)
onward.
"""


def test_ch04_imports():
    import ch04

    assert ch04 is not None


def test_chapter_constants():
    import ch04

    assert ch04.N_BINS == 256
    assert ch04.CHUNK_H == 16
    assert ch04.ACTION_DIM == 6
    assert ch04.ACT_TOKEN_BASE == 48896
