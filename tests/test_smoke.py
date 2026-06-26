"""Smoke tests for the ch04 package.

Verifies the package installs and imports cleanly. Real per-module
coverage lands in tests/test_action_tokenizer.py (PR 1),
tests/test_action_head_ar.py (PR 3), etc.
"""


def test_ch04_imports():
    import ch04

    assert ch04 is not None
