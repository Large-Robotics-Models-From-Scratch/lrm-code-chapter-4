"""Test the toy MSE-collapse demo (Listing 4.1).

The core is factored into ``run_toy_bimodal`` so it is importable and
testable; the test runs fewer steps than the manuscript's 5000 and
asserts the prediction collapses toward 0 (the empty valley between the
two modes), well inside |pred| < 0.2.
"""

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "toy_bimodal.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "toy_bimodal", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_toy_bimodal_collapses_to_valley():
    mod = _load_module()
    pred = mod.run_toy_bimodal(seed=0, steps=800)
    assert abs(pred) < 0.2
