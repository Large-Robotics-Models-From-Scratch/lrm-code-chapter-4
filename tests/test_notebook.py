import json
from pathlib import Path


def test_colab_is_valid_json_and_code_cells_compile():
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) >= 15
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"ch04.ipynb:cell-{index}", "exec")


def test_colab_setup_installs_public_chapter_packages_directly():
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    setup = "".join(notebook["cells"][1]["source"])

    assert "git+https://github.com/" in setup
    assert "GITHUB_TOKEN" not in setup
    assert "git', 'clone" not in setup
    assert "capture_output=True" in setup


def _notebook_code() -> str:
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def _notebook_text() -> str:
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )


def test_colab_trains_all_three_heads():
    code = _notebook_code()
    assert "build_action_head" in code
    for name in ("factorized", "autoregressive", "parallel"):
        assert f"results['{name}'] = run_head_experiment(" in code, name
    # Each head must get its own backbone, or the comparison is invalid.
    assert "def make_policy" in code
    assert "VLABackbone().to(device)" in code
    assert "ch04-train --head all" in _notebook_text()


def test_colab_defines_every_name_it_uses():
    """A spliced cell that reads a function local would fail at runtime."""
    import ast
    import builtins

    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    available = set(dir(builtins))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        loaded, stored = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                (loaded if isinstance(node.ctx, ast.Load) else stored).add(
                    node.id
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    stored.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                stored.add(node.name)
                stored.update(
                    argument.arg
                    for argument in ast.walk(node)
                    if isinstance(argument, ast.arg)
                )
        missing = sorted(loaded - stored - available)
        assert not missing, f"cell {index} reads undefined {missing}"
        available |= stored


def test_colab_produces_every_code_backed_figure():
    """Each manuscript figure with a code path must appear in the Colab."""
    code = _notebook_code()
    required = {
        "figure 4.4": "plot_bimodal_comparison",
        "figure 4.8": "neighborhood_softmax_figure",
        "figure 4.9": "plot_joint_mismatch_panels",
        "section 4.6.2": "plot_temporal_traces",
        "figure 4.10": "plot_execution_schedules",
        "figure 4.11": "plot_open_loop_episode",
    }
    missing = [
        figure for figure, symbol in required.items()
        if symbol not in code
    ]
    assert not missing, f"the Colab never produces {missing}"


def test_colab_records_the_provenance_section_461_requires():
    code = _notebook_code()
    for token in ("ANCHOR_INDEX", "N_NEIGHBORS", "SEED", "set_seed"):
        assert token in code
