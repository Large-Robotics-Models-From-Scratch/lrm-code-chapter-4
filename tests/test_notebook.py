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


def test_colab_builds_and_compares_all_three_heads():
    code = _notebook_code()
    for name in ("Factorized", "Autoregressive", "ParallelDecode"):
        assert f"{name}ActionHead" in code
    assert "build_action_head" in code
    assert "ch04-train --head all" in "".join(
        "".join(cell["source"])
        for cell in json.loads(
            (
                Path(__file__).parents[1] / "notebooks/ch04.ipynb"
            ).read_text()
        )["cells"]
    )


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
