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


def test_colab_setup_removes_only_broken_optional_torchaudio():
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    setup = "".join(notebook["cells"][1]["source"])

    install_index = setup.index("'pip', 'install'")
    probe_index = setup.index("'import torch, torchaudio'")
    uninstall_index = setup.index("'pip', 'uninstall'")

    assert install_index < probe_index < uninstall_index
    assert "if audio_probe.returncode:" in setup
    assert "'--yes',\n             'torchaudio'" in setup


def test_colab_runs_shared_experiment_for_all_heads_in_order():
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )

    assert "def run_head_experiment(" in source
    assert "action_head_logits(" in source
    assert "sample_action_grids(" in source
    calls = [
        source.index("run_head_experiment(\n    'factorized'"),
        source.index("run_head_experiment(\n    'autoregressive'"),
        source.index("run_head_experiment(\n    'parallel'"),
    ]
    assert calls == sorted(calls)
    assert "backbone, head = build_action_head(name)" in source
