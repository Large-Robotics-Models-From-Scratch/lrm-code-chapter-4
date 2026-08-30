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
    # The heads must run in manuscript order: one-shot, AR, then parallel.
    calls = [
        code.index(f"results['{name}'] = run_head_experiment(")
        for name in ("factorized", "autoregressive", "parallel")
    ]
    assert calls == sorted(calls)


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
        "figure 4.9": "plot_joint_logit_panels",
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


def test_colab_has_fixed_sanity_and_full_training_modes():
    code = _notebook_code()
    assert "RUN_MODE = 'sanity'" in code
    assert "'sanity': dict(steps=10" in code
    assert "'full': dict(steps=20_000" in code
    assert "log_every=1" in code
    assert "tensorboard_log_dir" in code
    assert "plot_per_joint_metrics" in code
    assert "export_action_chunk" in code

def test_colab_setup_removes_only_broken_optional_torchaudio():
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    setup = "".join(notebook["cells"][1]["source"])

    # Order matters: install first, then probe, then remove.
    assert (
        setup.index("'pip', 'install'")
        < setup.index("'import torch, torchaudio'")
        < setup.index("'pip', 'uninstall'")
    )
    # The probe must run out-of-process, or a failed import taints the
    # kernel that is about to load SigLIP.
    assert "subprocess.run(" in setup
    assert "audio_probe.returncode" in setup
    # Remove only a wheel that is both present and broken.
    assert "importlib.util.find_spec('torchaudio')" in setup
    assert "if audio_installed and audio_probe.returncode:" in setup
    assert "'uninstall', '--yes'" in setup
    assert "'torchaudio'" in setup


def test_colab_setup_verifies_the_import_that_was_failing():
    """The probe must cover `from ch03 import VLABackbone` itself."""
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    setup = "".join(notebook["cells"][1]["source"])

    # Match the probe command, not the comment that explains it.
    probe = "'from ch03 import VLABackbone; '"
    assert probe in setup
    assert setup.index("'pip', 'uninstall'") < setup.index(probe)


def test_colab_force_refreshes_and_verifies_branch_api():
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    setup = "".join(notebook["cells"][1]["source"])

    assert "--force-reinstall" in setup
    assert "--no-deps" in setup
    assert "chapter4_requirement" in setup
    assert "from ch04.decoding import decode_action_chunk" in setup
    assert "sample_action_grids" in setup
    assert "sys.modules.pop(name, None)" in setup
