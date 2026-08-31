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


def _bound_names(tree) -> set:
    """Every name a cell binds: imports, assignments, defs, loop targets."""
    import ast

    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
    return bound


def _module_level_loads(tree) -> set:
    """Names read at module level, ignoring deferred function bodies.

    A function body may reference a global defined by a later cell: that
    resolves at call time and is a normal notebook pattern. A name read at
    module level must already exist, which is the failure this catches.
    """
    import ast

    loaded = set()

    scopes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
    )

    def visit(node, deferred):
        if isinstance(node, scopes):
            deferred = True
        if (
            not deferred
            and isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
        ):
            loaded.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child, deferred)

    visit(tree, False)
    return loaded


def test_colab_defines_every_name_it_uses():
    """A cell reading a function local at module level would fail."""
    import ast
    import builtins

    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    available = set(dir(builtins))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        missing = sorted(
            _module_level_loads(tree) - _bound_names(tree) - available
        )
        assert not missing, f"cell {index} reads undefined {missing}"
        available |= _bound_names(tree)


def test_run_configuration_precedes_the_trainers():
    """TRAIN_STEPS must be editable without re-running a training cell."""
    path = Path(__file__).parents[1] / "notebooks/ch04.ipynb"
    notebook = json.loads(path.read_text())
    sources = [
        "".join(cell["source"]) if cell["cell_type"] == "code" else ""
        for cell in notebook["cells"]
    ]
    assigns = [
        index for index, src in enumerate(sources)
        if "TRAIN_STEPS = " in src
    ]
    trains = [
        index for index, src in enumerate(sources)
        if "run_head_experiment('" in src
    ]
    assert len(assigns) == 1, "TRAIN_STEPS must be set in exactly one cell"
    assert len(trains) == 3, "expected one cell per head"
    # Set before any trainer runs, and never in the same cell as one.
    assert assigns[0] < min(trains)
    assert assigns[0] not in trains
    # Every knob the three runs share lives in that one cell, so the
    # heads cannot silently receive different budgets.
    config = sources[assigns[0]]
    for knob in (
        "RUN_MODE",
        "GRID_SAMPLES",
        "AR_GRID_SAMPLES",
        "EVAL_BATCHES",
        "WARMUP_STEPS",
        "LOG_EVERY",
        "CHECKPOINT_EVERY",
        "VALIDATE_EVERY",
    ):
        assert f"{knob} = " in config, knob
    # The trainer cells take the budget from it rather than restating it.
    for index in trains:
        assert "steps=" not in sources[index], sources[index]


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


def test_full_colab_mirrors_and_resumes_checkpoints_from_google_drive():
    code = _notebook_code()
    text = _notebook_text()
    for token in (
        "SAVE_TO_GOOGLE_DRIVE = True",
        "RESUME_FROM_GOOGLE_DRIVE = False",
        "DRIVE_CHECKPOINT_ROOT",
        "drive.mount('/content/drive')",
        "checkpoint_mirror_dir=mirror_dir",
        "resume_from=resume_path",
    ):
        assert token in code
    assert "every 1,000-step local checkpoint" in text
    assert "atomically" in text

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
