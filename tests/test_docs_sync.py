"""README layout block stays in sync with the tracked file tree.

Parses the fenced tree under "## Repository layout" in README.md and
asserts every listed path is tracked by git. Catches the classic
scaffold rot where a file is renamed or moved but the README still
advertises the old layout.
"""

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
BRANCH = re.compile(r"^(?P<indent>(?:[│ ]\s{3})*)"
                    r"[├└]── (?P<name>\S+)")


def _layout_paths() -> list[str]:
    """Extract the paths listed in the README layout tree."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"## Repository layout.*?```\n(.*?)```", text, re.DOTALL
    )
    assert match, "README.md has no fenced Repository layout block"
    lines = match.group(1).splitlines()

    paths: list[str] = []
    stack: list[str] = []  # directory names by depth
    for line in lines[1:]:  # line 0 is the repo-name root
        m = BRANCH.match(line)
        if not m:
            continue
        depth = len(m.group("indent")) // 4
        name = m.group("name")
        stack = stack[:depth]
        if name.endswith("/"):
            stack.append(name.rstrip("/"))
            continue
        paths.append("/".join([*stack, name]))
    return paths


def test_readme_layout_matches_git_ls_files():
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    )
    listed = _layout_paths()
    assert listed, "no paths parsed from the README layout block"
    missing = [p for p in listed if p not in tracked]
    assert not missing, f"README lists untracked paths: {missing}"
