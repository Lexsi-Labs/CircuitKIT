"""The import hook serves exactly what ``git apply`` of the series would write to disk."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from circuitkit._tl_patches import PATCH_DIR, TransformerLensPatchFinder, parse_series


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_hook_source_matches_git_apply(tmp_path):
    pkg = Path(importlib.util.find_spec("transformer_lens").origin).parent
    shutil.copytree(pkg, tmp_path / "transformer_lens", ignore=shutil.ignore_patterns("__pycache__"))
    for patch in sorted(PATCH_DIR.glob("*.patch")):
        subprocess.run(
            ["git", "apply", str(patch)],
            cwd=tmp_path,
            check=True,
            env={"GIT_CEILING_DIRECTORIES": str(tmp_path.parent), "PATH": "/usr/bin:/bin"},
        )
    finder = TransformerLensPatchFinder(parse_series())
    assert len(finder.series) >= 14
    for name in finder.series:
        spec = finder.find_spec(name, _parent_path(name))
        expected = (tmp_path / (name.replace(".", "/"))).with_suffix(".py")
        if not expected.exists():
            expected = tmp_path / name.replace(".", "/") / "__init__.py"
        assert spec.loader.get_source(name) == expected.read_text(), name


def _parent_path(name):
    import importlib

    return importlib.import_module(name.rpartition(".")[0]).__path__
