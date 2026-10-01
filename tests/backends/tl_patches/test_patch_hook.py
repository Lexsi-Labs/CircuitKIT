"""The import hook serves exactly what ``git apply`` of the series would write to disk."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from circuitkit._tl_patches import (
    PATCH_DIR,
    TransformerLensPatchFinder,
    _patch_enabled,
    install,
    parse_series,
)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize("memory_opt_in", [False, True])
def test_hook_source_matches_git_apply(tmp_path, monkeypatch, memory_opt_in):
    if memory_opt_in:
        monkeypatch.setenv("CIRCUITKIT_TL_MEMORY_PATCH", "1")
    else:
        monkeypatch.delenv("CIRCUITKIT_TL_MEMORY_PATCH", raising=False)
    pkg = Path(importlib.util.find_spec("transformer_lens").origin).parent
    shutil.copytree(pkg, tmp_path / "transformer_lens", ignore=shutil.ignore_patterns("__pycache__"))
    for patch in sorted(PATCH_DIR.glob("*.patch")):
        if not _patch_enabled(patch):
            continue
        subprocess.run(
            ["git", "apply", str(patch)],
            cwd=tmp_path,
            check=True,
            env={"GIT_CEILING_DIRECTORIES": str(tmp_path.parent), "PATH": "/usr/bin:/bin"},
        )
    finder = TransformerLensPatchFinder(parse_series())
    assert len(finder.series) >= 12
    for name in finder.series:
        spec = finder.find_spec(name, _parent_path(name))
        expected = (tmp_path / (name.replace(".", "/"))).with_suffix(".py")
        if not expected.exists():
            expected = tmp_path / name.replace(".", "/") / "__init__.py"
        assert spec.loader.get_source(name) == expected.read_text(encoding="utf-8"), name


def test_memory_patch_is_opt_in(monkeypatch):
    memory_patch = PATCH_DIR / "0005-memory-no-weight-copies.patch"
    monkeypatch.delenv("CIRCUITKIT_TL_MEMORY_PATCH", raising=False)
    assert not _patch_enabled(memory_patch)
    assert "transformer_lens.components.mlps.gated_mlp" not in parse_series()

    monkeypatch.setenv("CIRCUITKIT_TL_MEMORY_PATCH", "1")
    assert _patch_enabled(memory_patch)
    assert "transformer_lens.components.mlps.gated_mlp" in parse_series()


def test_importing_transformer_lens_first_is_an_error(monkeypatch):
    import transformer_lens  # noqa: F401 — intentionally reverse import order

    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            finder
            for finder in sys.meta_path
            if not isinstance(finder, TransformerLensPatchFinder)
        ],
    )
    with pytest.raises(ImportError, match="import circuitkit before transformer_lens"):
        install()


def _parent_path(name):
    import importlib

    return importlib.import_module(name.rpartition(".")[0]).__path__
