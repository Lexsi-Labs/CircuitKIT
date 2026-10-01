"""Apply CircuitKIT's TransformerLens patch series at import time.

The ``*.patch`` files in this directory add Gemma-4, Sarvam-MoE and Cohere support
to ``transformer-lens==3.8.0`` (see README.md here). :func:`install` puts an import
hook on ``sys.meta_path`` that serves each patched ``transformer_lens`` module from
its installed source with the series applied in memory. Nothing on disk changes,
so ``pip install circuitkit`` needs no extra step. ``import circuitkit`` installs
the hook; it must run before ``transformer_lens`` is first imported.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import linecache
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TL_VERSION = "3.8.0"
PATCH_DIR = Path(__file__).resolve().parent
# Written by the old scripts/apply_tl_patches.py into a patched site-packages copy.
LEGACY_MARKER = "_circuitkit_patches.txt"

_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
Hunk = Tuple[int, List[str]]  # (old start line, body lines with their ' '/'-'/'+' prefix)
# One inner list per patch: hunk line numbers are relative to the file as the
# previous patches left it.
Series = Dict[str, List[List[Hunk]]]


def _patch_enabled(patch: Path) -> bool:
    """Keep the numerically different memory optimization explicitly opt-in."""
    return patch.name != "0005-memory-no-weight-copies.patch" or os.environ.get(
        "CIRCUITKIT_TL_MEMORY_PATCH"
    ) == "1"


def parse_series(patch_dir: Path = PATCH_DIR) -> Series:
    """``{module name: [hunks of each patch that touches it]}``, patches in file-name order."""
    series: Series = {}
    for patch in sorted(patch_dir.glob("*.patch")):
        if not _patch_enabled(patch):
            continue
        lines = patch.read_text(encoding="utf-8").splitlines()
        i, target = 0, None
        while i < len(lines):
            line = lines[i]
            if line.startswith("+++ b/"):
                path = line[len("+++ b/") :]
                target = path[: -len(".py")].replace("/", ".").removesuffix(".__init__")
                series.setdefault(target, []).append([])
            m = _HUNK.match(line)
            if m and target:
                old_n = int(m.group(2) or 1)
                new_n = int(m.group(4) or 1)
                body: List[str] = []
                i += 1
                while old_n or new_n:
                    text = lines[i]
                    i += 1
                    if text.startswith("\\"):  # "\ No newline at end of file"
                        continue
                    tag = text[:1] or " "
                    body.append(tag + text[1:])
                    old_n -= tag in " -"
                    new_n -= tag in " +"
                series[target][-1].append((int(m.group(1)), body))
                continue
            i += 1
    return series


def apply_patches(source: str, patches: List[List[Hunk]], name: str) -> str:
    """Apply each patch's hunks in order, checking every context and removed line."""
    out = source.splitlines(keepends=True)
    for hunks in patches:
        offset = 0
        for start, body in hunks:
            pos = max(start - 1, 0) + offset
            old = [b[1:] for b in body if b[0] in " -"]
            new = [b[1:] + "\n" for b in body if b[0] in " +"]
            if [x.rstrip("\n") for x in out[pos : pos + len(old)]] != old:
                raise ImportError(
                    f"CircuitKIT's TransformerLens patch series does not apply to {name} "
                    f"(line {start}). It targets transformer-lens=={TL_VERSION}; reinstall it."
                )
            out[pos : pos + len(old)] = new
            offset += len(new) - len(old)
    return "".join(out)


class _PatchedLoader(importlib.abc.Loader):
    def __init__(self, source: str):
        self.source = source

    def get_source(self, fullname: str) -> str:
        return self.source

    def exec_module(self, module) -> None:
        origin = module.__spec__.origin
        # Tracebacks and inspect show the patched lines, not the file on disk.
        linecache.cache[origin] = (len(self.source), None, self.source.splitlines(True), origin)
        exec(compile(self.source, origin, "exec"), module.__dict__)


class TransformerLensPatchFinder(importlib.abc.MetaPathFinder):
    """Serve patched ``transformer_lens`` modules; defer everything else."""

    def __init__(self, series: Series):
        self.series = series

    def find_spec(self, fullname, path=None, target=None):
        patches = self.series.get(fullname)
        if patches is None:
            return None
        real = importlib.machinery.PathFinder.find_spec(fullname, path)
        if real is not None:
            source = Path(real.origin).read_text(encoding="utf-8")
            origin, search = real.origin, real.submodule_search_locations
        else:  # a file the series creates
            parent = sys.modules[fullname.rpartition(".")[0]]
            source = ""
            origin = str(Path(parent.__path__[0]) / (fullname.rpartition(".")[2] + ".py"))
            search = None
        loader = _PatchedLoader(apply_patches(source, patches, fullname))
        return importlib.util.spec_from_file_location(
            fullname, origin, loader=loader, submodule_search_locations=search
        )


def install() -> Optional[TransformerLensPatchFinder]:
    """Put the patch hook on ``sys.meta_path`` (idempotent). Returns it, or ``None`` if skipped."""
    for finder in sys.meta_path:
        if isinstance(finder, TransformerLensPatchFinder):
            return finder
    try:
        version = importlib.metadata.version("transformer-lens")
    except importlib.metadata.PackageNotFoundError:
        return None
    spec = importlib.util.find_spec("transformer_lens")
    if spec is None or (Path(spec.origin).parent / LEGACY_MARKER).exists():
        return None  # not importable, or patched on disk by the old script
    if version != TL_VERSION:
        warnings.warn(
            f"transformer-lens {version} is installed; CircuitKIT's Gemma-4 / Sarvam-MoE / "
            f"Cohere support needs transformer-lens=={TL_VERSION} and is disabled.",
            stacklevel=2,
        )
        return None
    if "transformer_lens" in sys.modules:
        raise ImportError(
            "transformer_lens was imported before circuitkit, so CircuitKIT's Gemma-4 / "
            "Sarvam-MoE / Cohere compatibility patches were not installed. Restart the "
            "process and import circuitkit before transformer_lens."
        )
    series = parse_series()
    if not series:
        raise ImportError(f"CircuitKIT's TransformerLens patch files are missing from {PATCH_DIR}.")
    finder = TransformerLensPatchFinder(series)
    sys.meta_path.insert(0, finder)
    return finder


def is_applied() -> bool:
    """True iff the loaded ``transformer_lens`` carries this series (hook or legacy script)."""
    import transformer_lens.pretrained.weight_conversions as wc

    return hasattr(wc, "convert_cohere_weights")
