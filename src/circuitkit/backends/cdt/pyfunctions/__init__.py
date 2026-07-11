# -*- coding: utf-8 -*-
"""CD-T pyfunctions package.

Internal modules use top-level ``from pyfunctions.X import Y`` imports
rather than relative ones. To make them work when nested inside
``backends/cdt/``, we add the parent directory to sys.path at
package-import time so ``pyfunctions`` resolves as a top-level module.
"""
import sys as _sys
from pathlib import Path as _Path

_pkg_parent = str(_Path(__file__).resolve().parent.parent)
if _pkg_parent not in _sys.path:
    _sys.path.insert(0, _pkg_parent)
