"""Make `src/` and the repo root importable for the test suite.

After the 2026-04 reorganization, the original solver pipeline scripts
(``generate_functional_equation_system``, ``decode_solutions``, ``list_eqn``)
moved from the repo root into ``src/``. Tests import them by bare name, so
we add ``src/`` to sys.path here. We also keep the repo root on sys.path so
that the ``ca_search`` package and any classifier helpers resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for entry in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "classifier"):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
