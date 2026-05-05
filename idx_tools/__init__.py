"""
Package init for idx_tools.

This project keeps historical "top-level" imports inside idx_tools modules
(e.g. `from tools import ...`, `from Wiki import ...`).

To make those imports work when code is used via:
  `from idx_tools.xxx import ...`
we add this package directory to `sys.path` at import time.
"""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Make `idx_tools/*.py` importable as top-level modules (tools, Wiki, ...).
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
