"""Package initialization for hutton_lm.

Adds the project root to ``sys.path`` once so that the package's example
scripts can import modules when run directly from the source tree.
"""

from __future__ import annotations

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

__all__ = []
