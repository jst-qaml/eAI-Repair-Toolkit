"""Provides some utility methods."""
from __future__ import annotations

import sys


def get_python_version() -> tuple[int, int, int]:
    """Get Python version as tuple.

    Returns
    -------
    tuple[int, int, int]
        Python version (major, minor, micro)

    """
    return sys.version_info[:3]
