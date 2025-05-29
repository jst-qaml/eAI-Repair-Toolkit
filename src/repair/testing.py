"""Testing utilities.

This module provides some utilities for testing.
"""
import os
from pathlib import Path


def get_cache_root():
    """Get cache root.

    Returns
    -------
    Path
        A path to the root directory for caching. We use keras' cache directory
        at current implementation.

    """
    cache_root = Path(os.environ.get("KERAS_HOME", "~/.keras")).expanduser()
    if not cache_root.exists():
        cache_root.mkdir(parents=True)

    return cache_root
