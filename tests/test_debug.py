"""Debug test environment.

This test exists for debugging test environments and will not be run except directly
designated to run. If you want to run this tests, run
`pytest -s -m debug tests/test_debug.py`.
"""
import os
import sys

import pytest


@pytest.mark.debug
def test_enviromnent():
    print(f"{os.environ.get('PYTHONPATH')=}")
    print(f"{sys.path=}")
