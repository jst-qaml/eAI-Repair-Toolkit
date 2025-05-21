import pytest


def run(**kwargs):
    if (output_dir := kwargs.get("output_dir", None)) is None:
        raise pytest.fail("should be given 'output_dir' from test runner.")

    with open(output_dir / "output.txt", "w"):
        pass
