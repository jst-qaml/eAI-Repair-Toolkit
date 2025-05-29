from repair.core.utils import get_python_version


def test_get_python_version():
    version = get_python_version()

    assert len(version) == 3
    assert all(isinstance(v, int) for v in version)
