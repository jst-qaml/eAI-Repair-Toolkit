from unittest.mock import MagicMock

import pytest

from repair.core.exceptions import (
    RepairDatasetNotFoundError,
    RepairError,
    RepairMethodNotFoundError,
    RepairModelNotFoundError,
    RepairUtilNotFoundError,
)
from repair.core.loader import (
    _gather_repair_classes,
    load_repair_dataset,
    load_repair_method,
    load_repair_model,
    load_utils,
)
from repair.core.method import RepairMethod
from repair.dataset.demo import DemoDataset
from repair.dataset.demopkg import PkgDemoDataset
from repair.methods.demo import DemoMethod
from repair.methods.demo2 import DemoMethod2
from repair.model.demo import DemoModel


def test_dataset_loader_mod():
    loaded_class = load_repair_dataset(DemoDataset.get_name())

    assert loaded_class is DemoDataset


def test_dataset_loader_pkg():
    loaded_class = load_repair_dataset(PkgDemoDataset.get_name())

    assert loaded_class


def test_dataset_loader_not_found():
    with pytest.raises(RepairDatasetNotFoundError):
        load_repair_dataset("unknown_dataset")


def test_method_loader():
    loaded_class = load_repair_method(DemoMethod.get_name())

    assert loaded_class is DemoMethod


def test_method_loader_not_found():
    with pytest.raises(RepairMethodNotFoundError):
        load_repair_method("unknown_method")


def test_model_loader():
    loaded_class = load_repair_model(DemoModel.get_name())

    assert loaded_class is DemoModel


def test_model_loader_not_found():
    with pytest.raises(RepairModelNotFoundError):
        load_repair_model("unknown_model")


def test_raise_name_conflict_error(mocker):
    DemoMethod2.get_name = MagicMock(return_value=DemoMethod.get_name())

    with pytest.raises(RepairError):
        _gather_repair_classes("repair.methods", RepairMethod)


def test_load_utils(tmpdir):
    utils = load_utils("demo")
    utils.run(output_dir=tmpdir)

    assert (tmpdir / "output.txt").exists()


def test_load_uitils_not_found():
    with pytest.raises(RepairUtilNotFoundError):
        load_utils("unknown_util")


def test_load_utils_check_spec():
    with pytest.raises(RepairError):
        load_utils("demo_invalid")
