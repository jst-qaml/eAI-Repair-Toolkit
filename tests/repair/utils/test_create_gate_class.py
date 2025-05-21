import json

import h5py
import numpy as np

import pytest

from repair.utils import create_gate_class


@pytest.fixture
def fmnist_subtasks():
    return [[0, 2, 4, 6], [1, 3], [5, 7, 8, 9]]


@pytest.fixture
def hydra_settings(fmnist_subtasks, tmp_path):
    path = tmp_path / "hydra_fmnist.json"
    with open(path, "w") as f:
        json.dump(fmnist_subtasks, f)

    return path


def test_generate_subtask_labels(fmnist_subtasks):
    labels = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    new_labels = create_gate_class._create_new_labels(labels, fmnist_subtasks)

    np.testing.assert_equal(new_labels, expected)


def test_create_gate_dataset(fashion_mnist_repair_data_dir, hydra_settings):
    data_dir = fashion_mnist_repair_data_dir
    create_gate_class.run(data_dir=data_dir, hydra_setting_file=hydra_settings)

    assert (data_dir / "gate").exists(), "Gate dir was not created."

    assert (data_dir / "gate" / "train.h5").exists(), "Train data was not created."
    with h5py.File(data_dir / "gate" / "train.h5") as hf:
        assert hf["images"].shape[0] == hf["labels"].shape[0]
        assert (
            hf["labels"].shape[1] == 3
        ), "The shape of gate labels are not equal to the number of branches."

    assert (data_dir / "gate" / "test.h5").exists(), "Test data was not created."
    with h5py.File(data_dir / "gate" / "train.h5") as hf:
        assert hf["images"].shape[0] == hf["labels"].shape[0]
        assert (
            hf["labels"].shape[1] == 3
        ), "The shape of gate labels are not equal to the number of branches."

    assert (data_dir / "gate" / "repair.h5").exists(), "Repair data was not created."
    with h5py.File(data_dir / "gate" / "train.h5") as hf:
        assert hf["images"].shape[0] == hf["labels"].shape[0]
        assert (
            hf["labels"].shape[1] == 3
        ), "The shape of gate labels are not equal to the number of branches."
