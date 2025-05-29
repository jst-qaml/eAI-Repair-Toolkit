"""Test suite for repair.core.dataset."""

import h5py
import numpy as np

import pytest

from repair.core.dataset import RepairDataset


@pytest.fixture
def fake_dataset():
    def _fake_dataset(data_len: int = 3):
        rng = np.random.default_rng()
        fake_imgs = rng.random((data_len, 32, 32, 3))
        fake_labels = np.eye(10)[rng.choice(10, data_len)]
        return fake_imgs, fake_labels

    return _fake_dataset


def test_save_dataset_as_hdf(tmp_path, fake_dataset):
    fake_datafile = "fake.h5"
    fake_imgs, fake_labels = fake_dataset()

    RepairDataset.save_dataset_as_hdf(fake_imgs, fake_labels, tmp_path / fake_datafile)

    assert (tmp_path / fake_datafile).exists()

    hf = h5py.File(tmp_path / fake_datafile)
    np.testing.assert_equal(hf["images"], fake_imgs)
    np.testing.assert_equal(hf["labels"], fake_labels)


def test_load_dataset_from_hdf(tmp_path, fake_dataset):
    fake_datapath = tmp_path / "fake.h5"
    fake_imgs, fake_labels = fake_dataset()
    with h5py.File(fake_datapath, "w") as hf:
        hf.create_dataset("images", data=fake_imgs)
        hf.create_dataset("labels", data=fake_labels)

    loaded_imgs, loaded_labels = RepairDataset.load_dataset_from_hdf(tmp_path, "fake.h5")

    np.testing.assert_equal(loaded_imgs, fake_imgs)
    np.testing.assert_equal(loaded_labels, fake_labels)


def test_divide_train_dataset_without_seeds(fake_dataset):
    fake_imgs, fake_labels = fake_dataset(10)
    test_divide_rate = 0.2

    train_data, repair_data = RepairDataset.divide_train_dataset(
        list(fake_imgs), list(fake_labels), test_divide_rate
    )

    train_imgs, train_labels = train_data
    repair_imgs, repair_labels = repair_data

    assert len(train_imgs) == len(
        train_labels
    ), "Divided train labels and imags must be same length."
    assert len(repair_imgs) == len(
        repair_labels
    ), "Divided repair labels and imags must be same length."

    assert (
        len(repair_imgs) == 2
    ), "Divided repair length must be same as length specified by `divide_rate`."
    assert (
        len(train_imgs) == 8
    ), "Divided train length must be same as rest of length specified by `divide_rate`."


def test_divide_train_dataset_with_seeds(fake_dataset):
    fake_imgs, fake_labels = fake_dataset(10)
    test_divide_rate = 0.2

    train_data, repair_data = RepairDataset.divide_train_dataset(
        list(fake_imgs), list(fake_labels), test_divide_rate, 42
    )

    train_imgs, train_labels = train_data
    repair_imgs, repair_labels = repair_data

    assert len(train_imgs) == len(
        train_labels
    ), "Divided train labels and imags must be same length."
    assert len(repair_imgs) == len(
        repair_labels
    ), "Divided repair labels and imags must be same length."

    assert (
        len(repair_imgs) == 2
    ), "Divided repair length must be same as length specified by `divide_rate`."
    assert (
        len(train_imgs) == 8
    ), "Divided train length must be same as rest of length specified by `divide_rate`."
