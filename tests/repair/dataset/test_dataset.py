"""Test suite for repair.core.dataset."""

import h5py
import numpy as np

import pytest

from repair.core.dataset import RepairDataset


@pytest.fixture
def fake_dataset():
    data_counts = 3
    rng = np.random.default_rng()
    fake_imgs = np.random.random((data_counts, 32, 32, 3))
    fake_labels = np.eye(10)[rng.choice(10, data_counts)]
    return fake_imgs, fake_labels


def test_save_dataset_as_hdf(tmp_path, fake_dataset):
    fake_datafile = "fake.h5"
    fake_imgs, fake_labels = fake_dataset

    RepairDataset.save_dataset_as_hdf(fake_imgs, fake_labels, tmp_path / fake_datafile)

    assert (tmp_path / fake_datafile).exists()

    hf = h5py.File(tmp_path / fake_datafile)
    np.testing.assert_equal(hf["images"], fake_imgs)
    np.testing.assert_equal(hf["labels"], fake_labels)


def test_load_dataset_from_hdf(tmp_path, fake_dataset):
    fake_datapath = tmp_path / "fake.h5"
    fake_imgs, fake_labels = fake_dataset
    with h5py.File(fake_datapath, "w") as hf:
        hf.create_dataset("images", data=fake_imgs)
        hf.create_dataset("labels", data=fake_labels)

    loaded_imgs, loaded_labels = RepairDataset.load_dataset_from_hdf(tmp_path, "fake.h5")

    np.testing.assert_equal(loaded_imgs, fake_imgs)
    np.testing.assert_equal(loaded_labels, fake_labels)
