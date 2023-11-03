"""Utility function: merge repair data."""

import shutil
from pathlib import Path

import numpy as np

from repair.core.dataset import RepairDataset


def run(**kwargs):
    """Merge repair data.

    :param dataset:
    :param kwargs:
    """
    if "input_dir1" in kwargs:
        input_dir1 = Path(kwargs["input_dir1"])
    else:
        raise TypeError("Require --input_dir1")
    if "input_dir2" in kwargs:
        input_dir2 = Path(kwargs["input_dir2"])
    else:
        raise TypeError("Require --input_dir2")
    if "output_dir" in kwargs:
        output_dir = Path(kwargs["output_dir"])
    else:
        raise TypeError("Require --output_dir")
    # Load test data
    dataset1 = RepairDataset.load_repair_data(input_dir1)
    _test_images1, _test_labels1 = dataset1[0], dataset1[1]

    dataset2 = RepairDataset.load_repair_data(input_dir2)
    _test_images2, _test_labels2 = dataset2[0], dataset2[1]

    # Merge test data
    test_images = []
    test_labels = []
    for i in range(len(_test_images1)):
        test_images.append(_test_images1[i])
        test_labels.append(_test_labels1[i])
    for i in range(len(_test_images2)):
        test_images.append(_test_images2[i])
        test_labels.append(_test_labels2[i])
    # Format test data
    test_images = np.array(test_images, dtype="float32")
    test_labels = np.array(test_labels, dtype="float32")
    # Save test data
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    RepairDataset.save_dataset_as_hdf(test_images, test_labels, output_dir / "repair.h5")
