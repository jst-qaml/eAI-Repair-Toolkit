"""Utility function: merge repair data."""

import shutil

import numpy as np

from repair.core.dataset import RepairDataset


def run(*, input_dir1: str, input_dir2: str, output_dir: str):
    """Merge repair data.

    Parameters
    ----------
    input_dir1 : str
        A path to the directory where repair dataset exists.
    input_dir2 : str
        A path to the directory where repair dataset exists.
    output_dir : str
        A path to the directory where the merged dataset will be saved.

    """
    if input_dir1 is None:
        raise ValueError("'input_dir1' is required.")

    if input_dir2 is None:
        raise ValueError("'input_dir2' is required.")

    if output_dir is None:
        raise ValueError("'output_dir' is required.")

    dataset1 = RepairDataset.load_repair_data(input_dir1)
    _test_images1, _test_labels1 = dataset1[0], dataset1[1]

    dataset2 = RepairDataset.load_repair_data(input_dir2)
    _test_images2, _test_labels2 = dataset2[0], dataset2[1]

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
