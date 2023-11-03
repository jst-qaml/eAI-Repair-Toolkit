"""Create dataset for training vision transformer model."""
from pathlib import Path

import numpy

from repair.core.dataset import RepairDataset


def run(**kwargs):
    """Create dataset for training vision transformer model.

    :param dataset:
    """
    if "data_dir" in kwargs:
        data_dir = Path(kwargs["data_dir"])
    else:
        raise TypeError("Require --data_dir")

    _create_gate_file(data_dir, "train.h5")


def _create_gate_file(data_dir, target_file):
    data = RepairDataset.load_dataset_from_hdf(data_dir, target_file)
    images, labels = data[0], data[1]
    new_labels = _create_new_labels(labels)

    gate_dir = data_dir / "vision_transformer"
    if not gate_dir.exists():
        gate_dir.mkdir()

    RepairDataset.save_dataset_as_hdf(images, new_labels, gate_dir / target_file)


def _create_new_labels(labels):
    new_labels = numpy.argmax(labels, axis=1)
    return new_labels
