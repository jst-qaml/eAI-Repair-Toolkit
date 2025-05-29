"""Create dataset for training vision transformer model."""
from pathlib import Path

from repair.core.dataset import RepairDataset


def run(**kwargs):
    """Create dataset for training vision transformer model."""
    if "data_dir" in kwargs:
        data_dir = Path(kwargs["data_dir"])
    else:
        raise TypeError("Require --data_dir")

    _create_gate_file(data_dir, "train.h5")


def _create_gate_file(data_dir, target_file):
    images, labels = RepairDataset.load_dataset_from_hdf(data_dir, target_file)

    new_labels = labels_from_categorical(labels)

    gate_dir = data_dir / "vision_transformer"
    if not gate_dir.exists():
        gate_dir.mkdir(parents=True)

    RepairDataset.save_dataset_as_hdf(images, new_labels, gate_dir / target_file)


def labels_from_categorical(labels):  # noqa: D103
    return labels.argmax(axis=1)
