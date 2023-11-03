"""CIFAR-10.

cf. https://www.cs.toronto.edu/~kriz/cifar.html
"""
from __future__ import annotations

from pathlib import Path

from repair.core import dataset

from . import prepare

__all__ = [
    "CIFAR10",
]


class CIFAR10(dataset.RepairDataset):
    """API for DNN with CIFAR-10."""

    @classmethod
    def get_name(cls) -> str:
        """Returns dataset name."""
        return "cifar-10"

    @staticmethod
    def get_label_map() -> dict[str, int]:
        return {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }

    def _get_input_shape(self):
        """Set the input_shape and classes of CIFAR-10."""
        return (32, 32, 3), 10

    def prepare(self, input_dir: Path, output_dir: Path, divide_rate, random_state):
        """Prepare Cifar-10 dataset.

        Parameters
        ----------
        input_dir : Path
            (not used in CIFAR-10 dataset)
        output_dir : Path
        divide_rate : float
        random_state : int, optional

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        prepare.prepare(output_dir, divide_rate, random_state)
