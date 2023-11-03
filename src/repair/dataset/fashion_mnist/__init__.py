"""Fashion-MNIST.

cf.
https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from repair.core import dataset

from . import prepare

__all__ = [
    "FashionMNIST",
]


class FashionMNIST(dataset.RepairDataset):
    """API for DNN with Fashion-MNIST."""

    @classmethod
    def get_name(cls) -> str:
        """Returns dataset name."""
        return "fashion-mnist"

    @staticmethod
    def get_label_map() -> dict[str, int]:
        """Returns map of label name and its id/value."""
        return {
            "T-shirt/top": 0,
            "Trouser": 1,
            "Pullover": 2,
            "Dress": 3,
            "Coat": 4,
            "Sandal": 5,
            "Shirt": 6,
            "Sneaker": 7,
            "Bag": 8,
            "Ankle boot": 9,
        }

    def _get_input_shape(self):
        """Set the input_shape and classes of BDD."""
        return (32, 32, 3), 10

    def prepare(
        self,
        input_dir: Path,
        output_dir: Path,
        divide_rate: float,
        random_state: Optional[int],
    ):
        """Prepare Fashion-MNIST dataset.

        Parameters
        ----------
        input_dir : Path
            (not used in Fashion-MNIST dataset)
        output_dir : Path
        divide_rate : float
        random_state : int, optional

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        prepare.prepare(output_dir, divide_rate, random_state)
