"""Berkley Deep Drive (BDD100K).

cf. https://bdd-data.berkeley.edu/index.html
"""
from __future__ import annotations

from pathlib import Path

from repair.core import dataset

from . import prepare

__all__ = [
    "BDD",
]


class BDD(dataset.RepairDataset):
    """API for DNN with BDD."""

    def __init__(self):
        """Initialize."""
        self.target_label = "weather"

    @classmethod
    def get_name(cls) -> str:
        """Returns dataset name."""
        return "bdd"

    @staticmethod
    def get_label_map() -> dict[str, int]:
        # NOTE: BDD100k provides some labels by categories.
        #       currently adapted the labels for object detection.
        return {
            "pedestrian": 1,
            "rider": 2,
            "car": 3,
            "truck": 4,
            "bus": 5,
            "train": 6,
            "motorcycle": 7,
            "bicycle": 8,
            "traffic light": 9,
            "traffic sign": 10,
        }

    def _get_input_shape(self):
        """Set the input_shape and classes of BDD."""
        return (90, 160, 3), 6

    def set_target_label(self, target_label="weather"):
        """Set target label."""
        self.target_label = target_label

    def set_extra_config(self, **kwargs):
        """Set target label after cli.py."""
        target_label = kwargs.get("target_label", "weather")
        self.set_target_label(target_label)

    def prepare(self, input_dir: Path, output_dir: Path, divide_rate, random_state):
        """Prepare BDD dataset.

        Parameters
        ----------
        input_dir : Path
        output_dir : Path
        divide_rate : float
        random_state : int, optional

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        prepare.prepare(input_dir, output_dir, divide_rate, random_state, self.target_label)
