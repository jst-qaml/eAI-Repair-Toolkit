"""The German Traffic Sign Recognition Benchmark (GTSRB).

cf. https://github.com/wakamezake/gtrsb
"""
from __future__ import annotations

from pathlib import Path

from repair.core import dataset

from . import prepare

__all__ = [
    "GTSRB",
]


class GTSRB(dataset.RepairDataset):
    """API for DNN with GTSRB."""

    @classmethod
    def get_name(cls) -> str:
        """Returns dataset name."""
        return "gtsrb"

    @staticmethod
    def get_label_map() -> dict[str, int]:
        return {
            "Speed limit (20km/h)": 0,
            "Speed limit (30km/h)": 1,
            "Speed limit (50km/h)": 2,
            "Speed limit (60km/h)": 3,
            "Speed limit (70km/h)": 4,
            "Speed limit (80km/h)": 5,
            "End of speed limit (80km/h)": 6,
            "Speed limit (100km/h)": 7,
            "Speed limit (120km/h)": 8,
            "No passing": 9,
            "No passing for vechiles over 3.5 metric tons": 10,
            "Right-of-way at the next intersections": 11,
            "Priority road": 12,
            "Yield": 13,
            "Stop": 14,
            "No vechiles": 15,
            "Vechiles over 3.5 metric tons prohibited": 16,
            "No entry": 17,
            "General caution": 18,
            "Dangerous curve to the left": 19,
            "Dangerous curve to the right": 20,
            "Double curve": 21,
            "Bumpy road": 22,
            "Slippery road": 23,
            "Road narrows on the right": 24,
            "Road work": 25,
            "Traffic signals": 26,
            "Pedestrians": 27,
            "Children crossing": 28,
            "Bicycles crossing": 29,
            "Beware of ice/snow": 30,
            "Wild animals crossing": 31,
            "End of all speed and passing limits": 32,
            "Turn right ahead": 33,
            "Turn left ahead": 34,
            "Ahead only": 35,
            "Go straight or right": 36,
            "Go straight or left": 37,
            "Keep right": 38,
            "Keep left": 39,
            "Roundabout mandatory": 40,
            "End of no passing": 41,
            "End of no passing by vechiles over 3.5 metric tons": 42,
        }

    def prepare(self, input_dir: Path, output_dir: Path, divide_rate, random_state):
        """Prepare.

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

        prepare.prepare(input_dir, output_dir, divide_rate, random_state)

    def _get_input_shape(self):
        """Set the input_shape and classes of BDD."""
        classes = 43
        return (32, 32, 3), classes
