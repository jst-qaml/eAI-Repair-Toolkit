"""The German Traffic Sign Recognition Benchmark (GTSRB).

cf. https://github.com/wakamezake/gtrsb
"""

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
