"""Repair."""

from abc import ABCMeta, abstractmethod
from pathlib import Path


class RepairMethod(metaclass=ABCMeta):
    """Meta class of Repair method."""

    @classmethod
    def get_name(cls) -> str:
        """Returns name of this class."""
        return cls.__name__

    @abstractmethod
    def set_options(self, **kwargs):
        """Set options.

        Parameters
        ----------
        **kwargs
            Additional options for each repair methods.

        """
        pass

    @abstractmethod
    def localize(self, model, input_neg, output_dir: Path = Path("outputs"), **kwargs):
        """Localize neural weight candidates to repair.

        Parameters
        ----------
        model : repair.model.model.Model
            DNN model to be repaired
        input_neg : tuple[np.ndarray, np.ndarray]
            A set of inputs that reveal the fault
        output_dir : Path, default=Path("outputs")
            Path to directory to save the result
        **kwargs
            Additional args for localize

        """
        pass

    @abstractmethod
    def optimize(
        self,
        model,
        model_dir,
        weights,
        input_neg,
        input_pos,
        output_dir: Path = Path("outputs"),
        **kwargs,
    ):
        """Optimize neural weight candidates to repair.

        Parameters
        ----------
        model :
            DNN model to be repaired
        weights :
            Set of neural weights to target for repair
        input_neg:
            Dataset of unexpected behavior
        input_pos:
            Dataset of correct behavior
        output_dir : Path, default=Path("outputs")
            Path to directory to save the result
        **kwargs
            Additional args for optimize

        """
        pass

    @abstractmethod
    def save_weights(self, weights, output_dir: Path):
        """Save neural weight candidates.

        Parameters
        ----------
        weights
            Neural weights to be saved
        output_dir : Path
            Path to directory to save weights
        """
        pass

    @abstractmethod
    def load_weights(self, weights_dir: Path):
        """Load neural weight candidates.

        Parameters
        ----------
        weights_dir : Path
            Path to directory containing a target `labels.json`

        """
        pass

    @abstractmethod
    def load_input_neg(self, neg_dir: Path):
        """Load negative inputs.

        Parameters
        ----------
        neg_dir : Path
            Path to directory containing negative dataset

        """
        pass

    @abstractmethod
    def load_input_pos(self, pos_dir: Path):
        """Load positive inputs.

        Parameters
        ----------
        pos_dir : Path
            Path to directory containing positive dataset

        """
        pass

    @abstractmethod
    def evaluate(
        self,
        dataset,
        model_dir: Path,
        target_data,
        target_data_dir: Path,
        positive_inputs,
        positive_inputs_dir: Path,
        output_dir: Path,
        num_runs,
        **kwargs,
    ):
        """Evaluate repairing performance.

        Parameters
        ----------
        dataset :
            Dataset instance
        model_dir : Path
            Path to directory containing model files
        target_data:
            Negative dataset
        target_data_dir : Path
            Path to directory containing negative dataset
        positive_inputs :
            Positive dataset
        positive_inputs_dir : Path
            Path to directory containing positive dataset
        output_dir : Path
            Path to directory to save results
        num_runs : int
            Number of iterations for repairing

        """
        pass
