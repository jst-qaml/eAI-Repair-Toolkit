"""Utility function: test a model and display results as a bubble chart."""

from pathlib import Path

import numpy as np

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf

from .plot_bubble_chart import plot_bubble_chart


def run(
    *,
    model_dir: str,
    test_dir: str,
    test_data: str = "test.h5",
    output_dir: str = "outputs",
    filename: str = "bubble.png",
):
    """Draw bubble chart.

    This utilty plots the prediction result of the model as bubble chart.

    Parameters
    ----------
    model_dir : str
        A path to the directory where the target model exists.
    test_dir : str
        A path to the directory where the 'test_data' exists.
    test_data : str, default="test.h5"
        A file name of the test dataset.
    output_dir : str
        A path to the directory where the generated image will be saved.
    filename : str, default="bubble.png"
        A file name of the generated image.

    """
    if model_dir is None:
        raise ValueError("'model_dir' is required.")
    model_dir = Path(model_dir)

    if test_dir is None:
        raise ValueError("'test_dir' is required.")
    test_dir = Path(test_dir)

    if not test_data.endswith(".h5"):
        test_data = f"{test_data}.h5"

    output_dir = Path(output_dir)

    model = load_model_from_tf(model_dir)

    # get test data.
    # test_lables are to be set as ground truth.
    test_dataset = RepairDataset.load_dataset_from_hdf(test_dir, test_data)
    test_images, test_labels = test_dataset[0], test_dataset[1]

    def labels_from_categorical(labels):
        """Convert labels to integer.

        This function do reversal operation of `to_categorical()`

        Parameters
        ----------
        labels : numpy.ndarray
            A one-hot vectored labels

        Returns
        -------
        np.array
            Integer labels

        """
        return np.array(list(map(lambda label: label.argmax(), labels)))

    ground_truth = labels_from_categorical(test_labels)

    results = model.predict(test_images, verbose=0)
    pred_labels = labels_from_categorical(results)

    plot_bubble_chart(ground_truth, pred_labels, output_dir / filename)
