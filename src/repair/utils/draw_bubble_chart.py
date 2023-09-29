"""Utility function: test a model and display results as a bubble chart."""

from pathlib import Path

import numpy as np

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf

from .plot_bubble_chart import plot_bubble_chart


def run(**kwargs):
    """Draw bubble chart.

    :param kwargs:
    """
    if "model_dir" in kwargs:
        model_dir = Path(kwargs["model_dir"])
    else:
        raise TypeError("Require --model_dir")

    if "test_dir" in kwargs:
        test_dir = Path(kwargs["test_dir"])
    else:
        raise TypeError("Require --test_dir")

    if "output_dir" in kwargs:
        output_dir = Path(kwargs["output_dir"])
    else:
        output_dir = Path(r"outputs/")

    if "test_data" in kwargs:
        test_data = kwargs["test_data"]
        if not test_data.endswith(".h5"):
            raise TypeError("File type must be '.h5'")
    else:
        test_data = "test.h5"

    filename = kwargs["filename"] if "filename" in kwargs else "bubble.png"

    # get model
    model = load_model_from_tf(model_dir)

    # get test data.
    # test_lables are to be set as ground truth.
    test_dataset = RepairDataset.load_dataset_from_hdf(test_dir, test_data)
    test_images, test_labels = test_dataset[0], test_dataset[1]

    def convert_labels(labels):
        """Convert labels.

        convert labels from array representation into number,
        then reconvert them into string to set ticks properly.
        e.g.) [[0,0,...,1],...,[1,...,0]] => ["9",...,"0"]
        :param labels:
        """
        return np.array(list(map(lambda label: label.argmax(), labels)))

    ground_truth = convert_labels(test_labels)

    # get predicted labels
    results = model.predict(test_images, verbose=0)
    pred_labels = convert_labels(results)

    # plot confusion matrix as bubble chart
    plot_bubble_chart(ground_truth, pred_labels, output_dir / filename)
