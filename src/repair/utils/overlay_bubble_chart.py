"""Utility function: test a model and display results as a bubble chart."""

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf

from .plot_bubble_chart import plot_each_bubble_chart


def run(
    *,
    target_label: str,
    model_dir: str,
    model_dir_overlay: str,
    model_dir_overlay2: str = None,
    legend: str = "Inputs (base)",
    legend_overlay: str = "Inputs (overlay)",
    legend_overlay2: str = "Inputs (overlay2)",
    test_dir: str,
    test_data: str = "test.h5",
    output_dir: str = "outputs",
    filename: str = "bubble.png",
):
    """Draw bubble chart.

    Parameters
    ----------
    target_label : list(str)
        Target labels to draw chart.
    model_dir : str
        A path to the directory where the base model for ground truth exists.
    model_dir_overlay : str
        A path to the directory where the comparison model exists.
    model_dir_overlay2 : str, default=None
        A path to the directory where the extra comparison modelexists.
    legend : str, default="Inputs (base)"
        A legend for ground truth.
    legend_overlay : str, default="Inputs (overlay)"
        A legend for comparison data.
    legend_overlay2 : str, default="Inputs (overlay2)"
        A legend for extra comparison data.
    test_dir : str
        A path to the directory where the 'test_data' exists.
    test_data : str, default="test.h5"
        A file name of the test dataset.
    output_dir : str, default="outputs"
        A path to the directory where the chart will be generated.
    filename : str, default="radar.png"
        A file name of the generated image.

    """
    model, model_overlay, model_overlay2 = _get_models(
        model_dir=model_dir,
        model_dir_overlay=model_dir_overlay,
        model_dir_overlay2=model_dir_overlay2,
    )
    test_images, test_labels = _get_test_data(test_dir=test_dir, test_data=test_data)

    legends = []
    if model_overlay2 is None:
        legends = [legend, legend_overlay]
    else:
        legends = [legend, legend_overlay, legend_overlay2]

    if target_label is not None:
        target_label = [str(num) for num in target_label]

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

    # get predicted labels
    results = model.predict(test_images, verbose=0)
    results_overlay = model_overlay.predict(test_images, verbose=0)
    if model_overlay2 is not None:
        results_overlay2 = model_overlay2.predict(test_images, verbose=0)
    else:
        results_overlay2 = None

    pred_labels = labels_from_categorical(results)
    pred_labels_overlay = labels_from_categorical(results_overlay)
    if model_overlay2 is not None:
        pred_labels_overlay2 = labels_from_categorical(results_overlay2)

    fig = plt.figure()

    # plot confusion matrix as bubble chart
    ax = plot_each_bubble_chart(fig, ground_truth, pred_labels, target_label)
    plot_each_bubble_chart(fig, ground_truth, pred_labels_overlay, target_label, "h")

    if model_overlay2 is not None:
        plot_each_bubble_chart(fig, ground_truth, pred_labels_overlay2, target_label, "*")

    ax.legend(legends, bbox_to_anchor=(0, 1.15), fontsize=12, loc="upper left", borderaxespad=0)

    _save_fig(fig, output_dir=output_dir, filename=filename)


def _get_models(*, model_dir, model_dir_overlay, model_dir_overlay2):
    if model_dir is None:
        raise ValueError("'model_dir' is required.")
    model_dir = Path(model_dir)

    if model_dir_overlay is None:
        raise ValueError("'model_dir_overlay' is required.")
    model_dir_overlay = Path(model_dir_overlay)

    if model_dir_overlay2 is not None:
        model_dir_overlay2 = Path(model_dir_overlay2)

    # get models
    model = load_model_from_tf(model_dir)
    model_overlay = load_model_from_tf(model_dir_overlay)
    if model_dir_overlay2 is not None:
        model_overlay2 = load_model_from_tf(model_dir_overlay2)
    else:
        model_overlay2 = None

    return model, model_overlay, model_overlay2


def _get_test_data(*, test_dir, test_data):
    if test_dir is None:
        raise ValueError("'test_dir' is required.")

    if not test_data.endswith(".h5"):
        test_data = f"{test_data}.h5"

    # get test data.
    # test_lables are to be set as ground truth.
    test_dataset = RepairDataset.load_test_data(test_dir)
    test_images, test_labels = test_dataset[0], test_dataset[1]

    return test_images, test_labels


def _save_fig(fig, *, output_dir="outputs", filename):
    output_dir = Path(output_dir)
    fig.savefig(output_dir / filename, format="png", dpi=600)
