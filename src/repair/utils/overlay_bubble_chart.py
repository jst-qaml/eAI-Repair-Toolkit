"""Utility function: test a model and display results as a bubble chart."""

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf

from .plot_bubble_chart import plot_each_bubble_chart


def run(**kwargs):
    """Draw bubble chart.

    :param dataset:
    :param kwargs:
    """
    model, model_overlay, model_overlay2 = _get_models(**kwargs)
    test_images, test_labels = _get_test_data(**kwargs)
    legends = _get_legends(model_overlay2, **kwargs)

    if "target_label" in kwargs:
        target_label = [str(num) for num in kwargs["target_label"]]
    else:
        target_label = None

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
    results_overlay = model_overlay.predict(test_images, verbose=0)
    if model_overlay2 is not None:
        results_overlay2 = model_overlay2.predict(test_images, verbose=0)
    else:
        results_overlay2 = None

    pred_labels = convert_labels(results)
    pred_labels_overlay = convert_labels(results_overlay)
    if model_overlay2 is not None:
        pred_labels_overlay2 = convert_labels(results_overlay2)

    fig = plt.figure()

    # plot confusion matrix as bubble chart
    ax = plot_each_bubble_chart(fig, ground_truth, pred_labels, target_label)
    plot_each_bubble_chart(fig, ground_truth, pred_labels_overlay, target_label, "h")

    if model_overlay2 is not None:
        plot_each_bubble_chart(fig, ground_truth, pred_labels_overlay2, target_label, "*")

    ax.legend(
        legends, bbox_to_anchor=(0, 1.15), fontsize=12, loc="upper left", borderaxespad=0
    )

    _save_fig(fig, **kwargs)


def _get_models(**kwargs):
    if "model_dir" in kwargs:
        model_dir = Path(kwargs["model_dir"])
    else:
        raise TypeError("Require --model_dir")

    if "model_dir_overlay" in kwargs:
        model_dir_overlay = Path(kwargs["model_dir_overlay"])
    else:
        raise TypeError("Require --model_dir_overlay")

    if "model_dir_overlay2" in kwargs:
        model_dir_overlay2 = Path(kwargs["model_dir_overlay2"])
    else:
        model_dir_overlay2 = None
    # get models
    model = load_model_from_tf(model_dir)
    model_overlay = load_model_from_tf(model_dir_overlay)
    if model_dir_overlay2 is not None:
        model_overlay2 = load_model_from_tf(model_dir_overlay2)
    else:
        model_overlay2 = None

    return model, model_overlay, model_overlay2


def _get_test_data(**kwargs):
    if "test_dir" in kwargs:
        test_dir = Path(kwargs["test_dir"])
    else:
        raise TypeError("Require --test_dir")

    if "test_data" in kwargs:
        test_data = kwargs["test_data"]
        if not test_data.endswith(".h5"):
            raise TypeError("File type must be '.h5'")
    else:
        test_data = r"test.h5"
    # get test data.
    # test_lables are to be set as ground truth.
    test_dataset = RepairDataset.load_test_data(test_dir, test_data)
    test_images, test_labels = test_dataset[0], test_dataset[1]

    return test_images, test_labels


def _get_legends(model_overlay2, **kwargs):
    legend = kwargs["legend"] if "legend" in kwargs else "Inputs (base)"
    legend_overlay = (
        kwargs["legend_overlay"] if "legend_overlay" in kwargs else "Inputs (overlay)"
    )
    legend_overlay2 = (
        kwargs["legend_overlay2"] if "legend_overlay2" in kwargs else "Inputs (overlay2)"
    )

    # Legend
    if model_overlay2 is None:
        legends = [legend, legend_overlay]
    else:
        legends = [legend, legend_overlay, legend_overlay2]

    return legends


def _save_fig(fig, **kwargs):
    if "output_dir" in kwargs:
        output_dir = Path(kwargs["output_dir"])
    else:
        output_dir = Path("outputs")

    filename = kwargs["filename"] if "filename" in kwargs else "bubble.png"

    fig.savefig(output_dir / filename, format="png", dpi=600)
