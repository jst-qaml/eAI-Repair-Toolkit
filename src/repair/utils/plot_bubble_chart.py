"""Utility function:plot bubble chart."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_bubble_chart(ground_truth, pred_labels, output_path):
    """Plot bubble chart of confusion matrix.

    Parameters
    ----------
    ground_truth
    pred_labels
    output_path

    """
    # We convert used label lists into string to show ticks properly.
    labels = list(map(lambda label: str(label), np.unique(ground_truth)))
    cm = confusion_matrix(ground_truth, pred_labels, normalize="true")
    df = pd.DataFrame(cm).set_axis(labels, axis="columns").set_axis(labels, axis="index")
    x, y = np.meshgrid(df.columns, df.index)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x=x.flatten(), y=y.flatten(), s=df.values.flatten() * 500)

    ax.set_xticks(df.columns.values)
    ax.set_yticks(df.index.values)
    ax.set_xlabel("predicted labels")
    ax.set_ylabel("ground truth")
    ax.margins(0.1)
    fig.savefig(output_path, format="png", dpi=300)

    plt.close(fig)


def plot_each_bubble_chart(fig, ground_truth, pred_labels, target=None, marker="o"):
    """Plot overlay bubble chart of confusion matrix.

    :param fig:
    :param ground_truth:
    :param pred_labels:
    :param target:
    :param marker:
    """
    if target is not None:
        labels = target
    else:
        labels = list(map(lambda label: str(label), np.unique(ground_truth)))
    preds = list(map(lambda label: str(label), np.unique(pred_labels)))
    cm = confusion_matrix(ground_truth, pred_labels, normalize="true")
    df = pd.DataFrame(cm).set_axis(preds, axis="columns").set_axis(preds, axis="index")
    x, y = np.meshgrid(df.columns, df.filter(items=labels, axis="index").index)
    ax = fig.add_subplot()

    ax.scatter(
        x=x.flatten(),
        y=y.flatten(),
        s=df.filter(items=labels, axis="index").values.flatten() * 500,
        marker=marker,
        alpha=0.5,
    )
    ax.set_xticks(df.columns.values)
    ax.set_yticks(df.index.values)
    ax.set_xlabel("predicted labels")
    ax.set_ylabel("ground truth")
    ax.margins(0.1)

    return ax
