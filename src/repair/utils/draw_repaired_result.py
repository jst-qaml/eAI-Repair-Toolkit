"""Utility function: draw bubble chart of repaired results."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf


def run(**kwargs):
    """Draw repaired result.

    :param dataset:
    :param kwargs:
    """
    if "model_dir" in kwargs:
        model_dir = Path(kwargs["model_dir"])
    else:
        raise TypeError("Require --model_dir")

    if "target_dir" in kwargs:
        target_dir = Path(kwargs["target_dir"])
    else:
        raise TypeError("Require --target_dir")

    if "output_dir" in kwargs:
        output_dir = Path(kwargs["output_dir"])
    else:
        output_dir = Path("outputs")

    if "target_data" in kwargs:
        target_data = kwargs["target_data"]
        if not target_data.endswith(".h5"):
            raise TypeError("File type must be '.h5'")
    else:
        target_data = "repair.h5"

    filename = kwargs["filename"] if "filename" in kwargs else "repaired.png"

    repaired_model = load_model_from_tf(model_dir)

    negative_label_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir()])

    negative_label_names = [label_dir.name for label_dir in negative_label_dirs]

    def convert_labels(labels):
        """Convert labels.

        convert labels from array representation into number,
        then reconvert them into string to set ticks properly.
        e.g.) [[0,0,...,1],...,[1,...,0]] => [9,...,0]
        :param labels:
        """
        return np.array(list(map(lambda label: label.argmax(), labels)))

    results = {}
    for negative in negative_label_names:
        # discart test labels
        test_dataset = RepairDataset.load_dataset_from_hdf(
            target_dir / negative, target_data
        )
        test_images, _ = test_dataset[0], test_dataset[1]

        pred_results = repaired_model.predict(test_images, verbose=0)
        pred_results_index = convert_labels(pred_results)
        total_elms = len(pred_results_index)

        pred_results_summary = Counter(pred_results_index)
        results[negative] = dict(
            [(k, v / total_elms * 500) for k, v in pred_results_summary.items()]
        )

    df = pd.DataFrame.from_dict(results)
    df.fillna(0)
    x, y = np.meshgrid(df.columns.values, df.index.values)
    z = df.values.flatten()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x=x.flatten(), y=y.flatten(), s=z)

    ax.set_xlabel("negative labels")
    ax.set_xticks(df.columns.values)
    ax.set_ylabel("predicted labels")
    ax.set_yticks(df.index.values)
    ax.margins(0.1)
    fig.savefig(output_dir / filename, format="png", dpi=300)

    plt.close(fig)
