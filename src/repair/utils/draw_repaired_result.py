"""Utility function: draw bubble chart of repaired results."""

from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf


def run(
    *,
    model_dir: str,
    target_dir: str,
    target_data: str = "repair.h5",
    output_dir: str = None,
    filename: str = "repaired.png",
):
    """Draw repaired result.

    model_dir : str
        A path to the directory where the target model exists.
    target_dir : str
        A path to the directory where the 'target_data' exists.
    target_data : str, default="repair.h5"
        A file name of the test dataset.
    output_dir : str|None, default=None
        A path to the directory where the generated image will be saved.
        If it is None, the value of `target_data` will be set.
    filename : str, default="repaired.png"
        A file name of the generated image.

    """
    if model_dir is None:
        raise ValueError("'model_dir' is required.")

    if target_dir is None:
        raise ValueError("'target_dir' is required.")

    if not target_data.endswith(".h5"):
        target_data = f"{target_data}.h5"

    if output_dir is None:
        output_dir = target_dir

    repaired_model = load_model_from_tf(model_dir)

    negative_label_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir()])

    negative_label_names = [label_dir.name for label_dir in negative_label_dirs]

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

    results = {}
    for negative in negative_label_names:
        # discard test labels
        test_dataset = RepairDataset.load_dataset_from_hdf(target_dir / negative, target_data)
        test_images, _ = test_dataset[0], test_dataset[1]

        pred_results = repaired_model.predict(test_images, verbose=0)
        pred_results_index = labels_from_categorical(pred_results)
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
