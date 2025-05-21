# ruff: noqa: PLR0912
"""Utility function: test a model and display results as a radar chart."""

import json
from pathlib import Path

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf

from .plot_polar import plot_polar


def run(
    *,
    input_dir: str,
    output_dir: str = None,
    model_dir: str,
    target_data: str = "repair.h5",
    filename: str = "radar.png",
    min_lim: int = 0,
    max_lim: int = 100,
):
    """Draw radar chart.

    Parameters
    ----------
    input_dir : str
        A path to the directory where 'target_data' exists.
    output_dir : str|None, default=None
        A path to the directory where the generated images will be saved.
        If it is None, the value of `input_dir` is set.
    model_dir : str
        A path to the directory where the target model is saved.
    target_data : str, default="repair.h5"
        A file name of dataset to test model.
    filename : str, default="radar.png"
        A file name of the generated image.
    min_lim : int, default=0
        Set the minimum radial axis view limit. This value will be passed to pyplot.
    max_lim : int, default=100
        Set the maximum radial axis view limit. This value will be passed to pyplot.

    """
    if input_dir is None:
        raise ValueError("'input_dir' is required.")
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir

    if model_dir is None:
        raise ValueError("'model_dir' is required.")
    model_dir = Path(model_dir)

    if not target_data.endswith(".h5"):
        target_data = f"{target_data}.h5"

    # Load
    model = load_model_from_tf(model_dir)
    test_dataset = RepairDataset.load_dataset_from_hdf(input_dir, target_data)
    test_images, test_labels = test_dataset[0], test_dataset[1]

    summary = {}
    for test_label in test_labels:
        key = test_label.argmax()
        summary[str(key)] = {"success": 0, "failure": 0}

    # Execute
    results = model.predict(test_images, verbose=0)

    # Parse
    for i in range(len(test_labels)):
        test_label = test_labels[i : i + 1]
        test_label_index = test_label.argmax()

        result = results[i : i + 1]

        if result.argmax() == test_label_index:
            current = summary[str(test_label_index)]["success"]
            summary[str(test_label_index)]["success"] = current + 1
        else:
            current = summary[str(test_label_index)]["failure"]
            summary[str(test_label_index)]["failure"] = current + 1
    labels = []
    values = []
    for key in summary:
        labels.append(key)
        success = summary[key]["success"]
        failure = summary[key]["failure"]
        score = (success * 100) / (success + failure)
        summary[key]["score"] = score
        values.append(score)

    # Save
    with open(output_dir / "results.json", "w") as f:
        dict_sorted = sorted(summary.items(), key=lambda x: x[0])
        json.dump(dict_sorted, f, indent=4)

    # Draw
    plot_polar(labels, values, output_dir / filename, min_lim, max_lim)
