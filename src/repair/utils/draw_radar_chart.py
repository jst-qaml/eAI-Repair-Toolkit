# ruff: noqa: PLR0912
"""Utility function: test a model and display results as a radar chart."""

import json
from pathlib import Path

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf

from .plot_polar import plot_polar


def run(**kwargs):
    """Draw radar chart.

    :param kwargs:
    """
    if "input_dir" in kwargs:
        input_dir = Path(kwargs["input_dir"])
    else:
        raise TypeError("Require --input_dir")
    if "output_dir" in kwargs:
        output_dir = Path(kwargs["output_dir"])
    else:
        output_dir = input_dir
    if "model_dir" in kwargs:
        model_dir = Path(kwargs["model_dir"])
    else:
        raise TypeError("Require --model_dir")
    if "target_data" in kwargs:
        target_data = kwargs["target_data"]
        if not target_data.endswith(".h5"):
            raise TypeError("File type must be '.h5'")
    else:
        target_data = r"repair.h5"
    # For radar chart
    min_lim = kwargs["min_lim"] if "min_lim" in kwargs else 0
    max_lim = kwargs["max_lim"] if "max_lim" in kwargs else 100
    filename = kwargs["filename"] if "filename" in kwargs else r"radar.png"

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
