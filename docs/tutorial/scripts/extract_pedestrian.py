# ruff: noqa
"""File: extract_pedestrian.py.

This file is a sample script to extract pedestrain data from base dataset.
You can create a new script or modify this script to meet your requirements.
This script extracts all pedestrians data, but you may need only first 1000 elements of them.

Examples
--------
> # 6 is a label number for pedestrian in bdd
> python scripts/extract_pedestrian.py repair.h5 6

> ls
... dataset_6.h5

"""

import sys

import h5py
import numpy as np


def main(file: str, label: int):
    print(f"Extract {label} from {file}...")
    hf = h5py.File(file)
    indices = np.where(np.argmax(hf["labels"], axis=1) == label)

    print("extracting labels...")
    ex_labels = hf["labels"][indices]
    print("extracting images...")
    ex_images = hf["images"][indices]

    with h5py.File(f"dataset_{label}.h5", "w") as f:
        f.create_dataset("images", data=ex_images)
        f.create_dataset("labels", data=ex_labels)


if __name__ == "__main__":
    file = sys.argv[1]
    target_label = sys.argv[2]
    main(file, int(target_label))
