# ruff: noqa
"""File: split_repair_dataset.py.

This is a sample script to split repair dataset into 10 of chunked dataset.
You can create a new script or modify this script to meet your requirements.

Examples
--------
> python scripts/split_repair_dataset repair.h5

> tree .
.
│
...
├── repair.h5
├── repairsets
│   ├── 0
│   │   └── repair.h5
│   ├── 1
│   │   └── repair.h5
│   ├── 2
│   │   └── repair.h5
│   ├── 3
│   │   └── repair.h5
│   ├── 4
│   │   └── repair.h5
│   ├── 5
│   │   └── repair.h5
│   ├── 6
│   │   └── repair.h5
│   ├── 7
│   │   └── repair.h5
│   ├── 8
│   │   └── repair.h5
│   └── 9
│       └── repair.h5
...
"""

import sys
from pathlib import Path

import h5py

DEFAULT_CHUNK_SIZE = 100


def run(file: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
    repairset_root = Path(file).parent
    if not repairset_root.exists():
        raise ValueError("Invalid path")

    repairset_root /= "repairsets"
    repairset_root.mkdir(exist_ok=True)

    hf = h5py.File(file)
    dataset_len = len(hf["labels"])
    for i in range(10):
        if i * chunk_size > dataset_len:
            print("program finished with short data.")
            break

        chunk_root = repairset_root / str(i)
        chunk_root.mkdir()

        if (i + 1) * chunk_size > dataset_len:
            chunked_imgs = hf["images"][i * chunk_size :]
            chunked_lbls = hf["labels"][i * chunk_size :]
        else:
            chunked_imgs = hf["images"][i * chunk_size : (i + 1) * chunk_size]
            chunked_lbls = hf["labels"][i * chunk_size : (i + 1) * chunk_size]

        with h5py.File(chunk_root / "repair.h5", "w") as f:
            f.create_dataset("images", data=chunked_imgs)
            f.create_dataset("labels", data=chunked_lbls)


if __name__ == "__main__":
    file = sys.argv[1]
    run(file)
