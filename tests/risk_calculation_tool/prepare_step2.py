"""Generates a dataset in the same format of the BDD format.

This script need to execute with conda environemt of `eai-repair`.
Before execute this, execute `prepare_step1.py` with `extract-object` environment.

```py
# execute example
conda activate eai-repair
python -m tests.risk_calculation_tool.prepare_step2
```
"""
import shutil

from repair.dataset.bdd_objects.prepare import _get_labels, save_images

from .util import CREATE_IMAGE_SUBSET_DIR, H5_DATASET_DIR


def _main():
    if H5_DATASET_DIR.exists():
        shutil.rmtree(H5_DATASET_DIR)
    H5_DATASET_DIR.mkdir()
    image_paths = list(CREATE_IMAGE_SUBSET_DIR.glob("val/*.jpg"))
    all_labels = _get_labels(CREATE_IMAGE_SUBSET_DIR / "val/image_info.json")
    save_images(image_paths, "test", 32, 32, all_labels, H5_DATASET_DIR, 13)


if __name__ == "__main__":
    _main()
