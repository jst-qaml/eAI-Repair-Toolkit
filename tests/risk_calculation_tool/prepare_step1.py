"""Generates a dataset in the same format of the BDD format.

Need to execute with conda environemt of `extract-object`.
After execute this, execute `prepare_step2.py` with `eai-repair` environment.

```py
# execute example
conda activate extract-object
python -m tests.risk_calculation_tool.prepare_step1
```
"""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from bdd_image_extract.cli import CLI

from .util import (
    CREATE_IMAGE_SUBSET_DIR,
    LABEL_TO_CLASS,
    RESOURCE_DIR,
    SAMPLE_SIZE,
    SCALABEL_FORMAT_LABEL_PATH,
    TIMEOFDAY,
)

IMAGE_FOLDER_PATH = "images/100k/val"


def _main():
    SCALABEL_FORMAT_LABEL_PATH.unlink(missing_ok=True)
    if CREATE_IMAGE_SUBSET_DIR.exists():
        shutil.rmtree(CREATE_IMAGE_SUBSET_DIR)
    with TemporaryDirectory() as dname:
        tmpdir = Path(dname)
        _generate_dummy_bdd_dataset(tmpdir)
        _execute_bdd_image_extract(tmpdir)


def _generate_dummy_bdd_dataset(tmpdir):
    labels = [
        {
            "name": f"{idx}.jpg",
            "attributes": {
                "timeofday": TIMEOFDAY[idx % 2],
            },
            "labels": [
                {
                    "id": str(idx),
                    "attributes": {
                        "key": "value",
                    },
                    "category": LABEL_TO_CLASS[idx % len(LABEL_TO_CLASS)],
                    "box2d": {
                        "x1": 10,
                        "y1": 10,
                        "x2": 90,
                        "y2": 90,
                    },
                },
            ],
        }
        for idx in range(SAMPLE_SIZE)
    ]
    SCALABEL_FORMAT_LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALABEL_FORMAT_LABEL_PATH.write_text(json.dumps(labels, indent=2))
    image_folder = tmpdir / "bdd100k" / IMAGE_FOLDER_PATH
    image_folder.mkdir(parents=True, exist_ok=True)
    for label in labels:
        shutil.copyfile(RESOURCE_DIR / "dummy.jpg", image_folder / label["name"])


def _execute_bdd_image_extract(tmpdir):
    # execute step3.2 and 3.3 of the below document
    # https://github.com/jst-qaml/eAI-Repair/blob/3face652e393eea1580edf9a1a295674af94f62c/docs/bdd_image_extract/README.adoc
    cli = CLI()
    category_dir = tmpdir / "categories"
    category_dir.mkdir()
    cli.create_category_dir(
        str(SCALABEL_FORMAT_LABEL_PATH),
        str(tmpdir / "bdd100k" / IMAGE_FOLDER_PATH),
        str(category_dir),
    )
    CREATE_IMAGE_SUBSET_DIR.mkdir()
    cli.create_image_subset(str(category_dir), str(CREATE_IMAGE_SUBSET_DIR / "val"), resize_to=96)


if __name__ == "__main__":
    _main()
