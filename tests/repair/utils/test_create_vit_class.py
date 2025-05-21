import h5py
import numpy as np

import pytest

from repair.utils import create_vit_class


@pytest.mark.skip(
    reason="""'create_vit_class' overwrites the train.h5 created by fixture.
This process raises OSError and this utility itself will be useless when #233 is completed."""
)
def test_create_vit_class(fashion_mnist_repair_data_dir, raw_fashion_mnist_data):
    data_dir = fashion_mnist_repair_data_dir

    create_vit_class.run(data_dir=data_dir)

    assert (data_dir / "vision_transformer").exists()
    assert (data_dir / "vision_transformer" / "train.h5").exists()

    with h5py.File(data_dir / "vision_transformer" / "train.h5") as hf:
        labels = hf["labels"][:]
        (_x, expected), _test = raw_fashion_mnist_data
        np.testing.assert_equal(labels, expected[:5000])
