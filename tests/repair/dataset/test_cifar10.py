import h5py
import numpy as np

import pytest

from repair.dataset.cifar_10 import CIFAR10

pytestmarks = pytest.mark.usefixtures("fix_seed")


@pytest.fixture(scope="module")
def prepare_dataset(tmp_path_factory, module_mocker):
    # Replace with small dummy datasets to setup faster.
    module_mocker.patch(
        "tensorflow.keras.datasets.cifar10.load_data",
        return_value=(
            (np.zeros((250, 32, 32, 3), dtype=np.uint8), np.zeros((250, 1), dtype=np.uint8)),
            (np.zeros((50, 32, 32, 3), dtype=np.uint8), np.zeros((50, 1), dtype=np.uint8)),
        ),
    )

    output_dir = tmp_path_factory.mktemp("outputs")

    divide_rate = 0.2

    CIFAR10().prepare(
        input_dir=None, output_dir=output_dir, divide_rate=divide_rate, random_state=0
    )

    return output_dir


def test_dataset_name():
    assert CIFAR10.get_name() == "cifar-10"


def test_get_label_map():
    label_map = CIFAR10.get_label_map()

    assert len(label_map.items()) == 10


def test_prepare_dataset_generate_files(prepare_dataset):
    output_dir = prepare_dataset

    assert (output_dir / "train.h5").exists()
    assert (output_dir / "repair.h5").exists()
    assert (output_dir / "test.h5").exists()

    with h5py.File(output_dir / "train.h5") as hf:
        num_images = hf["images"].shape[0]
        num_labels = hf["labels"].shape[0]

        assert (
            num_images == num_labels
        ), "The number of labels and images do not match in 'train.h5'"

    with h5py.File(output_dir / "repair.h5") as hf:
        num_images = hf["images"].shape[0]
        num_labels = hf["labels"].shape[0]

        assert (
            num_images == num_labels
        ), "The number of labels and images do not match in 'repair.h5'"

    with h5py.File(output_dir / "test.h5") as hf:
        num_images = hf["images"].shape[0]
        num_labels = hf["labels"].shape[0]

        assert num_images == num_labels, "The number of labels and images do not match in 'test.h5'"


def test_prepare_dataset_generates_repair_data_with_divide_rate(prepare_dataset):
    output_dir = prepare_dataset

    with h5py.File(output_dir / "train.h5") as hf_train:
        train_shape = hf_train["images"].shape

    with h5py.File(output_dir / "repair.h5") as hf_repair:
        repair_shape = hf_repair["images"].shape

    num_train_dataset = train_shape[0]
    num_repair_dataset = repair_shape[0]

    assert num_repair_dataset / (num_train_dataset + num_repair_dataset) == pytest.approx(
        0.2, abs=1e-2
    )
