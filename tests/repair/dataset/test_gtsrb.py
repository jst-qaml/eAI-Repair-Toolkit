import csv
import random

import h5py
import numpy as np

import pytest
from PIL import Image

from repair.dataset.gtsrb import GTSRB

pytestmarks = pytest.mark.usefixtures("fix_seed")


def generate_dummy_img():
    rng = np.random.default_rng(seed=0)
    image = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    return image


@pytest.fixture(scope="module")
def gtsrb_dir(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp("gtsrb")
    train_dir = root_dir / "Final_Training" / "Images"
    train_dir.mkdir(parents=True)

    test_dir = root_dir / "Final_Test" / "Images"
    test_dir.mkdir(parents=True)

    num_classes = 43
    num_imgs_per_label = 10
    num_test_imgs = 100

    dummy_img = Image.fromarray(generate_dummy_img())

    for label in range(num_classes):
        train_label_dir = train_dir / f"{label:05}"
        train_label_dir.mkdir()

        for i in range(num_imgs_per_label):
            dummy_img.save(train_label_dir / f"{label:05}_{i:05}.ppm")

    for i in range(num_test_imgs):
        dummy_img.save(test_dir / f"{i:05}.ppm")

    with open(root_dir / "GT-final_test.csv", "w", newline="\n") as f:
        w = csv.writer(f, delimiter=";")

        w.writerow(
            ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]
        )
        for i in range(num_test_imgs):
            w.writerow(
                [
                    f"{i:05}.ppm",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    random.randrange(num_classes),
                ]
            )

    return root_dir


@pytest.fixture(scope="module")
def prepare_dataset(gtsrb_dir, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("outputs")

    divide_rate = 0.2

    GTSRB().prepare(
        input_dir=gtsrb_dir,
        output_dir=output_dir,
        divide_rate=divide_rate,
        random_state=0,
    )

    return output_dir


def test_dataset_name():
    assert GTSRB.get_name() == "gtsrb"


def test_get_label_map():
    label_map = GTSRB.get_label_map()

    assert len(label_map.items()) == 43


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

    assert num_repair_dataset / (num_train_dataset + num_repair_dataset) == pytest.approx(0.2)
