import json
import random
import string

import h5py
import numpy as np

import pytest
from PIL import Image

from repair.dataset.bdd_objects import BDDObjects

pytestmarks = pytest.mark.usefixtures("fix_seed")


def generate_dummy_img():
    rng = np.random.default_rng(seed=0)
    image = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    return image


def generate_img_id():
    _id = 0
    while True:
        yield _id
        _id += 1


def generate_img_name():
    p1 = "".join(random.choices(string.hexdigits[:16], k=8))
    p2 = "".join(random.choices(string.hexdigits[:16], k=8))
    return f"{p1}-{p2}.jpg"


def generate_dummy_label(img_name, label, area=1):
    return {
        "name": img_name,
        "label": label,
        "value_mean": 0.0,
        "area": area,
    }


def generate_dummy_image_info(image_info):
    image_info = {
        "args": {
            "num": 0,
            "category_min": 0,
            "random_state": 0,
            "resize_to": None,
            "excluded_labels": "",
        },
        "results": {
            "car": 0,
            "rider": 0,
            "motorcycle": 0,
            "bus": 0,
            "other vehicle": 0,
            "truck": 0,
            "pedestrian": 0,
            "traffic sign": 0,
            "train": 0,
            "bicycle": 0,
            "traffic light": 0,
        },
        "images": image_info,
    }

    return image_info


@pytest.fixture(scope="module")
def bdd_objects_dir(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp("outputs")
    train_root = root_dir / "train"
    train_root.mkdir()

    val_root = root_dir / "val"
    val_root.mkdir()

    labels = [
        "car",
        "rider",
        "motorcycle",
        "bus",
        "other vehicle",
        "truck",
        "pedestrian",
        "traffic sign",
        "train",
        "bicycle",
        "traffic light",
    ]

    num_samples_for_train_per_labels = 10
    num_samples_for_val_per_labels = 10

    train_img_info = []
    img_id = generate_img_id()
    dummy_img = Image.fromarray(generate_dummy_img())
    for label in labels:
        for _i in range(num_samples_for_train_per_labels):
            imgname = f"{next(img_id)}_{generate_img_name()}"
            dummy_img.save(train_root / imgname)
            train_img_info.append(generate_dummy_label(imgname, label))

    with open(train_root / "image_info.json", "w") as f:
        json.dump(generate_dummy_image_info(train_img_info), f)

    val_img_info = []
    for label in labels:
        for _i in range(num_samples_for_val_per_labels):
            imgname = f"{next(img_id)}_{generate_img_name()}"
            dummy_img.save(val_root / imgname)
            val_img_info.append(generate_dummy_label(imgname, label))

    with open(val_root / "image_info.json", "w") as f:
        json.dump(generate_dummy_image_info(val_img_info), f)

    return root_dir


@pytest.fixture(scope="module")
def prepare_dataset(bdd_objects_dir, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("outputs")

    bddobjects = BDDObjects()
    bddobjects.set_extra_config(data_ratio=(0.7, 0, 0.2, 0.1))
    bddobjects.prepare(
        input_dir=bdd_objects_dir,
        output_dir=output_dir,
        divide_rate=0,
        random_state=0,
    )

    return output_dir


def test_dataset_name():
    assert BDDObjects.get_name() == "bdd-objects"


def test_get_label_map():
    label_map = BDDObjects.get_label_map()

    assert len(label_map.items()) == 13


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


def test_prepare_dataset_generates_repair_data_with_data_ratio(prepare_dataset):
    output_dir = prepare_dataset

    with h5py.File(output_dir / "train.h5") as hf_train:
        train_shape = hf_train["images"].shape

    with h5py.File(output_dir / "repair.h5") as hf_repair:
        repair_shape = hf_repair["images"].shape

    with h5py.File(output_dir / "test.h5") as hf_test:
        test_shape = hf_test["images"].shape

    num_train_dataset = train_shape[0]
    num_repair_dataset = repair_shape[0]
    num_test_dataset = test_shape[0]

    assert num_repair_dataset / (
        num_train_dataset + num_repair_dataset + num_test_dataset
    ) == pytest.approx(0.1)
