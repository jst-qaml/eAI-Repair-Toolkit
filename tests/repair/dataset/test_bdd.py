import itertools
import json
import random
import string

import h5py
import numpy as np

import pytest
from PIL import Image

from repair.dataset.bdd import BDD

pytestmarks = pytest.mark.usefixtures("fix_seed")


def generate_dummy_img():
    rng = np.random.default_rng(seed=0)
    image = rng.integers(0, 255, size=(90, 160, 3), dtype=np.uint8)
    return image


def generate_img_name():
    p1 = "".join(random.choices(string.hexdigits[:16], k=8))
    p2 = "".join(random.choices(string.hexdigits[:16], k=8))
    return f"{p1}-{p2}.jpg"


def generate_dummy_label(img_name, weather, scene):
    label = {
        "name": img_name,
        "attributes": {
            "weather": weather,
            "scene": scene,
            "timeofday": "",
        },
        "timestamp": "",
        "labels": [
            {
                "category": {
                    "attributes": {
                        "occuluded": True,
                        "truncated": True,
                        "trafficLightColor": "none",
                    },
                },
                "manualshape": True,
                "manualAttributes": True,
                "box2d": {
                    "x1": 0,
                    "x2": 0,
                    "y1": 0,
                    "y2": 0,
                },
                # or
                # "poly2d": [
                #     {
                #         "vertices": [
                #             [ 0.0, 0.0],
                #         ],
                #     },
                # ],
                "id": 23,
            }
        ],
    }

    return label


@pytest.fixture(scope="module")
def bdd_dir(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp("bdd")
    imgs_root = root_dir / "images" / "100k"
    imgs_root.mkdir(parents=True)

    train_imgs_dir = imgs_root / "train"
    train_imgs_dir.mkdir()

    val_imgs_dir = imgs_root / "val"
    val_imgs_dir.mkdir()

    test_imgs_dir = imgs_root / "test"
    test_imgs_dir.mkdir()

    label_dir = root_dir / "labels"
    label_dir.mkdir()

    # Trim some labels to run tests faster.
    weathers = [
        "rainy",
        "snowy",
        "clear",
        # "overcast", "partly cloudy", "foggy",
    ]
    scenes = [
        "parking lot",
        "residential",
        "highway",
        # "gas stations", "city street", "tunnel"
    ]

    num_imgs_per_label_for_train = 10
    num_imgs_per_label_for_val = 10

    dummy_img = Image.fromarray(generate_dummy_img())

    train_labels = []
    for weather, scene in itertools.product(weathers, scenes):
        for _i in range(num_imgs_per_label_for_train):
            img_name = generate_img_name()
            dummy_img.save(test_imgs_dir / img_name)

            train_labels.append(generate_dummy_label(img_name, weather, scene))
    with open(label_dir / "bdd100k_labels_images_train.json", "w") as f:
        json.dump(train_labels, f)

    val_labels = []
    for weather, scene in itertools.product(weathers, scenes):
        for _i in range(num_imgs_per_label_for_val):
            img_name = generate_img_name()
            dummy_img.save(val_imgs_dir / img_name)

            val_labels.append(generate_dummy_label(img_name, weather, scene))
    with open(label_dir / "bdd100k_labels_images_val.json", "w") as f:
        json.dump(val_labels, f)

    return root_dir


@pytest.fixture(scope="module")
def prepare_dataset_with_weather(bdd_dir, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("outputs")

    divide_rate = 0.2

    BDD().prepare(
        input_dir=bdd_dir,
        output_dir=output_dir,
        divide_rate=divide_rate,
        random_state=0,
    )

    return output_dir


@pytest.fixture(scope="module")
def prepare_dataset_with_scene(bdd_dir, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("outputs")

    divide_rate = 0.2

    bdd = BDD()
    bdd.set_extra_config(target_label="scene")
    bdd.prepare(
        input_dir=bdd_dir,
        output_dir=output_dir,
        divide_rate=divide_rate,
        random_state=0,
    )

    return output_dir


def test_dataset_name():
    assert BDD.get_name() == "bdd"


def test_get_label_map():
    label_map = BDD.get_label_map()

    assert len(label_map.items()) == 6


def test_prepare_dataset_generate_files_for_weather(prepare_dataset_with_weather):
    output_dir = prepare_dataset_with_weather

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


def test_prepare_dataset_generates_repair_data_with_divide_rate_for_weather(
    prepare_dataset_with_weather,
):
    output_dir = prepare_dataset_with_weather

    with h5py.File(output_dir / "train.h5") as hf_train:
        train_shape = hf_train["images"].shape

    with h5py.File(output_dir / "repair.h5") as hf_repair:
        repair_shape = hf_repair["images"].shape

    num_train_dataset = train_shape[0]
    num_repair_dataset = repair_shape[0]

    # relax tolerances to support small dataset
    assert num_repair_dataset / (num_train_dataset + num_repair_dataset) == pytest.approx(
        0.2, abs=1e-2
    )


def test_prepare_dataset_generate_files_for_scene(prepare_dataset_with_scene):
    output_dir = prepare_dataset_with_scene

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


def test_prepare_dataset_generates_repair_data_with_divide_rate_for_scene(
    prepare_dataset_with_scene,
):
    output_dir = prepare_dataset_with_scene

    with h5py.File(output_dir / "train.h5") as hf_train:
        train_shape = hf_train["images"].shape

    with h5py.File(output_dir / "repair.h5") as hf_repair:
        repair_shape = hf_repair["images"].shape

    num_train_dataset = train_shape[0]
    num_repair_dataset = repair_shape[0]

    # relax tolerances to support small dataset
    assert num_repair_dataset / (num_train_dataset + num_repair_dataset) == pytest.approx(
        0.2, abs=1e-2
    )
