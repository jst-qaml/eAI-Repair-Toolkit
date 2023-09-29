"""Prepare BDD."""

import gc
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical

import ijson
from skimage import color, exposure, io, transform
from tqdm import tqdm

from repair.core.dataset import RepairDataset


def prepare(root_path: Path, output_path: Path, divide_rate, random_state, target_label):
    """Prepare.

    :param root_path:
    :param output_path:
    :param divide_rate:
    :param random_state:
    :param target_label:
    :return:
    """
    (
        train_images,
        train_labels,
        repair_images,
        repair_labels,
        test_images,
        test_labels,
    ) = _get_images_and_labels(
        divide_rate,
        random_state,
        root_path / "images/100k",
        root_path / "labels/bdd100k_labels_images_train.json",
        root_path / "labels/bdd100k_labels_images_val.json",
        target_label,
    )

    RepairDataset.save_dataset_as_hdf(
        train_images, train_labels, output_path / "train.h5"
    )
    RepairDataset.save_dataset_as_hdf(
        repair_images, repair_labels, output_path / "repair.h5"
    )
    RepairDataset.save_dataset_as_hdf(test_images, test_labels, output_path / "test.h5")


def _get_images_and_labels(
    divide_rate,
    random_state,
    img_path: Path,
    train_label_path: Path,
    val_label_path: Path,
    target_label,
    target_size_h=90,
    target_size_w=160,
    classes=6,
    gray=False,
):
    images = []
    labels = []

    # Get images
    train_image_paths = list(img_path.glob("train/*.jpg"))
    train_image_paths = np.array(train_image_paths)
    val_image_paths = list(img_path.glob("val/*.jpg"))
    val_image_paths = np.array(val_image_paths)
    all_image_paths = np.hstack((train_image_paths, val_image_paths))
    del val_image_paths
    del train_image_paths
    gc.collect()
    np.random.default_rng().shuffle(all_image_paths)

    # Get labels
    train_labels = _get_labels(train_label_path, target_label)
    train_labels = np.array(train_labels)
    val_labels = _get_labels(val_label_path, target_label)
    val_labels = np.array(val_labels)
    all_labels = np.hstack((train_labels, val_labels))
    del train_labels
    del val_labels
    gc.collect()

    data_count = 0
    for image_path in tqdm(all_image_paths):
        img = _preprocess_img(io.imread(image_path), (target_size_h, target_size_w))
        label = _get_train_class(all_labels, image_path, target_label)
        try:
            labels.append(to_categorical(label, num_classes=classes))
            images.append(img)

            data_count += 1
        except TypeError:
            continue

    test_num = data_count // 4
    train_images = images[test_num:]
    train_labels = labels[test_num:]
    test_images = images[:test_num]
    test_labels = labels[:test_num]

    train_dataset, repair_dataset = RepairDataset.divide_train_dataset(
        train_images, train_labels, divide_rate, random_state
    )
    train_images, train_labels = train_dataset
    repair_images, repair_labels = repair_dataset

    return (
        np.array(train_images, dtype="float32"),
        np.array(train_labels, dtype="uint8"),
        np.array(repair_images, dtype="float32"),
        np.array(repair_labels, dtype="uint8"),
        np.array(test_images, dtype="float32"),
        np.array(test_labels, dtype="uint8"),
    )


def _preprocess_img(img, target_size):
    # Rescale to target size
    img = transform.resize(img, target_size)

    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    return img


def _get_labels(file_path, target_label):
    label_file = open(file_path)
    df = ijson.parse(label_file)
    labels = []

    name = None
    attribute = None
    for prefix, _, value in df:
        if prefix == "item.name" and name is None:
            name = value
        if name is not None and prefix == "item.attributes.%s" % (target_label):
            attribute = value
            labels.append({"name": name, "attribute": attribute})
            name = None
            attribute = None

    label_file.close()
    return labels


def _get_train_class(labels, img_path, target_label):
    img_name = Path(img_path).name

    attribute = None

    for label in labels:
        if label["name"] == img_name:
            attribute = label["attribute"]
            break

    if target_label == "weather":
        attribute = _get_weather_class(attribute)
    elif target_label == "scene":
        attribute = _get_scene_class(attribute)

    return attribute


def _get_weather_class(attribute):
    if attribute == "rainy":
        attribute = 0
    elif attribute == "snowy":
        attribute = 1
    elif attribute == "clear":
        attribute = 2
    elif attribute == "overcast":
        attribute = 3
    elif attribute == "partly cloudy":
        attribute = 4
    elif attribute == "foggy":
        attribute = 5
    else:
        attribute = None

    return attribute


def _get_scene_class(attribute):
    if attribute == "parking lot":
        attribute = 0
    elif attribute == "residential":
        attribute = 1
    elif attribute == "highway":
        attribute = 2
    elif attribute == "gas stations":
        attribute = 3
    elif attribute == "city street":
        attribute = 4
    elif attribute == "tunnel":
        attribute = 5
    else:
        attribute = None

    return attribute
