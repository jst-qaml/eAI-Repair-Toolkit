"""Prepare GTSRB."""

import csv
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical

from skimage import color, exposure, io, transform

from repair.core.dataset import RepairDataset


def prepare(root_path: Path, output_path: Path, divide_rate, random_state):
    """Prepare.

    :param root_path:
    :param output_path:
    :param divide_rate:
    :param random_state:
    :return:
    """
    if not (path := Path(root_path / "Final_Training/Images")).exists():
        raise FileNotFoundError(f"No such file or directory: {str(path)}")
    if not (path := Path(root_path / "Final_Test/Images")).exists():
        raise FileNotFoundError(f"No such file or directory: {str(path)}")
    (
        train_images,
        train_labels,
        repair_images,
        repair_labels,
    ) = __get_train_images_and_labels(
        divide_rate, random_state, root_path / "Final_Training/Images"
    )
    test_images, test_labels = __get_test_images_and_labels(
        root_path / "Final_Test/Images",
        root_path / "GT-final_test.csv",
    )

    RepairDataset.save_dataset_as_hdf(
        train_images, train_labels, output_path / "train.h5"
    )
    RepairDataset.save_dataset_as_hdf(
        repair_images, repair_labels, output_path / "repair.h5"
    )
    RepairDataset.save_dataset_as_hdf(test_images, test_labels, output_path / "test.h5")


def __get_train_images_and_labels(
    divide_rate,
    random_state,
    root_path: Path,
    target_size=(32, 32),
    classes=43,
    gray=False,
):
    images = []
    labels = []
    all_image_paths = root_path.glob("*/*.ppm")
    all_image_paths = list(all_image_paths)
    np.random.default_rng().shuffle(all_image_paths)

    for img_path in all_image_paths:
        img = __preprocess_img(io.imread(img_path), target_size)
        images.append(img)
        label = __get_train_class(img_path)
        labels.append(to_categorical(label, num_classes=classes))

    train_dataset, repair_dataset = RepairDataset.divide_train_dataset(
        images, labels, divide_rate, random_state
    )
    train_images, train_labels = train_dataset
    repair_images, repair_labels = repair_dataset

    return (
        np.array(train_images, dtype="float32"),
        np.array(train_labels, dtype="uint8"),
        np.array(repair_images, dtype="float32"),
        np.array(repair_labels, dtype="uint8"),
    )


def __preprocess_img(img, target_size):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[
        centre[0] - min_side // 2 : centre[0] + min_side // 2,
        centre[1] - min_side // 2 : centre[1] + min_side // 2,
        :,
    ]

    # rescale to standard size
    img = transform.resize(img, target_size)

    return img


def __get_train_class(path):
    return int(path.parent.name)


def __get_test_images_and_labels(
    root_path, csv_path, target_size=(32, 32), classes=43, gray=False
):
    images = []
    labels = []
    all_image_paths = root_path.glob("*.ppm")

    for img_path, label in zip(
        sorted(all_image_paths, key=lambda x: int(x.stem)), __get_test_class_id(csv_path)
    ):
        img = __preprocess_img(io.imread(img_path), target_size)
        images.append(img)
        labels.append(to_categorical(label, num_classes=classes))
    return np.array(images, dtype="float32"), np.array(labels, dtype="uint8")


def __get_test_class_id(csv_path):
    with open(str(csv_path)) as f:
        rows = csv.reader(f, delimiter=";")
        next(rows)
        for row in rows:
            yield int(row[7])
