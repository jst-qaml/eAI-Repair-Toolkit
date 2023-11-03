"""Prepare CIFAR-10."""

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from skimage import color, exposure

from repair.core.dataset import RepairDataset


def prepare(output_path, divide_rate, random_state):
    """Prepare.

    :param output_path:
    :param divide_rate:
    :param random_state:
    :return:
    """
    (
        train_images,
        train_labels,
        repair_images,
        repair_labels,
        test_images,
        test_labels,
    ) = _get_images_and_labels(divide_rate, random_state)

    RepairDataset.save_dataset_as_hdf(train_images, train_labels, output_path / "train.h5")
    RepairDataset.save_dataset_as_hdf(repair_images, repair_labels, output_path / "repair.h5")
    RepairDataset.save_dataset_as_hdf(test_images, test_labels, output_path / "test.h5")


def _get_images_and_labels(divide_rate, random_state, classes=10):
    (images, labels), (test_images, test_labels) = cifar10.load_data()

    processed_images = []
    processed_labels = []
    for i in range(len(images)):
        processed_images.append(_preprocess_img(images[i]))
        processed_labels.append(to_categorical(labels[i][0], num_classes=classes))

    processed_test_images = []
    processed_test_labels = []
    for i in range(len(test_images)):
        processed_test_images.append(_preprocess_img(test_images[i]))
        processed_test_labels.append(to_categorical(test_labels[i][0], num_classes=classes))

    train_dataset, repair_dataset = RepairDataset.divide_train_dataset(
        processed_images, processed_labels, divide_rate, random_state
    )
    train_images, train_labels = train_dataset
    repair_images, repair_labels = repair_dataset

    return (
        np.array(train_images, dtype="float32"),
        np.array(train_labels, dtype="uint8"),
        np.array(repair_images, dtype="float32"),
        np.array(repair_labels, dtype="float32"),
        np.array(processed_test_images, dtype="float32"),
        np.array(processed_test_labels, dtype="uint8"),
    )


def _preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    return img
