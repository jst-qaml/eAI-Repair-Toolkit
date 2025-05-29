from logging import getLogger

import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow import keras

import pytest

from repair.core.model import target
from repair.testing import get_cache_root

logger = getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def fix_seed():
    import random

    random.seed(0)
    tf.random.set_seed(0)


@pytest.fixture(scope="session")
def fixed_nprng():
    return np.random.default_rng(0)


def load_fashion_mnist_data():
    """Returns raw fashion-mnist dataset via keras api."""
    return keras.datasets.fashion_mnist.load_data()


@pytest.fixture(scope="session")
def raw_fashion_mnist_data():
    return load_fashion_mnist_data()


@pytest.fixture(scope="session")
def global_fashion_mnist_repair_data_dir(tmp_path_factory):
    """Generate global extracted fashiom-mnist dataset for testing.

    This fixture create base extracted dataset for testing.
    You should use `fashion_mnist_repair_data_dir` instead of this fixture for each
    unit tests to avoid unexpected results caused by polluting cache directory.

    Returns
    -------
    Path
        A path to directory where each sub-datasets are saved.
        This directory contains 'train.h5', 'test.h5' and 'repair.h5'.

    Notes
    -----
    Images are reshaped to 3-channels shape to normalize.
    Labels are converted to one-hot vectors.

    """
    dataset_dir = get_cache_root() / "datasets" / "prepared" / "fashion_mnist"
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

    if (
        (dataset_dir / "train.h5").exists()
        and (dataset_dir / "repair.h5").exists()
        and (dataset_dir / "test.h5").exists()
    ):
        return dataset_dir

    logger.info("Cached prepared dataset not found.")

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_data()

    x_train = np.stack([x_train] * 3, axis=-1)
    y_train = to_categorical(y_train, 10)

    x_test = np.stack([x_test] * 3, axis=-1)
    y_test = to_categorical(y_test, 10)

    train_data = (x_train[:5000], y_train[:5000])
    repair_data = (x_train[5000:6000], y_train[5000:6000])
    test_data = (x_test[:1000], y_test[:1000])

    with h5py.File(dataset_dir / "train.h5", "w") as hf:
        hf.create_dataset("images", data=train_data[0])
        hf.create_dataset("labels", data=train_data[1])

    with h5py.File(dataset_dir / "repair.h5", "w") as hf:
        hf.create_dataset("images", data=repair_data[0])
        hf.create_dataset("labels", data=repair_data[1])

    with h5py.File(dataset_dir / "test.h5", "w") as hf:
        hf.create_dataset("images", data=test_data[0])
        hf.create_dataset("labels", data=test_data[1])

    return dataset_dir


@pytest.fixture(scope="function")
def fashion_mnist_repair_data_dir(tmp_path, global_fashion_mnist_repair_data_dir):
    """Provides dataset for testing.

    Returns
    -------
    Path
        A path to directory where each sub-datasets are saved.
        This directory contains 'train.h5', 'test.h5' and 'repair.h5'.

    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    (dataset_dir / "train.h5").symlink_to(global_fashion_mnist_repair_data_dir / "train.h5")
    (dataset_dir / "test.h5").symlink_to(global_fashion_mnist_repair_data_dir / "test.h5")
    (dataset_dir / "repair.h5").symlink_to(global_fashion_mnist_repair_data_dir / "repair.h5")

    return dataset_dir


@pytest.fixture(scope="session")
def fashionmnist_shape():
    return (28, 28, 3), 10


@pytest.fixture(scope="session")
def imagenet_shape():
    return (224, 224, 3), 1000


@pytest.fixture(scope="session")
def pretrained_keras_model_dir(fix_seed, global_fashion_mnist_repair_data_dir, fashionmnist_shape):
    """Pretrained keras model.

    Provides tiny sequential keras model trained with the subset of Fashion-MNIST.
    The model will be in the `<cache_root>/models/testing`.

    Returns
    -------
    model_dir : Path
        A path to the root directory where the model is stored.

    """
    model_dir = get_cache_root() / "models" / "testing"
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    if any(model_dir.iterdir()):
        return model_dir

    logger.info("Cached model not found. Model will be trained.")

    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=fashionmnist_shape[0]
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    with h5py.File(global_fashion_mnist_repair_data_dir / "train.h5") as hf:
        x_train, y_train = hf["images"][:].astype("float32") / 255.0, hf["labels"][:]

        model.fit(x_train, y_train, epochs=2, verbose=0)

        model.save(model_dir)

    return model_dir


@pytest.fixture(scope="session")
def repaired_keras_model_dir(fix_seed, pretrained_keras_model_dir):
    """Provides repaired model by retraining.

    This fixture provides a presudo-repaired model through retraining.

    Returns
    -------
    Path
        A path to the root directory where the repaired model is stored.

    """

    model_dir = get_cache_root() / "models" / "repaired"
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    if any(model_dir.iterdir()):
        return model_dir

    logger.info("Cache not found. Model will be trained.")

    model = keras.models.load_model(pretrained_keras_model_dir)

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_data()
    x_train = np.stack([x_train] * 3, axis=-1).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)

    x_test = np.stack([x_test] * 3, axis=-1).astype("float32") / 255.0
    y_test = to_categorical(y_test, 10)

    train_data = (x_train[7000:9000], y_train[7000:9000])
    test_data = (x_test[1000:2000], y_test[1000:2000])

    model.fit(train_data[0], train_data[1], epochs=2, validation_data=test_data, verbose=0)

    model.save(model_dir)

    return model_dir


@pytest.fixture(scope="session")
def global_targeted_data_dir(
    pretrained_keras_model_dir, global_fashion_mnist_repair_data_dir, tmp_path_factory
):
    """Generate base targeted dataset for testing.

    This fixture create base targeted dataset for testing.
    You should use `targeted_data_dir` instead of this fixture for each
    unit tests to avoid unexpected results caused by polluting cache directory.

    Returns
    -------
    Path
        A path to directory where each sub-datasets are saved.

    """
    data_dir = tmp_path_factory.mktemp("targeted")
    (data_dir / "train.h5").symlink_to(global_fashion_mnist_repair_data_dir / "train.h5")
    (data_dir / "test.h5").symlink_to(global_fashion_mnist_repair_data_dir / "test.h5")
    (data_dir / "repair.h5").symlink_to(global_fashion_mnist_repair_data_dir / "repair.h5")

    target(pretrained_keras_model_dir, data_dir, batch_size=32)

    return data_dir


@pytest.fixture(scope="function")
def targeted_data_dir(global_targeted_data_dir, tmp_path):
    data_dir = tmp_path / "targeted"
    data_dir.mkdir()

    (data_dir / "negative").symlink_to(
        global_targeted_data_dir / "negative", target_is_directory=True
    )
    (data_dir / "positive").symlink_to(
        global_targeted_data_dir / "positive", target_is_directory=True
    )

    return data_dir
