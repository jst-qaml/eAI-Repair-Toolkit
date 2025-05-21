# ruff: noqa: B027
"""Metaclass of model.

This class indicates a list of methods
to be implemented in concrete model classes.
"""

import json
import os
import shutil
from abc import abstractmethod
from pathlib import Path

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)

from repair.core import eai_dataset
from repair.core._base import RepairClass
from repair.core.dataset import RepairDataset


class RepairModel(RepairClass):
    """Meta class of model."""

    @classmethod
    def get_name(cls) -> str:
        """Returns name of this class."""
        return cls.__name__

    @abstractmethod
    def compile(self, input_shape, output_shape):
        """Configure model for training."""
        pass

    def set_extra_config(self, **kwargs):
        """Set extra config after generating this object.

        :param kwargs:

        TODO
        ----
        Find way to pass B027
        """
        pass


def train(
    model,
    epochs=50,
    data_dir=Path("outputs"),
    output_dir=Path("outputs"),
    gpu=False,
):
    """Train repair model with train dataset.

    Parameters
    ----------
    model : repair.model.RepairModel
        Target model class
    epochs : int, default=50
        Number of epochs to train model
    gpu : bool, default=False
        Flag whether to use GPU for trainig or not
    data_dir : Path, default=Path("outputs")
        Path to directory containig dataset
    output_dir : Path, default=Path("outputs")
        Path to directory to save trained model

    """
    os.environ["H5PY_DEFAULT_READONLY"] = "1"

    # GPU settings
    if gpu:
        config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True, per_process_gpu_memory_fraction=0.8
            )
        )
        session = tf.compat.v1.Session(config=config)
        set_session(session)

    # callbacks
    mc_path = output_dir / "logs" / "model_check_points"
    mc_path.mkdir(parents=True, exist_ok=True)
    tb_path = output_dir / "logs" / "tensor_boards"
    tb_path.mkdir(parents=True, exist_ok=True)

    weight_path = mc_path / "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    model_check_point = ModelCheckpoint(
        filepath=str(weight_path), save_best_only=True, save_weights_only=True
    )
    tensor_board = TensorBoard(log_dir=str(tb_path))
    early_stop = EarlyStopping(monitor="val_loss", patience=5)

    def __lr_schedule(epoch, lr=0.01):
        return lr * (0.1 ** int(epoch / 10))

    lr_sc = LearningRateScheduler(__lr_schedule)
    callbacks = [model_check_point, tensor_board, lr_sc, early_stop]
    dataset = eai_dataset.EAIDataset(data_dir / "train.h5")

    # Load Model
    try:
        _model: tf.keras.Model = model.compile(dataset.image_shape[1:], dataset.label_shape[1])
    except IndexError:
        # The case of training non one-hot vector
        _model = model.compile(dataset.image_shape[1:], 1)
    _model.summary()

    # generates the inputs and labels in batches, with shuffling and validation set
    (
        images_train,
        labels_trian,
        images_validation,
        labels_validation,
    ) = dataset.get_generators_split_validation()
    train_generator = tf.data.Dataset.zip((images_train, labels_trian))
    validation_generator = tf.data.Dataset.zip((images_validation, labels_validation))
    _model.fit(
        train_generator,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
    )

    _model.save(output_dir)


def test(model_dir: Path, data_dir: Path, target_data: str, verbose: int = 0, batch_size: int = 32):
    """Test repair model with given dataset.

    Parameters
    ----------
    model_dir : Path
        Path to directory containing model files.
    data_dir : Path
        Path to directory containing dataset.
    target_data : str
        File name of dataset.
    verbose : int, default=0
        The (0, 1, 2) means (silent, progress bar, one line per epoch) mode.
    batch_size : int, default=32
        Size of batch

    """
    model = load_model_from_tf(model_dir)
    # Load test images and labels
    images, labels = RepairDataset.load_dataset_from_hdf(data_dir, target_data)

    # Obtain accuracy as evaluation result of DNN model with test dataset
    score = model.evaluate(images, labels, verbose=verbose, batch_size=batch_size)
    return score


def target(model_dir, data_dir, batch_size):
    """Find target dataset.

    Parameters
    ----------
    model_dir : Path
        Path to directory containing model files
    data_dir : Path
        Path to directory containing target dataset
    batch_size: int
        Batch size of prediction

    """
    # Load DNN model
    model = load_model_from_tf(model_dir)
    # Load test images and labels
    test_images, test_labels = RepairDataset.load_dataset_from_hdf(data_dir, "repair.h5")

    # Predict labels from test images
    print("predict")
    results = model.predict(test_images, verbose=0, batch_size=batch_size)
    # Parse and save predict/test results
    print("parse test")
    successes, failures = _parse_test_results(test_images, test_labels, results)
    print("save positive")
    _save_positive_results(successes, data_dir, "positive")
    _save_negative_results(failures, data_dir, "negative")

    _save_label_data(successes, data_dir / "positive/labels.json")
    _save_label_data(failures, data_dir / "negative/labels.json")


def _parse_test_results(test_images, test_labels, results):
    """Parse test results.

    Parse test results and split them into success and failure datasets.
    Both datasets are dict of list consisted of dict of image and label.
    successes: {0: [{'image': test_image, 'label': test_label}, ...],
                1: ...}
    failures: {0: {1: [{'image': test_image, 'label': test_label}, ...],
                3: [{'image': test_iamge, 'label': test_label}, ...],
                ...},
            1: {0: [{'image': test_image, 'label': test_label}, ...],
                2: [{'image': test_iamge, 'label': test_label}, ...],
                ...}}

    Parameters
    ----------
    test_images :
        Images used to test moel
    test_labels :
        Labels used to test model
    results : list
        Results of prediction

    Returns
    -------
    successed, failure
        Results of successes and failures

    """
    successes = {}
    failures = {}
    dataset_len = len(test_labels)
    for i in range(dataset_len):
        test_image = test_images[i]
        test_label = test_labels[i]
        test_label_index = test_label.argmax()

        result = results[i]
        predicted_label = result.argmax()
        if predicted_label != test_label_index:
            if test_label_index not in failures:
                failures[test_label_index] = {}
            if predicted_label not in failures[test_label_index]:
                failures[test_label_index][predicted_label] = []
            failures[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label}
            )
        else:
            if test_label_index not in successes:
                successes[test_label_index] = []
            successes[test_label_index].append({"image": test_image, "label": test_label})
    return successes, failures


def _save_label_data(data, path):
    """Save label data.

    Create `labels.json` for Athena and save it to given path.

    Parameters
    ----------
    data : list
        Labels to be saved
    path : Path
        Path to save `labels.json`

    Todos
    -----
    Move this functionality to Athena repo.

    """
    summary = {}
    for label in data:
        summary[str(label)] = {
            "repair_priority": 0,
            "prevent_degradation": 0,
        }
    with open(path, "w") as f:
        dict_sorted = sorted(summary.items(), key=lambda x: x[0])
        json.dump(dict_sorted, f, indent=4)


def _create_merged_dataset(dataset):
    """Crerate merged dataset.

    Create merged dataset with all labels.
    given: {0: [{'image': image, 'label': label}, ...],
            1: [{'image': image, 'label': label}, ...],
            ...}

    return [image, ...], [label, ...]

    Parameters
    ----------
    dataset : list
        List of dataset grouped by labels

    Returns
    -------
    imgs, labels : tuple
        Dataset

    """
    imgs = []
    labels = []
    for label in dataset:
        dataset_per_label = dataset[label]
        for data in dataset_per_label:
            imgs.append(data["image"])
            labels.append(data["label"])
    return imgs, labels


def _save_test_result(results, path):
    """Save result for single label dataset to given path.

    Parameters
    ----------
    results : list[dict[list, list]]
        List of data.
    path : Path
        Path of save `repair.h5`

    """
    images, labels = _extract_dataset(results)
    RepairDataset.save_dataset_as_hdf(images, labels, path)


def _save_test_results(results, data_dir):
    """Save results for multi labels datset to given path.

    given: {0: [{'image': image, 'label': label}, ...],
            1: [{'image': image, 'label': label}, ...],
            ...}
    Save to `data_dir`/<label>/repair.h5.

    Parameters
    ----------
    results : dict[int, list[dict[list, list]]]
        List of data grouped by test label
    data_dir : Path
        Path to directory to save results

    """
    for test_label in results:
        # Make directory for each class
        output_dir = data_dir / str(test_label) / "repair.h5"
        output_dir.parent.mkdir(parents=True)

        _save_test_result(results[test_label], output_dir)


def _extract_dataset(dataset):
    """Extract images and labels from datatset.

    Parameters
    ----------
    dataset : list[dict[list, list]]
        List of data. Data consists of image and its label.

    Returns
    -------
    images, labels : tuple[list, list]
        List of extracted images and labels

    """
    images = []
    labels = []
    for result in dataset:
        image = result["image"]
        label = result["label"]
        images.append(image)
        labels.append(label)
    return images, labels


def _save_positive_results(results, data_dir: Path, path):
    """Save positive data.

    Parameters
    ----------
    results :
        Result dataset
    data_dir : Path
        Path to root directory to save result
    path :
        Path to directory relative to `data_dir` to save result

    """
    output_dir = data_dir / path
    _cleanup_dir(output_dir)

    _save_test_results(results, output_dir)

    # create all-in-one dataset
    all_images, all_labels = _create_merged_dataset(results)
    RepairDataset.save_dataset_as_hdf(all_images, all_labels, output_dir / "repair.h5")


def _save_negative_results(results, data_dir: Path, path):
    """Save negative data.

    Parameters
    ----------
    results :
    data_dir : Path
        Path to root directory to save result
    path : Path
        Path to directory relative to `data_dir` to save result

    """
    output_dir = data_dir / path
    _cleanup_dir(output_dir)

    # create each labels repair.h5
    for test_label in results:
        test_label_dir = output_dir / str(test_label)
        test_label_dir.mkdir()
        _save_test_results(results[test_label], test_label_dir)

        # create all-in-one dataset per test label
        images_per_test_label, labels_per_test_label = _create_merged_dataset(results[test_label])
        RepairDataset.save_dataset_as_hdf(
            images_per_test_label,
            labels_per_test_label,
            test_label_dir / "repair.h5",
        )
        _save_label_data(results[test_label], test_label_dir / "labels.json")

    # create all-in-one dataset
    all_imgs = []
    all_labels = []
    for labels in results:
        _imgs, _labels = _create_merged_dataset(results[labels])
        all_imgs.extend(_imgs)
        all_labels.extend(_labels)
    RepairDataset.save_dataset_as_hdf(all_imgs, all_labels, output_dir / "repair.h5")


def _cleanup_dir(path):
    """Clean up given directory.

    Parameters
    ----------
    path : Path
        Path to directory to be cleaned up

    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()


def load_model_from_tf(model_dir: Path):
    """Load model from SaveModel format file.

    Parameters
    ----------
    model_dir : Path
        Path to directory containing keras model files.

    Returns
    -------
    model : tf.keras.Model
        Loaded model

    """
    model = tf.keras.models.load_model(model_dir)
    return model
