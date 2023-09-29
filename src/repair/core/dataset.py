"""Metaclass of dataset.

This class indicates a list of methods
to be implemented in concrete dataset classes.
"""

import importlib
import random
from abc import ABCMeta, abstractmethod
from pathlib import Path

import h5py

from repair.core import eai_dataset


class RepairDataset(metaclass=ABCMeta):
    """Meta class of dataset."""

    @classmethod
    def get_name(cls) -> str:
        """Returns name of this class."""
        return cls.__name__

    @abstractmethod
    def _get_input_shape(self):
        """Return the input_shape and classes in this function.

        return: two values as below
            1.A three-values tuple like (height, width, color) and
            2.The number of classes as the output result

        """
        pass

    @abstractmethod
    def prepare(self, input_dir, output_dir, divide_rate, random_state):
        """Prepare dataset to train.

        :param input_dir:
        :param output_dir:
        :param divide_rate:
        :param random_state:

        """
        pass

    @classmethod
    def load_test_data(cls, data_dir):
        """Load test data.

        :param data_dir:
        :return:
        """
        return cls._load_data(data_dir, r"test.h5")

    @classmethod
    def load_repair_data(cls, data_dir):
        """Load repair data.

        :param data_dir:
        :return:
        """
        return cls._load_data(data_dir, r"repair.h5")

    @classmethod
    def _load_data(cls, data_dir, target):
        """Load data from the input file.

        :param data_dir:
        :param target:
        :return:
        """
        data_dir = str(data_dir)
        if data_dir[-1] != "/" and target[0] != "/":
            target = "/" + target
        dataset = eai_dataset.EAIDataset(data_dir + target)
        return dataset

    # ruff: noqa
    def set_extra_config(self, **kwargs):
        """Set extra config after generating this object.

        :param kwargs:
        """
        pass

    @staticmethod
    def divide_train_dataset(train_images, train_labels, divide_rate, random_state):
        """Divide the train data into normal training data and repair data.

        Parameters
        ----------
        train_image : np.ndarray
            List of original train images
        train_label : np.ndarray
            List of original train labels
        divide_rate : float, default=0.2
            Fraction of repair dataset
        random_state : int, optional
            Seed value

        Returns
        -------
        (train_images, train_labels), (repair_images, repair_labels)
            Tuple of train and repair dataset.
            Each dataset consist of pair of images and labels.

        """
        if random_state is not None:
            random.seed(int(random_state))

        sample_list = random.sample(
            range(len(train_images)), int(len(train_images) * divide_rate)
        )
        sample_list.sort(reverse=True)
        repair_images = []
        repair_labels = []
        for i in sample_list:
            repair_images.append(train_images.pop(i))
            repair_labels.append(train_labels.pop(i))

        return (train_images, train_labels), (repair_images, repair_labels)

    @staticmethod
    def load_dataset_from_hdf(data_dir: Path, target):
        """Load test dataset.

        Parameters
        ----------
        data_dir : Path
            Path to directory containing dataset

        target : str
            File name of dataset

        """
        with h5py.File(data_dir / target) as hf:
            test_images = hf["images"][()]
            test_labels = hf["labels"][()]

        return test_images, test_labels

    @staticmethod
    def save_dataset_as_hdf(images, labels, path: Path):
        """Save datasets.

        Parameters
        ----------
        path : Path
            Path to file to be saved
        images : np.ndarray
            Images of dataset
        labels : np.ndarray
            Labels of dataset

        """
        with h5py.File(path, "w") as hf:
            hf.create_dataset("images", data=images)
            hf.create_dataset("labels", data=labels)
