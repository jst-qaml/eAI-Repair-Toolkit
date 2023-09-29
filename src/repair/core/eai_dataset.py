# ruff: noqa: PLR0912
"""Class EAIDataset.

This class indicates a wrapper for tfio.Dataset, adding some size and shape hints
we know when we open a hdf5 file.
Also has methods for sampling some positive inputs as numpy array.
"""

import logging
import math

import h5py
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class EAIDataset:
    """class of EAIDataset."""

    def __init__(self, file_name, batch_size=32):
        """Initialize.

        :param file_name:
        :param batch_size:
        """
        hf = h5py.File(file_name, "r")
        self.hf = hf
        self.image_shape = hf["images"].shape
        self.label_shape = hf["labels"].shape
        self.image_type = hf["images"].dtype
        self.label_type = hf["labels"].dtype
        self.keys = list(hf.keys())
        self.file_name = file_name
        self.batch_size = batch_size

    def __len__(self):
        """Returns chunks of images divided by batch_size."""
        return int(math.ceil(self.image_shape[0] / self.batch_size))

    def __del__(self):
        """Close file."""
        self.hf.close()

    def __getitem__(self, idx):
        """Get dataset depends on given index."""
        if idx == 0:
            return np.array(self.hf["images"])
        else:
            return np.array(self.hf["labels"])

    def get_generators_split_validation(self, validation_size=0.2):
        """Get generators to lazily read the datasets.

        Splits them into training and validation.

        :param validation_size: proportion of images to be used for validation
        :return: images_train, labels_train, images_validation, labels_validation.
        The images and labels datasets can be zipped together to use with model.fit.
        """
        train_indices = np.arange(len(self))  # one index per batch
        np.random.default_rng().shuffle(train_indices)
        validation_indices = train_indices[
            : int(math.floor(validation_size * len(train_indices)))
        ]
        train_indices = train_indices[
            int(math.floor(validation_size * len(train_indices))) :
        ]
        train_index_handler = EAIIndexHandler(
            self.image_shape[0], self.batch_size, train_indices
        )
        validation_index_handler = EAIIndexHandler(
            self.image_shape[0], self.batch_size, validation_indices
        )

        images_train = self.get_images_generator(train_index_handler)
        labels_train = self.get_labels_generator(train_index_handler)
        images_validation = self.get_images_generator(validation_index_handler)
        labels_validation = self.get_labels_generator(validation_index_handler)

        return images_train, labels_train, images_validation, labels_validation

    def get_generators(self):
        """Get generators to lazily read the datasets.

        Note that this does not split the dataset for validation purposes during training.

        :return: a pair of generators, one for images and one for labels.
        """
        indices = np.arange(len(self))  # one index per batch
        np.random.default_rng().shuffle(indices)
        index_handler = EAIIndexHandler(self.image_shape[0], self.batch_size, indices)

        images_generator = self.get_images_generator(index_handler)
        labels_generator = self.get_labels_generator(index_handler)

        return images_generator, labels_generator

    def get_test_data_generator(self):
        """Get generator of dataset."""
        indices = np.arange(len(self))
        np.random.default_rng().shuffle(indices)
        index_handler = EAIIndexHandler(self.image_shape[0], self.batch_size, indices)
        dataset = tf.data.Dataset.from_generator(
            EAIGeneratorForDataset(index_handler, self.hf["images"], self.hf["labels"]),
            output_signature=(
                tf.TensorSpec(
                    shape=(
                        self.batch_size,
                        self.image_shape[1],
                        self.image_shape[2],
                        self.image_shape[3],
                    ),
                    dtype=self.image_type,
                ),
                tf.TensorSpec(
                    shape=(self.batch_size, self.label_shape[1]), dtype=self.label_type
                ),
            ),
        )

        return dataset

    def get_images_generator(self, index_handler):
        """Get images dataset."""
        images_ds = tf.data.Dataset.from_generator(
            EAIGenerator(index_handler, self.hf["images"]),
            output_types=self.image_type,
            output_shapes=tf.TensorShape(
                [
                    self.batch_size,
                    self.image_shape[1],
                    self.image_shape[2],
                    self.image_shape[3],
                ]
            ),
        )
        return images_ds

    def get_labels_generator(self, index_handler):
        """Get labels dataset."""
        images_ds = tf.data.Dataset.from_generator(
            EAIGenerator(index_handler, self.hf["labels"]),
            output_types=self.label_type,
            output_shapes=tf.TensorShape([self.batch_size, self.label_shape[1]]),
        )
        return images_ds

    def sample_from_file(self, sample_amount):
        """Sample from file with sample_amount.

        To be used when only some inputs from the dataset will be sampled for use.
        The random choice is done in a per image basis, not like the generator
        which randomizes per batch.
        The arrays are sorted per index, so the labels will not be shuffled.

        Todo:
            handle the case if sample_amount is more than the label length

        :param sample_amount: how many images to sample. Size of returned arrays.
        :return: tupple of np.array of images and labels

        """
        # divide the index of the inputs depending on their label
        labels = {}
        batch_size = self.batch_size
        for batch_num in range(len(self)):
            if batch_num == len(self) - 1:  # to avoid out of range
                current_labels = self.hf["labels"][batch_num * batch_size :]
            else:
                current_labels = self.hf["labels"][
                    batch_num * batch_size : (batch_num + 1) * batch_size
                ]

            for idx, label in enumerate(np.argmax(current_labels, axis=1)):
                correct_index = (batch_num * batch_size) + idx
                if label in labels:
                    labels[label].append(correct_index)
                else:
                    labels[label] = [correct_index]

        # dict that storess the amounts of indexes per labels
        label_pools = {label: len(labels[label]) for label in labels}

        # allocate amount per label
        # at least put as many images of each label,
        # as if you put a third of an equitative distribution
        allocation_per_label = {label: 0 for label in labels}
        default_allocation = int((sample_amount / len(labels)) / 3)
        for label, amount in label_pools.items():
            if amount <= default_allocation:
                allocation_per_label[label] += amount
                continue

            allocated_amount = default_allocation + int(
                (2 / 3) * (sample_amount * (amount / self.label_shape[0]))
            )
            allocation_per_label[label] = allocated_amount
            label_pools[label] -= allocated_amount

        # adjust amount
        total = sum(allocation_per_label.values())
        if total != sample_amount:
            remain = sample_amount - total
            available_labels = [
                label for label, amount in label_pools.items() if amount > 0
            ]
            for picked_label in available_labels:
                available_amount = label_pools[picked_label]
                if available_amount >= remain:
                    allocation_per_label[picked_label] += remain
                    break

                allocation_per_label[picked_label] += available_amount
                label_pools[picked_label] = 0
                remain -= available_amount

        # actual sampling
        sample = np.empty(0, int)
        for label in allocation_per_label:
            while allocation_per_label[label] >= len(labels[label]):
                sample = np.concatenate((sample, labels[label]))
                allocation_per_label[label] -= len(labels[label])
            sample = np.concatenate(
                (
                    sample,
                    np.random.default_rng().choice(
                        labels[label], allocation_per_label[label], replace=False
                    ),
                )
            )

        assert len(sample) == sample_amount, f"{len(sample)=}, {sample_amount=}"
        sample.sort()
        return self.hf["images"][sample], self.hf["labels"][sample]


class EAIIndexHandler:
    """Indices wrapper."""

    def __init__(self, image_amount, batch_size, indices):
        """Set params."""
        self.batch_size = batch_size
        self.indices = indices  # the indices we shuffle and use to access h5 file
        self.image_amount = (
            image_amount  # number of images and labels in total in the h5 file.
        )

    def shuffle(self):
        """Shuffle slices with np."""
        np.random.default_rng().shuffle(self.indices)

    def __len__(self):
        """Returns number of chunks of dataset."""
        return int(math.ceil(self.image_amount / self.batch_size))


class EAIGenerator(tf.keras.utils.Sequence):
    """A generator to iterate over a h5 dataset.

    Never loads the whole dataset, so it can deal with OutOfMemory issues.
    Gets the images or labels in batches from the file, to make I/O faster,
    because of this, when we shuffle or access randomly, we access random batches,
    but that batch of images is always in the same order.
    Gets the indexes from EAI_Index_Handler,
    so images and labels are accessed in the same order.
    """

    def __init__(self, index_handler, dataset):
        """Set params."""
        self.index_handler = index_handler
        self.dataset = dataset

    def __getitem__(self, idx):
        """Get batch number idx from file.

        Parameters
        ----------
        idx: int
            given index.

        Returns
        -------
        np.ndarray
            slice of image dataset

        Notes
        -----
        Accessing the h5 file with slices as done here is faster than
        providing a list or range of indices to fetch.

        https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing

        """
        batch_size = self.index_handler.batch_size
        if idx == len(self) - 1:
            batch = self.dataset[idx * batch_size :]
            batch = np.concatenate((batch, self.dataset[: batch_size - len(batch)]))
        else:
            batch = self.dataset[idx * batch_size : (idx + 1) * batch_size]
        return batch

    def __call__(self):
        """Yield chunk of dataset."""
        for idx in self.index_handler.indices:
            batch = self.__getitem__(idx)
            yield batch

    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        self.index_handler.shuffle()

    def __len__(self):
        """Returns number of chunks divided by batch size."""
        return len(self.index_handler)


class EAIGeneratorForDataset(tf.keras.utils.Sequence):
    """See #153."""

    def __init__(self, index_handler, image_dataset, label_dataset):
        """Set params."""
        self.index_handler = index_handler
        self.image_dataset = image_dataset
        self.label_dataset = label_dataset

    def __getitem__(self, idx):
        """Get batch number idx from file.

        Parameters
        ----------
        idx: int
            given index.

        Returns
        -------
        img_batch : np.ndarray
            slice of image dataset
        label_batch: np.ndarray
            slice of label dataset

        Notes
        -----
        Accessing the h5 file with slices as done here is faster than
        providing a list or range of indices to fetch.

        https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing

        """
        batch_size = self.index_handler.batch_size
        if idx == len(self) - 1:
            img_batch = self.image_dataset[idx * batch_size :]
            img_batch = np.concatenate(
                (img_batch, self.image_dataset[: batch_size - len(img_batch)])
            )
            label_batch = self.label_dataset[idx * batch_size :]
            label_batch = np.concatenate(
                (label_batch, self.label_dataset[: batch_size - len(label_batch)])
            )
        else:
            img_batch = self.image_dataset[idx * batch_size : (idx + 1) * batch_size]
            label_batch = self.label_dataset[idx * batch_size : (idx + 1) * batch_size]
        return img_batch, label_batch

    def __call__(self):
        """Yield tuple of image and label."""
        for idx in self.index_handler.indices:
            batch = self.__getitem__(idx)
            yield batch

    def __len__(self):
        """Returns number of chunks divided by batch size."""
        return len(self.index_handler)
