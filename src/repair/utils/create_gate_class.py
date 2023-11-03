"""Create dataset for training gate model of HydraNet."""
import json
import math
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from repair.core.dataset import RepairDataset


def run(**kwargs):
    """Create classes for gate in hydranet."""
    if "hydra_setting_file" in kwargs:
        hydra_setting_file = Path(kwargs["hydra_setting_file"])
        with open(hydra_setting_file) as f:
            hydra_subtask = json.load(f)
    else:
        raise TypeError("Require --hydra_setting_file")

    if "data_dir" in kwargs:
        data_dir = Path(kwargs["data_dir"])
    else:
        raise TypeError("Require --data_dir")

    _create_gate_file(data_dir, "train.h5", hydra_subtask)
    _create_gate_file(data_dir, "test.h5", hydra_subtask)
    _create_gate_file(data_dir, "repair.h5", hydra_subtask)


def _create_gate_file(data_dir, target_file, hydra_subtask):
    print(f"=====creating gate class for {target_file}=========")
    dataset = RepairDataset._load_data(data_dir, target_file)
    images_generator, labels_generator = dataset.get_generators()
    generator = tf.data.Dataset.zip((images_generator, labels_generator))
    iterator = iter(generator)

    gate_dir = data_dir / "gate"
    if not gate_dir.exists():
        gate_dir.mkdir()

    # TODO: use tf api instead EAIDataset
    chunk_size = int(math.ceil(1000 / dataset.batch_size))
    previous_size = 0
    with h5py.File(gate_dir / target_file, "w") as hf:
        chunk_counter = 0
        images_list = []
        labels_list = []
        img_maxshape = None
        for i in tqdm(range(len(dataset))):
            img_batch, lbl_batch = next(iterator)  # each next is a batch, with default size 32.

            try:
                labels_list = labels_list + list(lbl_batch)
                images_list = images_list + list(img_batch)

                chunk_counter += 1
                # at the end of each chunk, pass the list to the h5 file,
                # so we never have a big list.
                if chunk_counter == chunk_size or i == len(dataset) - 1:
                    images_list = np.array(images_list)
                    labels_list = _create_new_labels(labels_list, hydra_subtask)
                    labels_list = np.array(labels_list)

                    if img_maxshape is None:  # if it's the first time, initialize the h5py datasets
                        img_maxshape = (None,) + images_list.shape[1:]
                        lbl_maxshape = (None,) + labels_list.shape[1:]
                        imgs_dset = hf.create_dataset(
                            "images",
                            shape=images_list.shape,
                            maxshape=img_maxshape,
                            chunks=images_list.shape,
                            dtype=images_list.dtype,
                        )  # , compression="gzip") may make ussage much slower
                        lbls_dset = hf.create_dataset(
                            "labels",
                            shape=labels_list.shape,
                            maxshape=lbl_maxshape,
                            chunks=labels_list.shape,
                            dtype=labels_list.dtype,
                        )  # , compression="gzip")
                        imgs_dset[:] = images_list
                        lbls_dset[:] = labels_list
                    else:  # append to the datasets
                        # Resize the dataset to accommodate the next chunk of rows
                        imgs_dset.resize(previous_size + len(images_list), axis=0)
                        lbls_dset.resize(previous_size + len(labels_list), axis=0)

                        # Write the next chunk
                        imgs_dset[previous_size:] = images_list
                        lbls_dset[previous_size:] = labels_list

                    previous_size += len(images_list)
                    chunk_counter = 0
                    images_list = []
                    labels_list = []

            except TypeError as e:
                print(e)
                continue


def _create_new_labels(labels, hydra_subtask):
    new_labels = []
    for label in labels:
        new_label = np.zeros(len(hydra_subtask), dtype=int)
        for idx in range(len(label)):
            if label[idx] == 1:
                for sub_idx in range(len(hydra_subtask)):
                    if idx in hydra_subtask[sub_idx]:
                        new_label[sub_idx] = 1
                        break
        new_labels.append(new_label)
    return new_labels
