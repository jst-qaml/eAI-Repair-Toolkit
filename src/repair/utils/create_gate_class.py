"""Create dataset for training gate model of HydraNet."""
import json

import h5py
import numpy as np


def run(*, hydra_setting_file: str, data_dir: str):
    """Create classes for gate in hydranet.

    Parameters
    ----------
    hydra_setting_file : str
        A path to the settings file.
    data_dir : str
        A path to the directory of the datasets exist.

    """
    if hydra_setting_file is None:
        raise ValueError("'hydra_settings_file' is required.")

    if data_dir is None:
        raise ValueError("'data_dir' is required.")

    with open(hydra_setting_file) as f:
        hydra_subtask = json.load(f)

    _create_gate_file(data_dir, "train.h5", hydra_subtask)
    _create_gate_file(data_dir, "test.h5", hydra_subtask)
    _create_gate_file(data_dir, "repair.h5", hydra_subtask)


def _create_gate_file(data_dir, target_file, hydra_subtask):
    # NOTE: This function is temporally reverted due to the error caused by generator.
    print(f"=====creating gate class for {target_file}=========")
    hf = h5py.File(data_dir / target_file)
    images, labels = hf["images"][:], hf["labels"][:]
    new_labels = _create_new_labels(labels, hydra_subtask)

    gate_dir = data_dir / "gate"
    if not gate_dir.exists():
        gate_dir.mkdir()

    with h5py.File(gate_dir / target_file, "w") as gatehf:
        gatehf.create_dataset("images", data=images)
        gatehf.create_dataset("labels", data=new_labels)

    hf.close()


def _create_new_labels(labels, hydra_subtask):
    new_labels = np.zeros((labels.shape[0], len(hydra_subtask)))
    for label_idx, label in enumerate(labels):
        idx = np.argmax(label)
        for branch_idx, branch in enumerate(hydra_subtask):
            if idx in branch:
                new_labels[label_idx][branch_idx] = 1
                break
    return new_labels
