"""Requirements-driven Repair of Deep Neural Networks.

athena_test.py

Copyright (c) 2020 Udzuki, Inc.

Released under the BSD license for academic use only.
https://opensource.org/licenses/BSD-3-Clause

For commercial use, contact Udzuki, Inc. | https://www.udzuki.co.jp
"""
import itertools
import json
from pathlib import Path

import keras
import numpy as np

from repair.core.dataset import load_dataset_from_hdf
from repair.core.model import load_model_from_tf
from repair.methods.athena.athena import Athena


def run(kwargs):
    """Athena test with weights and their repaired values in labels.json.

    :param kwargs:
    """
    if "model_dir" in kwargs:
        model_dir = Path(kwargs["model_dir"])
    else:
        raise TypeError("Require --model_dir")
    if "data_dir" in kwargs:
        data_dir = Path(kwargs["data_dir"])
    else:
        raise TypeError("Require --data_dir")
    if "length" in kwargs:
        length = int(kwargs["length"])
    else:
        length = 1

    results = {}
    if data_dir.joinpath(r"athena_results.json").exists():
        with open(data_dir.joinpath(r"athena_results.json")) as f:
            results = {}
            for key in json.load(f):
                label = key[0]
                results[label] = key[1]

    athena = Athena()
    weights_repaired = _load_weights_repaired(data_dir, athena)
    negative_map = _load_data_map(data_dir)
    negative_all_images = []
    negative_all_labels = []
    for lab in negative_map:
        negative_all_images += negative_map[lab][0].tolist()
        negative_all_labels += negative_map[lab][1].tolist()
    negative_all_images = np.array(negative_all_images, dtype="float32")
    negative_all_labels = np.array(negative_all_labels, dtype="float32")

    weights_candidates = _candidates(weights_repaired, length)
    for key in weights_candidates:
        if key in results:
            print(str(key) + " has been already tested")
            continue
        model_repaired = _load_and_mutate(model_dir, weights_candidates[key])
        labels = key.split("_")
        results[key] = {}

        for lab in labels:
            score = model_repaired.evaluate(negative_map[lab][0], negative_map[lab][1])
            results[key][lab] = {}
            results[key][lab]["loss"] = score[0]
            results[key][lab]["accuracy"] = score[1]

        score = model_repaired.evaluate(
            negative_all_images, negative_all_labels, verbose=0
        )
        results[key]["loss"] = score[0]
        results[key]["accuracy"] = score[1]
        print(key + str(score))

        with open(data_dir.joinpath(r"athena_results.json"), "w") as f:
            results_sorted = sorted(results.items(), key=lambda x: x[0])
            json.dump(results_sorted, f, indent=4)


def _candidates(weights_repaired, p_num):
    """Generate the combinations of labels for repairing model if needed.

    :param weights_reaired:
    :param p_num:
    :return:
    """
    label_list = weights_repaired.keys()
    cs = itertools.combinations(label_list, p_num)
    wc = []
    cs_list = list(cs)
    for c in cs_list:
        wc.extend(_permutations_if_needed(weights_repaired, c))
    weights_candidates = {}
    for p in wc:
        weights = []
        keyword = ""
        for key in p:
            keyword += key + "_"
            weights.extend(weights_repaired[key])
        keyword = keyword.rstrip("_")
        weights_candidates[keyword] = weights
    return weights_candidates


def _permutations_if_needed(weights_repaired, c):
    """Generate the permutations of labels for repairing model if needed.

    :param weights_repaired:
    :param c:
    :return:
    """
    weights_dict = {}
    permutations_set = set()
    for key in c:
        ws = weights_repaired[key]
        for w in ws:
            pos = str(w[1]) + "_" + str(w[2]) + "_" + str(w[3])
            if pos not in weights_dict:
                weights_dict[pos] = key
            else:
                permutations_set.add(key)
                permutations_set.add(weights_dict[pos])
    if len(permutations_set) != 0:
        non_permutations_set = set(c) - permutations_set
        permutations_tuple = itertools.permutations(list(permutations_set))
        return [e + tuple(non_permutations_set) for e in permutations_tuple]
    else:
        return [c]


def _load_weights_repaired(target_data_dir, athena, skip_repaired_value=False):
    """Load neural weight candidates.

    :param target_data_dir:
    :param athena:
    :skip_repaired_value:
    :return:
    """
    weights = athena.load_weights(target_data_dir)
    weights_repaired = {}
    for label in weights:
        # Not subject
        if "repair_priority" not in weights[label]:
            weights[label]["repair_priority"] = 0
            continue

        if not weights[label]["repair_priority"]:
            continue

        if "repaired_values" in weights[label] or skip_repaired_value:
            idx = 0
            label_weights = []
            for _weight in weights[label]["weights"]:
                layer_index = _weight[0]
                nw_i = _weight[1]
                nw_j = _weight[2]
                if skip_repaired_value:
                    label_weights.append([-1, layer_index, nw_i, nw_j])
                else:
                    val = np.float32(weights[label]["repaired_values"][idx])
                    idx += 1
                    label_weights.append([val, layer_index, nw_i, nw_j])
            weights_repaired[label] = label_weights
    return weights_repaired


def _load_data_map(data_dir):
    """Load images from the negative files.

    :param data_dir:
    :return:
    """
    negative_map = {}
    for p in data_dir.iterdir():
        if p.is_dir():
            if p.joinpath(r"repair.h5").exists():
                test_images, test_labels = load_dataset_from_hdf(p, r"repair.h5")
                negative_map[p.name] = [test_images, test_labels]
    return negative_map


def _load_and_mutate(model_dir, weights):
    """Load a DNN model and mutate it by weights.

    :param model_dir:
    :param weights:
    :return:
    """
    model = load_model_from_tf(model_dir)
    model = _mutate(model, weights)
    return model


def _mutate(model, locations):
    """Mutate model with location.

    :param model:
    :param location:
    :return:
    """
    # Clone model for manipulating neural weights
    # cf. https://stackoverflow.com/questions/54366935/
    model_clone = keras.models.clone_model(model)
    model_clone.compile(optimizer=model.optimizer, loss=model.loss, metrics=["accuracy"])
    model_clone.set_weights(model.get_weights())

    for location in locations:
        # Parse location data
        val = location[0]
        layer_index = location[1]
        nw_i = location[2]
        nw_j = location[3]

        # Set neural weight at given position with given value
        layer = model_clone.get_layer(index=layer_index)
        weights = layer.get_weights()
        weights[0][nw_i][nw_j] = val
        layer.set_weights(weights)

    return model_clone
