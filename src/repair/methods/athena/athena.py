"""Requirements-driven Repair of Deep Neural Networks.

athena.py

Copyright (c) 2020 Udzuki, Inc.

Released under the BSD license for academic use only.
https://opensource.org/licenses/BSD-3-Clause

For commercial use, contact Udzuki, Inc. | https://www.udzuki.co.jp
"""
import json
from pathlib import Path

import numpy as np

from tqdm import tqdm

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf
from repair.methods.arachne import arachne


class Athena(arachne.Arachne):
    """Search-based repair extended from Arachne."""

    def __init__(self):
        """Initialize."""
        super().__init__()
        self.min_iteration_range = 10
        self.num_input_pos_sampled = 200
        self.input_protected = None
        self.batch_size = 32

    def set_options(self, **kwargs):
        """Set options."""
        super().set_options(**kwargs)
        if "min_iteration_range" in kwargs:
            self.min_iteration_range = kwargs["min_iteration_range"]
        if "num_input_pos_sampled" in kwargs:
            self.num_input_pos_sampled = kwargs["num_input_pos_sampled"]
            if self.num_input_pos_sampled == "none":
                self.num_input_pos_sampled = None
        if "batch_size" in kwargs:
            self.min_iteration_range = kwargs["batch_size"]

    def localize(self, model, input_neg, target_data_dir: Path, verbose=1):
        """Localize faulty neural weights.

        TODO: Need to evaluate effectiveness.

        Parameters
        ----------
        model : repair.core.model.RepairModel
            DNN model to be repaired
        input_neg : tuple[np.ndarray, np.ndarray]
            A set of inputs that reveal the fault
        target_data_dir : Path, default=Path("outputs")
            Path to directory containing target "labels.json"
        verbose : int, default=1
            Log level

        """
        # Load requirements
        req = self._load_requirements(target_data_dir)
        # Declaration for output results
        req_sorted = None

        # Localize each subset
        reshaped_model = self._reshape_target_model(model, input_neg)
        for label in tqdm(req, desc="Localizing for repair data"):
            # Not subject
            if "repair_priority" not in req[label]:
                req[label]["repair_priority"] = 0
                continue

            if not req[label]["repair_priority"]:
                continue

            # Already computed
            if "weights" in req[label]:
                continue

            # Load each subset
            subset_dir = target_data_dir / str(label)
            input_neg = RepairDataset.load_repair_data(subset_dir)
            # Compute back propagation
            candidates = self._compute_gradient(reshaped_model, input_neg)
            # Compute forward impact
            num_grad = len(input_neg[0]) * 20  # TODO reduce computation
            pool = self._compute_forward_impact(
                reshaped_model, input_neg, candidates, num_grad
            )
            # Extract pareto front
            weights_t = self._extract_pareto_front(pool)

            weights_t = self._modify_layer_before_reshaped(model, weights_t)

            # Store results
            req[label]["weights"] = weights_t

            # Save localization results
            with open(target_data_dir / "labels.json", "w") as f:
                req_sorted = sorted(req.items(), key=lambda x: x[0])
                json.dump(req_sorted, f, indent=4)
                self.output_files.add(target_data_dir / "labels.json")
        if req_sorted is not None:
            self._log_localize(req_sorted, verbose)

    def _load_requirements(self, input_dir: Path):
        """Load requirements.

        Requirements are written in "labels.json"
        under a given directory.

        Parameters
        ----------
        input_dir : Path
            Path to directory containing "labels.json"

        Returns
        -------
        req
            Loaded requirements

        """
        file = input_dir / "labels.json"
        with open(file) as f:
            req = {}
            for key in json.load(f):
                label = key[0]
                req[label] = key[1]
        return req

    def load_input_neg(self, input_dir):
        """Load negative inputs."""
        repair_images = []
        repair_labels = []

        req = self._load_requirements(input_dir)
        for label in tqdm(req, desc="Loading requirements for negative inputs"):
            # Not subject
            if "repair_priority" not in req[label]:
                continue

            if not req[label]["repair_priority"]:
                continue

            subset_dir = input_dir / label
            dataset = RepairDataset.load_repair_data(subset_dir)
            _repair_images = dataset[0]
            _repair_labels = dataset[1]

            # Sampling according to given "repair_priority"
            sampling_rate = req[label]["repair_priority"]
            samples = np.random.default_rng().choice(
                len(_repair_images), len(_repair_images) * sampling_rate
            )
            _repair_images = _repair_images[samples]
            _repair_labels = _repair_labels[samples]

            # Appending sampled data
            for _repair_image in _repair_images:
                repair_images.append(list(_repair_image))
            for _repair_label in _repair_labels:
                repair_labels.append(list(_repair_label))

        if len(repair_images) == 0:
            raise ValueError(
                "No negative dataset is loaded. "
                "Be sure to set appropreate `repair_priority`."
            )

        return np.array(repair_images), np.array(repair_labels)

    def load_input_pos(self, input_dir):
        """Load positive inputs."""
        repair_images = []
        repair_labels = []

        req = self._load_requirements(input_dir)
        for label in tqdm(req, desc="Loading requirements for positive inputs"):
            # Not subject
            if "prevent_degradation" not in req[label]:
                continue

            if not req[label]["prevent_degradation"]:
                continue

            subset_dir = input_dir / label
            dataset = RepairDataset.load_repair_data(subset_dir)
            _repair_images = dataset[0]
            _repair_labels = dataset[1]

            # Sampling according to given "prevent_degradation"
            sampling_rate = req[label]["prevent_degradation"]
            print(f"{sampling_rate=}")
            sample_num = int(len(_repair_images) * sampling_rate)
            samples = np.random.default_rng().choice(len(_repair_images), sample_num)
            _repair_images = _repair_images[samples]
            _repair_labels = _repair_labels[samples]

            # Appending sampled data
            for _repair_image in _repair_images:
                repair_images.append(list(_repair_image))
            for _repair_label in _repair_labels:
                repair_labels.append(list(_repair_label))

        self.input_protected = np.array(repair_images), np.array(repair_labels)

        dataset = RepairDataset.load_repair_data(input_dir)
        ans_imgs = dataset[0]
        ans_labels = dataset[1]
        return ans_imgs, ans_labels

    def load_weights(self, output_dir):
        """Load neural weight candidates.

        Parameters
        ----------
        output_dir : Path
            Path to directory containing "labels.json"

        Returns
        -------
        req
            Loaded weights

        """
        with open(output_dir / "labels.json") as f:
            req = {}
            for key in json.load(f):
                label = key[0]
                req[label] = key[1]
        return req

    def _sample_positive_inputs(self, input_pos):
        """Sample positive inputs."""
        # Use all positive inputs for optimization
        if self.num_input_pos_sampled is None:
            return input_pos
        sample = np.random.default_rng().choice(
            len(input_pos[0]), self.num_input_pos_sampled
        )
        input_pos_sampled = (input_pos[0][sample], input_pos[1][sample])
        return input_pos_sampled

    def _fail_to_find_better_patch(self, t, history):
        """Stop PSO iterations.

        :param t:
        :param history:
        :return:
        """
        # "stop earlier if it fails to find a better patch
        # than the current best during ten consecutive iterations"
        if self.min_iteration_range < t:
            scores_in_history = np.array(history)[:, 0]

            # Still not yet find better patch
            n_patched_in_history = np.array(history)[:, 1]
            if not 0 < max(n_patched_in_history):
                return False

            best_x_before = scores_in_history[-self.min_iteration_range - 1]
            # Get the current best during ten consecutive iterations
            best_last_x = max(scores_in_history[-self.min_iteration_range :])

            # found a better patch, continue PSO
            if best_last_x > best_x_before:
                return False
            # fail to find a better patch, stagnated, stop PSO
            else:
                return True
        else:
            return False

    def _criterion(self, model, location, input_pos, input_neg):
        """Compute fitness.

        :param model: subject DNN model
        :param location: consists of a neural weight value to mutate,
                          an index of a layer of the model,
                          and a neural weight position (i, j) on the layer
        :param input_pos: positive inputs sampled
        :param input_neg: negative inputs targeted
        :return: fitness, n_patched and n_intact
        """
        # Keep original weights and set specified weights on the model
        orig_location = np.copy(location)
        orig_location = self._copy_weights_to_location(model, orig_location)
        model = self._copy_location_to_weights(location, model)

        # "N_{intact} is th number of inputs in I_{pos}
        # whose output is still correct"
        loss_input_pos, acc_input_pos = model.evaluate(
            input_pos[0], input_pos[1], verbose=0, batch_size=self.batch_size
        )
        n_intact = int(np.round(len(input_pos[1]) * acc_input_pos))

        # "N_{patched} is the number of inputs in I_{neg}
        # whose output is corrected by the current patch"
        loss_input_neg, acc_input_neg = model.evaluate(
            input_neg[0], input_neg[1], verbose=0, batch_size=self.batch_size
        )
        n_patched = int(np.round(len(input_neg[1]) * acc_input_neg))

        # Impose a heavy penalty
        # if a patch violates prediction with inputs
        # that users want to protect
        if 0 < len(self.input_protected[0]):
            results = model.predict(
                self.input_protected[0], verbose=0, batch_size=self.batch_size
            )
            n_protected = 0
            for i in range(len(self.input_protected[1])):
                result = results[i]
                expect = self.input_protected[1][i]
                if result.argmax() == expect.argmax():
                    n_protected += 1
            if n_protected < len(self.input_protected[0]):
                return -1, n_patched, n_intact

        fitness = (n_patched + 1) / (loss_input_neg + 1) + (n_intact + 1) / (
            loss_input_pos + 1
        )

        # Restore original weights to the model
        model = self._copy_location_to_weights(orig_location, model)

        return fitness, n_patched, n_intact

    def optimize(
        self,
        model,
        model_dir: Path,
        weights,
        input_neg,
        input_pos,
        output_dir: Path,
        verbose=1,
    ):
        """Optimization extended from Arachne.

        Parameters
        ----------
        model :
            DNN model to repair
        model_dir : Path
            (Not used)
        weights
            Set of neural weights to target for repair
        input_neg
            (Not used)
        input_pos
            Dataset for correct behavior
        output_dir : Path
            Path to directory to save result
        verbose : int, default=1
            Log level

        """
        weights_repaired = {}
        for label in weights:
            # Not subject
            if "repair_priority" not in weights[label]:
                weights[label]["repair_priority"] = 0
                continue

            if not weights[label]["repair_priority"]:
                continue

            if "repaired_values" in weights[label]:
                idx = 0
                label_weights = []
                for _weight in weights[label]["weights"]:
                    layer_index = _weight[0]
                    nw_i = _weight[1]
                    nw_j = _weight[2]
                    val = np.float32(weights[label]["repaired_values"][idx])
                    idx += 1
                    label_weights.append([val, layer_index, nw_i, nw_j])
                weights_repaired[label] = label_weights
                continue

            # Parse weights localized for each label
            _weights = weights[label]["weights"]
            # Optimize with Arachne
            _output_dir = output_dir / str(label)
            _input_neg = super().load_input_neg(_output_dir)
            _model_repaired = super().optimize(
                model,
                model_dir,
                _weights,
                _input_neg,
                input_pos,
                _output_dir,
                verbose=0,
            )

            # Get values of subject weightw of reparied model
            vals = []
            label_weights = []
            for _weight in _weights:
                layer_index = _weight[0]
                nw_i = _weight[1]
                nw_j = _weight[2]
                # Remove repaired values assigned by Arachne's function.
                del _weight[4]
                del _weight[3]

                layer = _model_repaired.get_layer(index=layer_index)
                nw = layer.get_weights()[0]
                val = nw[nw_i][nw_j]

                label_weights.append([val, layer_index, nw_i, nw_j])
                vals.append(str(val))
            weights_repaired[label] = label_weights
            weights[label]["repaired_values"] = vals
            with open(output_dir / "labels.json", "w") as f:
                req_sorted = sorted(weights.items(), key=lambda x: x[0])
                json.dump(req_sorted, f, indent=4)
                self.output_files.add(output_dir / "labels.json")

        # Generate and save repaired model in final
        # TODO duplicate check
        final_model = load_model_from_tf(model_dir)
        for label in weights_repaired:
            final_model = self._copy_location_to_weights(
                weights_repaired[label], final_model
            )

        self._output_repaired_model(output_dir, final_model)
        self._log_optimize(weights_repaired.items(), verbose)

    def evaluate(
        self,
        dataset,
        model_dir,
        target_data,
        target_data_dir,
        positive_inputs,
        positive_inputs_dir,
        output_dir,
        num_runs,
    ):
        """Not implemented."""
        print("Not implemented.")
