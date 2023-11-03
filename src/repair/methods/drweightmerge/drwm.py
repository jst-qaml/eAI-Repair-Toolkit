"""DRWeightMerge Repair."""

from __future__ import annotations

import csv
import inspect
import json
import math

# TODO: move logger configurations to core
from logging import DEBUG, StreamHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from tqdm import tqdm, trange

from repair.core.loader import load_repair_dataset
from repair.methods.arachne import Arachne

logger = getLogger("repair.methods.drweightmerge")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

if TYPE_CHECKING:
    Mode = Literal["WEIGHTMERGE", "DISTREPAIR"]
    LabelName = str
    LabelValue = str
    Weight = float
    LayerIndex = int
    SourceNeuron = int
    TargetNeuron = int
    NeuralConnection = tuple[Weight, LayerIndex, SourceNeuron, TargetNeuron]
    SharedWeights = dict[tuple[LayerIndex, SourceNeuron, TargetNeuron], set[list[int] | int]]


class DRWeightMerge(Arachne):
    """Distributed Repair.

    Attributes
    ----------
    general_misclassifications : list[str]
        List of label name of targetting misclassified
    target_misclassifications : list[tuple[str, str]]
        List of pair of name of targetting misclassified.
        The first element is true label, the later is misclassified label name.
    weights_precisions : dict[str, float]
        Dict of pairs consisted of label name and float value.
        Set of the priority that the accuracy of key should be preserved.
        The bigger value means the higher priority.
    weights_misclassifications: dict[str|tuple, float]
        Dict of pairs consisted of label name and float value.
        Set of the risk level that should be prioritized for repair.
        The bigger value means the higher risk level.
    negative_root_dir: Path, default=Path("outputs/negative")
        Path to root directory containing negative datasets.

    """

    def __init__(self):
        super().__init__()

        self.whole_location: dict[int, list[NeuralConnection]] = {}
        self.original_location = []
        self.true = None
        self.images = None
        self.ratio = 0.2
        self.shared_weights: SharedWeights = {}
        self.num_run = 0
        self.label_map: dict[LabelName, LabelValue] = {}
        self.class_distr: dict[LabelName, int] = {}
        self.optimize_mode: Mode = "DISTREPAIR"
        self.weight_path = Path()
        self.target_misclassifications: set[tuple[LabelName, LabelName]] = set()
        self.general_misclassifications: list[LabelName] = []
        self.weights_precisions: dict[LabelName, float] = {}
        self.weights_misclassifications: dict[LabelName | tuple[LabelName, LabelName], float] = {}
        self.negative_root_dir: Path = Path("outputs/negative")
        self.__seed: int | None = None

    def set_options(self, **kwargs):
        """Set options."""
        if (value := kwargs.pop("seed", None)) is not None:
            self.__seed = value

        if (value := kwargs.pop("dataset", None)) is not None:
            dataset = load_repair_dataset(value)
            self.label_map = dataset.get_label_map()
            self.class_distr = {key: 0 for key in self.label_map.keys()}

        self.weight_path = Path(kwargs.pop("weight_path", "dr_setting.json"))

        if (value := kwargs.pop("negative_root_dir", None)) is not None:
            self.negative_root_dir = Path(value)

        with open(self.weight_path) as f:
            settings = json.load(f)
            self._load_dr_setting(settings)

        super().set_options(**kwargs)

    def _set_optimizer_mode(self, mode: Mode):
        self.optimize_mode = mode

    def _make_misclassification_dictkey(self, keys: str):
        if "," in keys:
            return tuple(keys.split(","))
        else:
            return keys

    def _load_dr_setting(self, settings: dict):
        tm: dict[str, list[str]] | None = settings.get("target_misclassifications")
        if tm is None:
            raise KeyError(
                f"Required key not found in {str(self.weight_path)}: " "'target_misclassifications'"
            )

        gm: list[str] | None = settings.get("general_misclassifications")
        if gm is None:
            raise KeyError(
                f"Required key not found in {str(self.weight_path)}: "
                "'general_misclassifications'"
            )

        wp: dict[str, float] | None = settings.get("weights_precisions")
        if wp is None:
            raise KeyError(
                f"Required key not found in {str(self.weight_path)}: 'weights_precisions'"
            )
        # Normalize
        sum_wp_values = math.fsum(wp.values())
        if sum_wp_values == 0:
            raise ValueError("Invalid sum of weight precisions. The value must not be 0.")
        for wp_key in wp.keys():
            wp_value = float(wp[wp_key])
            if wp_value < 0:
                raise ValueError("Invalid weight precision. The value must be 0 or greater.")
            wp[wp_key] = wp_value / sum_wp_values

        wm: dict[str, float] | None = settings.get("weights_misclassifications")
        if wm is None:
            raise KeyError(
                f"Required key not found in {str(self.weight_path)}: "
                "'weights_misclassifications'"
            )
        # Normalize
        sum_wm_values = math.fsum(wm.values())
        if sum_wm_values == 0:
            raise ValueError("Invalid sum of weight misclassifications. The value must not be 0.")
        for wm_key in wm.keys():
            wm_value = float(wm[wm_key])
            if wm_value < 0:
                raise ValueError(
                    "Invalid weight misclassification. The value must be 0 or greater."
                )
            wm[wm_key] = wm_value / sum_wm_values

        self.target_misclassifications = {
            (key, target) for key, targets in tm.items() for target in targets
        }

        self.general_misclassifications = gm

        self.weights_precisions = {key: precision for key, precision in wp.items()}

        self.weights_misclassifications = {
            self._make_misclassification_dictkey(key): weight for key, weight in wm.items()
        }

    def _reset_arachne_params_for_localize(self):
        self.target_layer = None
        self.output_files.clear()

    def _reset_arachne_params_for_optimize(self):
        self.output_files.clear()

    def load_weights(self, weights_path: Path):
        """Load weights.

        DRWeightMerge method doesn't require global weights because its optimizer loads weights from
        several directories designated by setting. Then this method just returns nothing
        when called from cli, otherwise call Arachne's method to invoke Arachne's `optimize`.

        Parameters
        ----------
        weights_path : Path
            Path to weights csv

        Returns
        -------
        None
            if called from cli
        Neural weight candidates
            otherwise

        """
        currframe = inspect.currentframe()
        if currframe is not None:
            frame = inspect.getouterframes(currframe, 2)[1]
            if frame.filename != __file__ and frame.function == "optimize":
                return None

        return super().load_weights(weights_path)

    def localize(self, model, input_neg, output_dir: Path, verbose=1):
        """Localize weights.

        Run for respective `target_misclassifications` and `global_misclassifications`.

        Parameters
        ----------
        model : repair.core.model.RepairModel
            DNN model to be repaired
        input_neg : tuple[np.ndarray, np.ndarray]
            Not used for this method
        output_dir : Path, default=Path("outputs")
            Path to directory to save the result
        verbose : int, default=1
            Log level

        Todos
        -----
        Handle if target_misclassifications is empty, general_misc as well.

        """
        for msc in self.target_misclassifications:
            self._reset_arachne_params_for_localize()

            true_class = self.label_map[msc[0]]
            misclassified_class = self.label_map[msc[1]]
            target_data_path = self.negative_root_dir / str(true_class) / str(misclassified_class)

            if not target_data_path.exists():
                logger.warn(f"{str(target_data_path)} does not exist. Skipped.")
                continue

            super().localize(
                model=model,
                input_neg=self.load_input_neg(target_data_path),
                output_dir=target_data_path,
            )

        for g_msc in self.general_misclassifications:
            self._reset_arachne_params_for_localize()

            true_class = self.label_map[g_msc]
            target_data_path = self.negative_root_dir / str(true_class)

            if not target_data_path.exists():
                logger.warn(f"{str(target_data_path)} does not exist. Skipped.")
                continue

            super().localize(
                model=model,
                input_neg=self.load_input_neg(target_data_path),
                output_dir=target_data_path,
                verbose=verbose,
            )

    def optimize(
        self,
        model,
        model_dir,
        weights,
        input_neg,
        input_pos,
        output_dir,
        verbose=1,
    ):
        """Optimize.

        Distributed repair has two steps to optimize weights.
        1. optimize respective target misclassifications and general ones
        2. merge optimized weights by 1.

        Parameters
        ----------
        model :
            DNN model to repair
        model_dir : Path
            (Not used)
        weights
            Set of neural weights to target for repair
        input_neg
            Dataset for unexpected behavior
        input_pos
            Dataset for correct behavior
        output_dir : Path
            Path to directory to save result
        verbose : int, default=1
            Log level

        Todos
        -----
        Pass negative input root instead of using output_dir

        """
        # First load negative inputs and store them as class attributes
        neg_images, neg_labels = input_neg[0], input_neg[1]
        neg_labels = np.argmax(neg_labels, axis=1)

        # Add the positive inputs
        pos_images, pos_labels = input_pos[0], input_pos[1]
        pos_labels = np.argmax(pos_labels, axis=1)
        self.images = np.concatenate([neg_images, pos_images], axis=0)
        self.true = np.concatenate([neg_labels, pos_labels], axis=0)

        # Step 1. optimize respective target misclassifications
        # make sure to set DISTREPAIR mode
        self._set_optimizer_mode("DISTREPAIR")
        logger.debug("Optimize each target misclassified")
        for msc in self.target_misclassifications:
            self._reset_arachne_params_for_optimize()

            true_class = self.label_map[msc[0]]
            misclassified_class = self.label_map[msc[1]]
            target_data_path = self.negative_root_dir / str(true_class) / str(misclassified_class)

            if not target_data_path.exists():
                logger.warn(f"{str(target_data_path)} does not exist. Skipped.")
                continue

            super().optimize(
                model=model,
                model_dir=model_dir,
                weights=self.load_weights(target_data_path),
                input_neg=self.load_input_neg(target_data_path),
                input_pos=input_pos,
                output_dir=target_data_path,
                verbose=verbose,
            )

        logger.debug("Optimize each general misclassified")
        for g_msc in self.general_misclassifications:
            self._reset_arachne_params_for_optimize()

            true_class = self.label_map[g_msc]
            target_data_path = self.negative_root_dir / str(true_class)

            if not target_data_path.exists():
                logger.warn(f"{str(target_data_path)} does not exist. Skipped.")
                continue

            super().optimize(
                model=model,
                model_dir=model_dir,
                weights=self.load_weights(target_data_path),
                input_neg=self.load_input_neg(target_data_path),
                input_pos=input_pos,
                output_dir=target_data_path,
                verbose=verbose,
            )

        # Step 2. merge optimized weights
        self._set_optimizer_mode("WEIGHTMERGE")

        # first, search coefficiences
        self.shared_weights = self._find_shared_weights()
        self.whole_location = self._retrieve_all_weights_modelwise(self.negative_root_dir)
        self.model = model

        solution = self.find_best_weights(model)
        model = self._copy_location_to_weights(solution, model)
        self.save_weights(solution, output_dir)
        self._output_repaired_model(output_dir, model)

    def _find_shared_weights(self):
        intersection_pairs: dict[tuple[LayerIndex, SourceNeuron, TargetNeuron], set] = {}
        loc = self.whole_location.keys()
        for i in loc:
            for j in loc:
                if i != j:
                    m1 = self.whole_location[i]
                    m2 = self.whole_location[j]

                    m1 = [(x[1], x[2], x[3]) for x in m1]
                    m2 = [(x[1], x[2], x[3]) for x in m2]

                    inters = [w for w in m1 if w in m2]

                    if len(inters) > 0:
                        for m in inters:
                            intersection_pairs[m] = set()
                            if m not in intersection_pairs.keys():
                                intersection_pairs[m] = set([i, j])
                            else:
                                intersection_pairs[m].add(i)
                                intersection_pairs[m].add(j)
        return intersection_pairs

    def _retrieve_all_weights_modelwise(self, target_data_dir: Path):
        """Retrieve all weights.

        Loads the weights of all the repaired misclassifications and
        merges them in a single weight set.

        """
        weights = {}
        count = 0

        # load all repaired misclassifications
        for msc in self.target_misclassifications:
            true_class = self.label_map[msc[0]]
            mispred_class = self.label_map[msc[1]]

            local_miscl_path = target_data_dir / str(true_class) / str(mispred_class)

            weights[count] = self._load_full_weights(local_miscl_path)

            count += 1

        for cl in self.general_misclassifications:
            true_class = self.label_map[cl]
            general_miscl_path = target_data_dir / str(true_class)
            weights[count] = self._load_full_weights(general_miscl_path)

            count += 1

        # merge weights
        for i in weights:
            weight_list = weights[i]
            weights[i] = [(float(w[4]), int(w[0]), int(w[1]), int(w[2])) for w in weight_list]

        return weights

    def _load_full_weights(self, output_dir):
        """Load neural weight candidates.

        :param output_dir: path to directory containing 'weights.csv'
        :return: Neural weight candidates
        """
        output_dir = Path(output_dir)

        candidates = []
        with open(output_dir / "weights.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                candidates.append(row[:])
        return candidates

    def find_best_weights(self, model):
        """Find best weights.

        Find the weights that scores the best fitness value by PSO.
        In this method, location is same meaning as weights.

        Parameters
        ----------
        model : keras.Model

        """
        self._store_initial_locations(model)

        dimensions = len(self.whole_location) + len(self.shared_weights)

        locations = self._get_initial_particle_positions_for_dr(dimensions)

        # "The initial velocity of each particle is set to zero"
        velocities = np.zeros((self.num_particles, dimensions))

        # Compute velocity bounds
        velocity_bounds = self._get_velocity_bounds_for_dr(model)

        # Initialize for PSO search
        personal_best_positions = list(locations)
        personal_best_scores = self._initialize_personal_best_scores_for_weight_merge(locations)
        best_particle = np.argmax(np.array(personal_best_scores)[:, 0])
        global_best_position = personal_best_positions[best_particle]

        # Search
        history = []
        # "PSO uses ... the maximum number of iterations is 100"
        for t in range(self.num_iterations):
            # "PSO uses a population size of 100"
            for n in trange(
                self.num_particles,
                desc=f"Updating particle positions (it={t + 1}/{self.num_iterations})",
            ):
                new_weights, new_v, score, n_patched, n_intact = self._update_particle_for_dr(
                    locations[n],
                    velocities[n],
                    velocity_bounds,
                    personal_best_positions[n],
                    global_best_position,
                )

                # Update position
                locations[n] = new_weights
                # Update velocity
                velocities[n] = new_v
                # Update score
                if personal_best_scores[n][0] < score:
                    personal_best_scores[n] = [score, n_patched, n_intact]
                    personal_best_positions[n] = locations[n]

            # Update global best
            best_particle = np.argmax(np.array(personal_best_scores)[:, 0])
            global_best_position = personal_best_positions[best_particle]
            # add current best
            history.append(personal_best_scores[best_particle])

            # Stop earlier
            if self._fail_to_find_better_patch(t, history):
                break

        new_location = self.adjust_weights(global_best_position)
        return new_location

    def _store_initial_locations(self, model):
        """Store initial locations.

        Store the weights of given model as initialized locations for PSO.

        Parameters
        ----------
        model : keras.Model

        """
        original_location = []
        for i in self.whole_location:
            w_list = self.whole_location[i]

            for w in w_list:
                _, n_layer, ni, nj = w

                layer = model.get_layer(index=n_layer)
                weights = layer.get_weights()
                w_orig = weights[0][ni][nj]
                w_orig = (w_orig, n_layer, ni, nj)

                if w_orig not in original_location:
                    original_location.append(w_orig)

        self.original_location = original_location

    def _get_initial_particle_positions_for_dr(self, dims, std=0.125):
        mu = 0.5

        rng = np.random.default_rng(self.__seed)
        initial_positions = [
            rng.normal(mu, std, size=dims).tolist() for i in range(self.num_particles)
        ]

        return initial_positions

    def _initialize_personal_best_scores_for_weight_merge(self, locations: list[list[Weight]]):
        personal_best_scores = []
        for location in tqdm(locations, desc="Initializing particle's scores"):
            fitness = self.fitness_float(location)
            personal_best_scores.append([fitness, 0, 0])
        return personal_best_scores

    def _update_particle_for_dr(
        self, locations, v, velocity_bounds, personal_best_position, global_best_position
    ):
        new_locations = self._update_position(locations, v)
        new_v = self._update_velocity_for_dr(
            new_locations, v, personal_best_position, global_best_position, velocity_bounds
        )

        score = self.fitness_float(new_locations)
        return new_locations, new_v, score, 0, 0

    def _update_velocity_for_dr(self, x, v, p, g, vb, velocity_phi=4.1):
        # "We follow the general recommendation in the literature and set both to 4.1"
        phi = velocity_phi
        # "Equation 3"
        chi = 2 / (phi - 2 + np.sqrt(phi * phi - 4 * phi))
        # "Equation 2"
        rng = np.random.default_rng(self.__seed)
        ro1 = rng.uniform(0, phi)
        ro2 = rng.uniform(0, phi)
        # TODO Using same value 'chi'
        #  to 'w', 'c1', and 'c2' in PSO hyper-parameters?
        new_v = chi * (v + ro1 * (p - x) + ro2 * (g - x))
        for n in range(len(new_v)):
            _new_v = np.abs(new_v[n])
            _sign = 1 if 0 < new_v[n] else -1
            if _new_v < vb[0]:
                new_v[n] = vb[0] * _sign
            if vb[1] < _new_v:
                new_v[n] = vb[1] * _sign
        return new_v

    def _get_velocity_bounds_for_dr(self, model):
        """Velocity bounds for weight merge optimize.

        Todos
        -----
        Define metrics

        """
        return [1 / 5, 5]

    def fitness_float(self, coefficients: list[float]):
        """Fitness.

        Fitness function used to implement this:
            Coeffient [0-1] for each model [c1, c2, c3],
            the individual should contain weight [0, 1] floats
            Suppose for model Mi we have weights Wi
            (all the weights considered by all the models).
            When merging we generate a new model sum(ciWi)
        Here `self.whole_location` is a dictionary `d` where `d[i]`
        contains the weights associatd with the i-th model.
        Coefficients is simply a list of coefficients.

        """
        for coefficient in coefficients:
            if coefficient < 0 or coefficient > 1:
                return -1

        new_location = self.adjust_weights(coefficients)

        old_location = []

        # THis impl make flat list of float, not list of tuple.
        for loc in self.whole_location:
            old_location += self.whole_location[loc]

        old_location = np.array(old_location)

        self.model = self._copy_location_to_weights(new_location, self.model)

        confidences = self.model.predict(self.images)
        predictions = np.argmax(confidences, axis=1)

        self.model = self._copy_location_to_weights(old_location, self.model)

        score = self.compute_score(predictions)
        return score

    def adjust_weights(self, coefficients: list[Weight]) -> list[NeuralConnection]:
        """Adjust weights.

        Given a set of coefficients it derives the corresponding weights,
        considering the shared weights.

        Todos
        -----
        I have no idea where 8 comes from.
        It should be better to split shared_weights into two group.

        """
        new_location: list[NeuralConnection] = []

        for i, c in enumerate(coefficients):
            # TODO: What mean 8?
            limit = 8
            # Code handling non-shared weights
            if i < limit:
                weight_set = self.whole_location[i]

                for w in weight_set:
                    if (w[1], w[2], w[3]) in self.shared_weights.keys():
                        # we want to select weights not shared
                        continue

                    # Find the original value w0
                    w_orig: Weight = 0
                    for w_old in self.original_location:
                        if (w[1], w[2], w[3]) == (w_old[1], w_old[2], w_old[3]):
                            w_orig = w_old[0]

                    w1: Weight = w[0]

                    # adjust weight
                    w_new: float = min(w1, w_orig) + c * (max(w1, w_orig) - min(w1, w_orig))
                    new_location.append((w_new, w[1], w[2], w[3]))
            else:
                # handling for shared weights
                shared_w_index = i - 8
                curr_key = list(self.shared_weights.keys())[shared_w_index]
                # list of models that share this weight
                # NOTE: sharers should be list of int if weights is actually shared.
                sharers: list[int] = list(self.shared_weights[curr_key])
                # list storing the values of the repaired models, for all sharers
                repaired_values: list[float] = []
                for m in sharers:
                    weight_set = self.whole_location[m]

                    # Search among the weights of model m
                    # until you find the value of that model
                    for w in weight_set:
                        if (w[1], w[2], w[3]) == curr_key:
                            repaired_values.append(w[0])
                            break
                v0 = 0
                # Find the orginal value
                for w_old in self.original_location:
                    if curr_key == (w_old[1], w_old[2], w_old[3]):
                        v0 = w_old[0]

                # adjust weight
                v_list = repaired_values + [v0]
                w_new = min(v_list) + c * (max(v_list) - min(v_list))
                new_location.append((w_new, curr_key[0], curr_key[1], curr_key[2]))

        return new_location

    def compute_score(self, predicted):
        """Computes the competition score.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted labels

        Returns
        -------
        score : float
            Computed score

        """
        tot = len(predicted)
        diff_misclassifications = {}
        diff_general_miscl = {}

        for msc in self.target_misclassifications:
            diff_misclassifications[msc] = 0

        for cl in self.general_misclassifications:
            diff_general_miscl[cl] = 0

        for cl in self.class_distr:
            self.class_distr[cl] = 0

        class_map_inv = {v: k for k, v in self.label_map.items()}

        for i in range(tot):
            self.class_distr[class_map_inv[self.true[i]]] += 1
            if predicted[i] != self.true[i]:
                true_class_name = class_map_inv[self.true[i]]
                pred_class_name = class_map_inv[predicted[i]]

                if true_class_name in self.general_misclassifications:
                    diff_general_miscl[true_class_name] += 1

                if (true_class_name, pred_class_name) in self.target_misclassifications:
                    diff_misclassifications[(true_class_name, pred_class_name)] += 1

        score = 0

        for cl in diff_general_miscl:
            precision = 1 - diff_general_miscl[cl] / self.class_distr[cl]

            score += self.weights_precisions[cl] * precision

            if cl in self.weights_misclassifications.keys():
                miscl_rate = diff_general_miscl[cl] / self.class_distr[cl]
                score += -0.5 * miscl_rate * self.weights_misclassifications[cl]

        for mscl in diff_misclassifications:
            miscl_rate = diff_misclassifications[mscl] / self.class_distr[mscl[0]]
            w = self.weights_misclassifications[mscl]
            score += -0.5 * miscl_rate * w

        return score

    def _criterion(self, model, location, input_pos, input_neg):
        """Criterion.

        In DISTREPAIR mode, use Arachne's criterion.
        In WEIGHTMERGE mode, use fitness_float

        """
        if self.optimize_mode == "DISTREPAIR":
            return super()._criterion(model, location, input_pos, input_neg)
        else:
            return (self.fitness_float(location), 0, 0)
