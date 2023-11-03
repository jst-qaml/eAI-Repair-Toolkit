"""Fine-Grained Search-Based Repair."""

import csv
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K  # noqa: N812

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from repair.core import method
from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf


def model_evaluate(model, input_data, verbose, batch_size):
    """Model evaluate.

    Parameters
    ----------
        model
        input_data
        verbose
        batch_size

    """
    if hasattr(input_data, "get_generators"):
        images_generator, labels_generator = input_data.get_generators()
        train_generator = tf.data.Dataset.zip(
            (images_generator, labels_generator)
        )  # generates the inputs and labels in batches
        loss, acc = model.evaluate(train_generator, verbose=verbose)
        n = int(np.round(input_data.image_shape[0] * acc))
        return (loss, acc, n)
    else:
        loss, acc = model.evaluate(
            input_data[0], input_data[1], verbose=verbose, batch_size=batch_size
        )
        n = int(np.round(len(input_data[1]) * acc))
        return (loss, acc, n)


class Arachne(method.RepairMethod):
    """Search-based repair."""

    def __init__(self):
        """Initialize."""
        self.num_grad = None
        self.num_particles = 100
        self.num_iterations = 100
        self.num_input_pos_sampled = 200
        self.velocity_phi = 4.1
        self.min_iteration_range = 10
        self.target_layer = None
        self.output_files = set()
        self.batch_size = 32

    def set_options(self, **kwargs):
        """Set options."""
        if "num_grad" in kwargs:
            self.num_grad = kwargs["num_grad"]
        if "num_particles" in kwargs:
            self.num_particles = kwargs["num_particles"]
        if "num_iterations" in kwargs:
            self.num_iterations = kwargs["num_iterations"]
        if "num_input_pos_sampled" in kwargs:
            self.num_input_pos_sampled = kwargs["num_input_pos_sampled"]
        if "velocity_phi" in kwargs:
            self.velocity_phi = kwargs["velocity_phi"]
        if "min_iteration_range" in kwargs:
            self.min_iteration_range = kwargs["min_iteration_range"]
        if "target_layer" in kwargs:
            self.target_layer = int(kwargs["target_layer"])
        if "batch_size" in kwargs:
            self.batch_size = int(kwargs["batch_size"])

    def localize(self, model, input_neg, output_dir: Path, verbose=1):
        """Localize faulty neural weights.

        NOTE: The arguments of 'weights' and 'loss_func'
              are included in 'model'.

        Parameters
        ----------
        model : repair.core.model.RepairModel
            DNN model to be repaired
        input_neg : tuple[np.ndarray, np.ndarray]
            A set of inputs that reveal the fault
        output_dir : Path, default=Path("outputs")
            Path to directory to save the result
        verbose : int, default=1
            Log level

        Returns
        -------
        A set of neural weights to target for repair

        """
        # "N_g is set to be the number of negative inputs to repair
        # multiplied by 20"
        if self.num_grad is None:
            self.num_grad = len(input_neg[0]) * 20

        reshaped_model = self._reshape_target_model(model, input_neg)

        candidates = self._compute_gradient(reshaped_model, input_neg)

        pool = self._compute_forward_impact(reshaped_model, input_neg, candidates, self.num_grad)

        weights_t = self._extract_pareto_front(pool, output_dir)

        weights_t = self._modify_layer_before_reshaped(model, weights_t)

        # Output neural weight candidates to repair
        self._append_weights(model, weights_t)
        self.save_weights(weights_t, output_dir)

        self._log_localize(weights_t, verbose)
        return weights_t

    def _compute_gradient(self, model, input_neg, desc=True):
        """Compute gradient.

        Arachne sorts the neural weights according to the gradient loss
        of the faulty input backpropagated to the corresponding neuron.

        :param model:
        :param input_neg:
        :param desc:
        :return:
        """
        # For return
        candidates = []

        # Identify class of loss function
        loss_func = tf.keras.losses.get(model.loss)
        layer_index = len(model.layers) - 1
        layer = model.get_layer(index=layer_index)

        # Evaluate grad on neural weights
        with tf.GradientTape() as tape:
            # import pdb; pdb.set_trace()
            logits = model(input_neg[0])  # get the forward pass gradient
            loss_value = loss_func(input_neg[1], logits)
            grad_kernel = tape.gradient(
                loss_value, layer.kernel
            )  # TODO bias?# Evaluate grad on neural weights

        for j in trange(grad_kernel.shape[1], desc="Computing gradient"):
            for i in range(grad_kernel.shape[0]):
                dl_dw = grad_kernel[i][j]
                # Append data tuple
                # (layer, i, j) is for identifying neural weight
                candidates.append([layer_index, i, j, np.abs(dl_dw)])

        # Sort candidates in order of grad loss
        candidates.sort(key=lambda tup: tup[3], reverse=desc)

        return candidates

    def _reshape_target_model(self, model, input_neg):
        """Re-shape target model for localize.

        :param model:
        :param input_neg:
        """
        if self.target_layer is None:
            # "only considers the neural weights connected
            # to the final output layer"
            layer_index = len(model.layers) - 1
            # Search the target layer that satisfies as below
            # 1. The layer is DENSE
            # 2. The output shape corresponds to the final prediction.
            while (
                type(model.get_layer(index=layer_index)) is not tf.keras.layers.Dense
                or model.layers[layer_index].output.shape[1] != input_neg[1].shape[1]
            ) and layer_index > 0:
                layer_index -= 1
            # update the target_layer
            # because this value is used in _modify_layer_before_reshaped
            self.target_layer = layer_index
        else:
            # Considers the neural weights in designated layer.
            layer_index = self.target_layer
            if layer_index == len(model.layers) - 1:
                raise TypeError("Designated layer index is output layer")
        if type(model.get_layer(index=layer_index)) is not tf.keras.layers.Dense:
            raise IndexError(
                "Invalid layer_index: "
                + str(layer_index)
                + " should be keras.layers.core.Dense, but "
                + str(type(model.get_layer(index=layer_index)))
            )
        if layer_index == len(model.layers) - 1:
            return model
        reshaped = tf.keras.models.Model(model.layers[0].input, model.layers[layer_index].output)
        reshaped.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return reshaped

    def _modify_layer_before_reshaped(self, orig_model, weights_t):
        """Modify the target layer to repair the original target model.

        :param orig_model:
        :param weights_t:
        """
        if self.target_layer < 0:
            target_layer = len(orig_model.layers) + self.target_layer
        else:
            target_layer = self.target_layer
        for weight in weights_t:
            weight[0] = target_layer
        return weights_t

    def _compute_forward_impact(self, model, input_neg, candidates, num_grad):
        """Compute forward impact.

        :param model:
        :param input_neg:
        :param candidates:
        :param num_grad:
        :return:
        """
        pool = {}
        layer_index = candidates[0][0]
        _num_grad = num_grad if num_grad < len(candidates) else len(candidates)
        previous_layer = model.get_layer(index=layer_index - 1)
        target_layer = model.get_layer(index=layer_index)

        # Evaluate activation value of the corresponding neuron
        # in the previous layer
        get_activations = K.function([model.input], previous_layer.output)
        activations = get_activations(input_neg[0])
        # Evaluate the neuron weight
        w = K.eval(target_layer.kernel)

        for num in trange(_num_grad, desc="Computing forward impact"):
            layer_index, i, j, grad_loss = candidates[num]
            fwd_imp = self._compute_each_forward_impact(
                model, input_neg, [layer_index, i, j], activations, w
            )
            pool[num] = [layer_index, i, j, grad_loss, fwd_imp]
        return pool

    def _compute_each_forward_impact(self, model, input_neg, weight, activations, w):
        """Compute forward impact of each weight.

        The product of the given weight
        and the activation value of the corresponding neuron
        in the previous layer.

        :param model:
        :param input_neg:
        :param weight:
        :return:
        """
        layer_index = weight[0]
        neural_weight_i = weight[1]
        neural_weight_j = weight[2]

        if layer_index < 1:
            raise IndexError(f"Not found previous layer: {layer_index!r}")

        o_i = activations[0][neural_weight_i]  # TODO correct?

        # Evaluate the neural weight
        w_ij = w[neural_weight_i][neural_weight_j]

        return np.abs(o_i * w_ij)

    def _extract_pareto_front(self, pool, output_dir=None, filename=r"pareto_front.png"):
        """Extract pareto front.

        :param pool:
        :param output_dir:
        :param filename:
        :return:
        """
        # Compute pareto front
        objectives = []
        for key in tqdm(pool, desc="Collecting objectives for pareto-front"):
            weight = pool[key]
            grad_loss = weight[3]
            fwd_imp = weight[4]
            objectives.append([grad_loss, fwd_imp])
        scores = np.array(objectives)
        pareto = self._identify_pareto(scores)
        pareto_front = scores[pareto]

        if output_dir is not None:
            self._save_pareto_front(scores, pareto_front, output_dir, filename)

        # Find neural weights on pareto front
        results = []
        for key in tqdm(pool, desc="Extracting pareto-front"):
            weight = pool[key]
            layer_index = weight[0]
            neural_weight_i = weight[1]
            neural_weight_j = weight[2]
            grad_loss = weight[3]
            fwd_imp = weight[4]
            for _pareto_front in pareto_front:
                _grad_loss = _pareto_front[0]
                _fwd_imp = _pareto_front[1]
                if grad_loss == _grad_loss and fwd_imp == _fwd_imp:
                    results.append([layer_index, neural_weight_i, neural_weight_j])
                    break

        return results

    def _identify_pareto(self, scores):
        """Identify pareto.

        cf. https://pythonhealthcare.org/tag/pareto-front/

        :param scores: Each item has two scores
        :return:
        """
        # Count number of items
        population_size = scores.shape[0]
        # Create a NumPy index for scores on the pareto front (zero indexed)
        population_ids = np.arange(population_size)
        # Create a starting list of items on the Pareto front
        # All items start off as being labelled as on the Parteo front
        pareto_front = np.ones(population_size, dtype=bool)
        # Loop through each item.
        # This will then be compared with all other items
        for i in trange(population_size, desc="Identifying pareto-front"):
            # Loop through all other items
            for j in range(population_size):
                # Check if our 'i' pint is dominated by out 'j' point
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i'
                    # (no more comparisons needed)
                    break
        # Return ids of scenarios on pareto front
        return population_ids[pareto_front]

    def _save_pareto_front(self, scores, pareto_front, output_dir: Path, filename: str):
        """Save pareto front in image."""
        x_all = scores[:, 0]
        y_all = scores[:, 1]
        x_pareto = pareto_front[:, 0]
        y_pareto = pareto_front[:, 1]

        plt.scatter(x_all, y_all)
        plt.plot(x_pareto, y_pareto, color="r")
        plt.xlabel("Objective A")
        plt.ylabel("Objective B")

        plt.savefig(output_dir / filename)

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
        """Optimize.

        cf. https://qiita.com/sz_dr/items/bccb478965195c5e4097

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

        """
        # Initialize particle positions
        locations = self._get_initial_particle_positions(weights, model)

        # "The initial velocity of each particle is set to zero"
        velocities = np.zeros((self.num_particles, len(weights)))

        # Compute velocity bounds
        velocity_bounds = self._get_velocity_bounds(model)

        # "We sample 200 positive inputs"
        input_pos_sampled = self._sample_positive_inputs(input_pos)

        # Convert the dataset into numpy-format not to use generators of tensorflow.
        # When the number of dataset is not so large,
        # the cost of making generators can not be ignored, and it causes memory-leak.
        input_neg = input_neg.sample_from_file(input_neg.image_shape[0])

        # Initialize for PSO search
        personal_best_positions = list(locations)
        personal_best_scores = self._initialize_personal_best_scores(
            locations, model, input_pos_sampled, input_neg
        )
        best_particle = np.argmax(np.array(personal_best_scores)[:, 0])
        global_best_position = personal_best_positions[best_particle]

        # Search
        history = []
        # "PSO uses ... the maximum number of iterations is 100"
        for t in range(self.num_iterations):
            g = self._get_weight_values(global_best_position)
            # "PSO uses a population size of 100"
            for n in trange(
                self.num_particles,
                desc="Updating particle positions" f" (it={t + 1}/{self.num_iterations})",
            ):
                new_weights, new_v, score, n_patched, n_intact = self._update_particle(
                    locations[n],
                    velocities[n],
                    velocity_bounds,
                    personal_best_positions[n],
                    g,
                    model,
                    input_pos_sampled,
                    input_neg,
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

            # Add current best
            history.append(personal_best_scores[best_particle])

            # Stop earlier
            if self._fail_to_find_better_patch(t, history):
                break
        self._append_weights(model, weights)
        model = self._copy_location_to_weights(global_best_position, model)
        self._append_weights(model, weights)
        self.save_weights(weights, output_dir)

        self._output_repaired_model(output_dir, model)
        self._log_optimize(global_best_position, verbose)

        return model

    def _sample_positive_inputs(self, input_pos):
        """Sample 200 positive inputs.

        :param input_pos:
        :return:
        """
        rg = np.random.default_rng()
        sample = rg.choice(len(input_pos[0]), self.num_input_pos_sampled)
        input_pos_sampled = (input_pos[0][sample], input_pos[1][sample])
        # NOTE: Temporally reverted to work with small dataset.
        # returns tuple of the sampled images from input_pos[0]
        # and their respective labels from input_pos[1]
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

    def _update_particle(
        self,
        weights,
        v,
        velocity_bounds,
        personal_best_position,
        g,
        model,
        input_pos_sampled,
        input_neg,
    ):
        """Update position, velocity, and score of each particle.

        Note: for concurrent execution.

        :param weights:
        :param v:
        :param velocity_bounds:
        :param personal_best_position:
        :param g:
        :param model
        :param input_pos_sampled:
        :param input_neg:
        :return:
        """
        x = []
        layer_index = []
        nw_i = []
        nw_j = []
        for weight in weights:
            x.append(weight[0])
            layer_index.append(weight[1])
            nw_i.append(weight[2])
            nw_j.append(weight[3])

        # Computing new position
        new_x = self._update_position(x, v)
        new_weights = []
        for n_new_x in range(len(new_x)):
            new_weights.append([new_x[n_new_x], layer_index[n_new_x], nw_i[n_new_x], nw_j[n_new_x]])

        # Computing new velocity
        p = self._get_weight_values(personal_best_position)
        new_v = self._update_velocity(new_x, v, p, g, velocity_bounds, layer_index)

        # Computing new score
        score, n_patched, n_intact = self._criterion(
            model, new_weights, input_pos_sampled, input_neg
        )

        return new_weights, new_v, score, n_patched, n_intact

    def _get_weight_values(self, weights):
        """Get weight values.

        :param weights:
        :return:
        """
        values = []
        for w in weights:
            values.append(w[0])
        return values

    def _append_weights(self, model, weights):
        for weight in weights:
            layer = model.get_layer(index=int(weight[0]))
            all_weights = layer.get_weights()[0]
            weight.append(all_weights[int(weight[1])][int(weight[2])])

    def _get_layer_index(self, weights):
        """Get layer index.

        :param weights:
        :return:
        """
        values = []
        for w in weights:
            values.append(w[1])
        return values

    def _initialize_personal_best_scores(self, locations, model, input_pos_sampled, input_neg):
        """Initialize personal best scores.

        :param locations: particle locations initialized
        :param model:
        :param input_pos_sampled: to compute scores
        :param input_neg: to compute scores
        :return: personal best scores
        """
        personal_best_scores = []
        for location in tqdm(locations, desc="Initializing particle's scores"):
            fitness, n_patched, n_intact = self._criterion(
                model, location, input_pos_sampled, input_neg
            )
            personal_best_scores.append([fitness, n_patched, n_intact])
        return personal_best_scores

    def _output_repaired_model(self, output_dir, model_repaired):
        """Output repaired model.

        :param output_dir:
        :param model_repaired:
        :return:
        """
        # Output
        output_dir = Path(output_dir)
        # Clean directory
        repair_dir = output_dir / "repair"
        if repair_dir.exists():
            shutil.rmtree(repair_dir)
        repair_dir.mkdir()
        # Save model
        model_repaired.save(repair_dir)

    def _get_initial_particle_positions(self, weights, model):
        """Get initial particle positions based on sibling weights.

        :param weights:
        :param model:
        :paran num_particles:

        :return:
        """
        locations = [[0 for j in range(len(weights))] for i in range(self.num_particles)]
        for n_w in trange(len(weights), desc="Initializing particles"):
            weight = weights[n_w]
            layer_index = int(weight[0])
            nw_i = int(weight[1])
            nw_j = int(weight[2])

            # "By sibling weights, we refer to all weights
            # corresponding to connections between ... L_{n} and L_{n-1}"
            sibling_weights = []
            # L_{n}
            layer = model.get_layer(index=layer_index)
            target_weights = layer.get_weights()[0]
            for j in range(target_weights.shape[1]):
                for i in range(target_weights.shape[0]):
                    # TODO ignore all candidates?
                    if j is not nw_j and i is not nw_i:
                        sibling_weights.append(target_weights[i][j])

            # Each element of a particle vector
            # is sampled from a normal distribution
            # defined by the mean and the standard deviation
            # of all sibling neural weighs.
            mu = np.mean(sibling_weights)
            std = np.std(sibling_weights)
            samples = np.random.default_rng().normal(loc=mu, scale=std, size=self.num_particles)

            for n_p in range(self.num_particles):
                sample = samples[n_p]
                locations[n_p][n_w] = (sample, layer_index, nw_i, nw_j)

        return locations

    def _get_velocity_bounds(self, model):
        """Get velocity bounds.

        "W is the set of all neural weights
        between our target layer and the preceding one."

        wb = np.max(all_weights) - np.min(all_weights)
        vb = (wb / 5, wb * 5)

        :param model:
        :return: dictionary whose key is layer index
                 and value is velocity bounds
        """
        # Range from 1 to #layers
        velocity_bounds = {}
        for layer_index in trange(1, len(model.layers), desc="Computing velocity bounds"):
            layer = model.get_layer(index=layer_index)

            # Target only trainable layers
            if not layer.trainable:
                continue
            # Out of scope if layer does not have kernel
            if not hasattr(layer, "kernel"):
                continue

            # Get all weights
            all_weights = []
            target_weights = layer.get_weights()[0]
            for j in range(target_weights.shape[1]):
                for i in range(target_weights.shape[0]):
                    all_weights.append(target_weights[i][j])

            # Velocity bounds defined at equations 5 and 6
            wb = np.max(all_weights) - np.min(all_weights)
            vb = (wb / 5, wb * 5)

            velocity_bounds[layer_index] = vb
        return velocity_bounds

    def _criterion(self, model, location, input_pos, input_neg):
        """Compute fitness.

        :param model:  subject DNN model
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

        # for input_neg we use the generator, since it can be a big dataset.
        # For input_pos, fow now we leave the numpy array from sampling,
        # but it can be changed to also be a generator.

        # "N_{patched} is the number of inputs in I_{neg}
        # whose output is corrected by the current patch"
        loss_input_neg, acc_input_neg, n_patched = model_evaluate(
            model, input_neg, verbose=0, batch_size=self.batch_size
        )

        # "N_{intact} is th number of inputs in I_{pos}
        # whose output is still correct"
        loss_input_pos, acc_input_pos, n_intact = model_evaluate(
            model, input_pos, verbose=0, batch_size=self.batch_size
        )

        fitness = (n_patched + 1) / (loss_input_neg + 1) + (n_intact + 1) / (loss_input_pos + 1)

        # Restore original weights to the model
        model = self._copy_location_to_weights(orig_location, model)

        return fitness, n_patched, n_intact

    def _copy_location_to_weights(self, location, model):
        """Copy the candidate weights of the target locations to the model.

        :param location: consists of a neural weight value to mutate,
                          an index of a layer of the model,
                          and a neural weight position (i, j) on the layer
        :param model: subject DNN model
        :return: Modified DNN model
        """
        for w in location:
            # Parse location data
            val = w[0]
            layer_index = int(np.round(w[1]))
            nw_i = int(np.round(w[2]))
            nw_j = int(np.round(w[3]))

            # Set neural weight at given position with given value
            layer = model.get_layer(index=layer_index)
            weights = layer.get_weights()
            weights[0][nw_i][nw_j] = val
            layer.set_weights(weights)

        return model

    def _copy_weights_to_location(self, model, location):
        """Store the original weights of the target location.

        :param model: subject DNN model
        :param location: consists of a neural weight value to mutate,
                          an index of a layer of the model,
                          and a neural weight position (i, j) on the layer
        :return: Modified location
        """
        for w in location:
            # Parse location data
            layer_index = int(np.round(w[1]))
            nw_i = int(np.round(w[2]))
            nw_j = int(np.round(w[3]))

            # Set weight value in location with neural weight
            layer = model.get_layer(index=layer_index)
            weights = layer.get_weights()
            w[0] = weights[0][nw_i][nw_j]

        return location

    def _generate_input_data(self, data, labels, batch_size):
        steps_per_epoch = int((len(data) - 1) / batch_size) + 1
        data_size = len(data)
        while True:
            for batch_num in range(steps_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                x = data[start_index:end_index]
                y = labels[start_index:end_index]
                yield x, y

    def _update_position(self, x, v):
        """Update position.

        :param x: current position
        :param v: current velocity
        :return: new position
        """
        return x + v

    def _update_velocity(self, x, v, p, g, vb, layer_index):
        """Update velocity.

        :param x: current position
        :param v: current velocity
        :param p: personal best position
        :param g: global best position
        :param vb: velocity bounds computed in each layer
        :param layer_index:
        :return: new velocity
        """
        # "We follow the general recommendation
        # in the literature and set both to 4.1"
        phi = self.velocity_phi
        # "Equation 3"
        chi = 2 / (phi - 2 + np.sqrt(phi * phi - 4 * phi))
        # "Equation 2"
        ro1 = random.uniform(0, phi)
        ro2 = random.uniform(0, phi)
        # TODO Using same value 'chi'
        #  to 'w', 'c1', and 'c2' in PSO hyper-parameters?
        new_v = chi * (v + ro1 * (p - x) + ro2 * (g - x))
        "we additionally set velocity bounds"
        for n in range(len(new_v)):
            _vb = vb[layer_index[n]]
            _new_v = np.abs(new_v[n])
            _sign = 1 if 0 < new_v[n] else -1
            if _new_v < _vb[0]:
                new_v[n] = _vb[0] * _sign
            if _vb[1] < _new_v:
                new_v[n] = _vb[1] * _sign
        return new_v

    def save_weights(self, weights, output_dir: Path):
        """Save neural weight candidates.

        Parameters
        ----------
        weights
            Weights to be saved
        output_dir : Path
            Path to directory to save weights

        """
        with open(output_dir / "weights.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(weights)
            self.output_files.add(output_dir / "weights.csv")

    def load_weights(self, weights_dir: Path):
        """Load neural weight candidates.

        Parameters
        ----------
        weights_dir : Path
            Path to directory containing 'wights.csv'

        Returns
        -------
        Neural weight candidates

        """
        candidates = []
        with open(weights_dir / "weights.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                candidates.append(row[:3])
        return candidates

    def load_input_neg(self, neg_dir: Path):
        """Load negative inputs.

        Parameters
        ----------
        neg_dir : Path
            Path to directory containing negative dataset

        Returns
        -------
        input_neg
            Loaded negative dataset

        """
        return RepairDataset.load_repair_data(neg_dir)

    def load_input_pos(self, pos_dir: Path):
        """Load positive inputs.

        Parameters
        ----------
        pos_dir : Path
            Path to directory containing positive dataset

        Returns
        -------
        input_pos
            Loaded positive dataset

        """
        return RepairDataset.load_repair_data(pos_dir)

    def evaluate(
        self,
        dataset,
        model_dir: Path,
        target_data,
        target_data_dir: Path,
        positive_inputs,
        positive_inputs_dir: Path,
        output_dir: Path,
        num_runs,
        verbose=1,
    ):
        """Evaluate.

        Parameters
        ----------
        dataset :
            Dataset instance
        model_dir : Path
            Path to directory containing model files
        target_data:
            Negative dataset
        target_data_dir : Path
            Path to directory containing negative dataset
        positive_inputs :
            Positive dataset
        positive_inputs_dir : Path
            Path to directory containing positive dataset
        output_dir : Path
            Path to directory to save results
        num_runs : int
            Number of iterations for repairing
        verbose : int, default=1
            Log level

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        score_rr = []
        score_br = []

        for i in range(num_runs):
            # Load
            model = load_model_from_tf(model_dir)

            # Localize
            localized_data_dir = output_dir / f"localized_data_{i}"
            localized_data_dir_path = localized_data_dir
            if localized_data_dir_path.exists():
                shutil.rmtree(localized_data_dir_path)
            localized_data_dir_path.mkdir()
            self.output_files = set()
            self.localize(model, target_data, localized_data_dir, verbose)

            # Optimize
            weights = self.load_weights(localized_data_dir_path)
            repaired_model_dir = output_dir / f"repaired_model_{i}"
            if repaired_model_dir.exists():
                shutil.rmtree(repaired_model_dir)
            repaired_model_dir.mkdir()
            self.output_files = set()
            self.optimize(
                model,
                model_dir,
                weights,
                target_data,
                positive_inputs,
                repaired_model_dir,
                verbose,
            )

            # Compute RR
            model = load_model_from_tf(repaired_model_dir / "repair")
            repair_target_dataset = dataset.load_repair_data(target_data_dir)
            repair_images, repair_labels = (
                repair_target_dataset[0],
                repair_target_dataset[1],
            )

            score = model.evaluate(
                repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            )
            rr = score[1] * 100
            score_rr.append(rr)

            # Compute BR
            repair_positive_dataset = dataset.load_repair_data(positive_inputs_dir)
            repair_images, repair_labels = (
                repair_positive_dataset[0],
                repair_positive_dataset[1],
            )

            score = model.evaluate(
                repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            )
            br = (1 - score[1]) * 100
            score_br.append(br)

        # Output results
        self._save_evaluate_results(
            dataset,
            model_dir,
            target_data_dir,
            positive_inputs_dir,
            output_dir,
            num_runs,
            score_rr,
            score_br,
        )
        self._log_evaluate(score_rr, score_br, num_runs, verbose)

    def _save_evaluate_results(
        self,
        dataset,
        model_dir: Path,
        target_data_dir: Path,
        positive_inputs_dir: Path,
        output_dir: Path,
        num_runs,
        score_rr,
        score_br,
    ):
        """Save evaluate results.

        Parameters
        ----------
        dataset
        model_dir : Path
        target_data_dir : Path
        positive_inputs_dir : Path
        output_dir : Path
        num_runs : int
        score_rr
        score_br

        """
        # Compute average value
        ave_rr = sum(score_rr) / len(score_rr)
        ave_br = sum(score_br) / len(score_br)

        with open(output_dir / "result.txt", mode="w") as f:
            f.write("# Settings\n")
            f.write("dataset: %s\n" % (dataset.__class__.__name__))
            f.write("method: %s\n" % (self.__class__.__name__))
            f.write("model_dir: %s\n" % (model_dir))
            f.write("num_grad: %s\n" % (self.num_grad))
            f.write("target_data_dir: %s\n" % (target_data_dir))
            f.write("positive_inputs_dir: %s\n" % (positive_inputs_dir))
            f.write("output_dir: %s\n" % (output_dir))
            f.write("num_particles: %s\n" % (self.num_particles))
            f.write("num_iterations: %s\n" % (self.num_iterations))
            f.write("num_runs: %s\n" % (num_runs))
            f.write("\n# Results\n")
            for i in range(num_runs):
                f.write("%d: RR %.2f%%, BR %.2f%%\n" % (i, score_rr[i], score_br[i]))
            f.write(f"\nAverage: RR {ave_rr:.2f}%, BR {ave_br:.2f}%")

            self.output_files.add(output_dir / "result.txt")

    def _log_localize(self, results, verbose):
        if verbose > 0:
            print("=================")
            print("Localize results")
            print("created files:")
            for file_path in self.output_files:
                print("    ", file_path)
            if verbose > 1:
                print("localized weights:")
                for result in results:
                    print("    ", result)
            print("=================")

    def _log_optimize(self, results, verbose):
        if verbose > 0:
            print("=================")
            print("Optimize results")
            print("created files:")
            for file_path in self.output_files:
                print("    ", file_path)
            if verbose > 1:
                print("optimized weights:")
                for result in results:
                    print("    ", result)
            print("=================")

    def _log_evaluate(self, score_rr, score_br, num_runs, verbose):
        if verbose > 1:
            print("=================")
            print("Evaluate results")
            print("RR/BR")
            for i in range(num_runs):
                print("    %d: RR %.2f%%, BR %.2f%%\n" % (i, score_rr[i], score_br[i]))
            print("=================")
