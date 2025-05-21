"""Search-based repair extended from Arachne."""

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import clone_model

from repair.core.dataset import RepairDataset
from repair.methods.arachne import Arachne
from repair.methods.arachne.arachne import model_evaluate

_verbose = 1
_veryverbose = 2


class NeuRecover(Arachne):
    """Search-based repair extended from Arachne."""

    def __init__(self):
        """Initialize."""
        super().__init__()
        self.reduction_rate = 1.0

        self.weights_dir: Path = Path("outputs/logs/model_check_points")
        self.positive_data_dir: Path = Path("outputs/positive")

        # hyper parameter to adjust the degree of regression suppression
        # to calculate fitness
        # TODO: make this configurable in the future
        self.alpha = 1.0

        # Logger settings
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def set_options(self, **kwargs):
        """Set options."""
        super().set_options(**kwargs)

        if (wd := kwargs.get("weights_dir")) is not None:
            self.weights_dir = Path(wd)
            self.logger.debug(f"Set weights dir: {wd}")

        if (pd := kwargs.get("positive_data_dir")) is not None:
            self.positive_data_dir = Path(pd)
            self.logger.debug(f"Set positive data dir: {pd}")

    def localize(self, model, input_neg, output_dir: Path = Path("outputs"), verbose=1):
        """Localize faulty neural weights.

        TODO: Need to evaluate effectiveness.

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

        """
        if verbose == _verbose:
            self.logger.setLevel(logging.INFO)
        elif verbose == _veryverbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        test_images, test_labels = input_neg[0], input_neg[1]

        self.logger.debug(f"Loading positive datasets from {self.positive_data_dir}")
        positive_data = RepairDataset.load_repair_data(self.positive_data_dir)
        pos_images, pos_labels = positive_data[0], positive_data[1]

        test_images = np.concatenate((test_images, pos_images), axis=0)
        test_labels = np.concatenate((test_labels, pos_labels), axis=0)

        # Fault Localization Stage

        # Step i: Data classification with training history
        model_hist, hit_hist = self._get_model_and_hit_histories(model, test_images, test_labels)

        # Step ii: Impact Calculation
        # get indices of improved and regressed data
        (
            test_images_deg,
            test_images_imp,
        ) = self._retrieve_deg_and_imp_images(test_images, test_labels, hit_hist)

        # calculate the weights of weight diff
        weight_deg, weight_imp = self._calc_weight_deg_and_imp(
            model_hist, test_images_deg, test_images_imp
        )

        model_last = model_hist[-1]
        # calculate the weights of forward impact
        self.logger.debug("gathering regresion data")
        weight_fwd_deg, weight_fwd_deg_norm_by_layer = self._calc_weight_by_fwd(
            model_last, test_images_deg
        )

        self.logger.debug("gathering improved data")
        weight_fwd_imp, weight_fwd_imp_norm_by_layer = self._calc_weight_by_fwd(
            model_last, test_images_imp
        )

        # Step iii: Fault localization by set operation
        # calculate the weight diff
        weight_diff_idx_ordered, weight_diff_norm_by_layer, num_layers = self._calc_weight_diff(
            model_hist
        )

        # Patch Generation Stage
        # find suspected layers
        layer_susp = []
        for layer_index in range(num_layers):
            if weight_fwd_deg_norm_by_layer[layer_index] != 0:
                susp = (
                    weight_fwd_deg_norm_by_layer[layer_index]
                    / (
                        weight_fwd_deg_norm_by_layer[layer_index]
                        + weight_fwd_imp_norm_by_layer[layer_index]
                    )
                    * weight_diff_norm_by_layer[layer_index]
                )
            else:
                susp = 0
            layer_susp.append(susp)
        layer_susp = np.array(layer_susp)

        # target weights to repair for each layers
        lens_weights = []
        for w in weight_diff_idx_ordered:
            lens_weights.append(len(w))
        target_weight_num = np.round(lens_weights * layer_susp * self.reduction_rate).astype(int)

        # localization
        # fwd + back, W-deg & W-diff – W-imp, for Mn and Mn-1
        localized_weights = []
        for layer_idx in range(num_layers):
            weight_num = target_weight_num[layer_idx]
            w_diff_topk = set(weight_diff_idx_ordered[layer_idx][0:weight_num])
            w_deg_topk = set(weight_fwd_deg[layer_idx][0:weight_num]) & set(
                weight_deg[layer_idx][0:weight_num]
            )
            w_imp_topk = set(weight_fwd_imp[layer_idx][0:weight_num]) & set(
                weight_imp[layer_idx][0:weight_num]
            )
            for w in w_deg_topk & w_diff_topk - w_imp_topk:
                localized_weights.extend([(layer_idx,) + w])

        modified_weights = self.modify_target_index(localized_weights, model_hist[-1])
        self.save_weights(modified_weights, output_dir)

        return modified_weights

    def _get_model_and_hit_histories(self, model, test_images, test_labels):
        """Get model history(List) and hit history(List).

        Parameters
        ----------
        model : tf.keras.Model
            Target model
        test_images : np.ndarray
            Dataset of images
        test_labels : np.ndarray
            Labels of `test_images`

        Returns
        -------
        model_hist: list[tf.keras.Model]
            List of each epoch models
        hit_hist: list[list[bool]]
            List of prediction results for each epoch models

        Raises
        ------
        FileNotFoundError
            when no weight files in given `weights_dir`

        """
        weight_files = list(self.weights_dir.glob("weights.*"))
        if len(weight_files) == 0:
            errmsg = f"No weight files in {str(self.weights_dir)}"
            raise FileNotFoundError(errmsg)

        weight_files.sort()

        model_hist = []
        hit_hist = []
        for weight_file in weight_files:
            cloned_model = clone_model(model)
            cloned_model.load_weights(weight_file)
            pred_y = cloned_model.predict(test_images)
            hit_hist.append([np.argmax(y) == np.argmax(t) for y, t in zip(pred_y, test_labels)])

            model_hist.append(cloned_model)

        return model_hist, hit_hist

    def _retrieve_deg_and_imp_images(self, test_images, test_labels, hit_hist):
        """Retrieve deg/imp test images and labels according to hit history.

        Parameters
        ----------
        test_images: np.ndarray
            Images
        test_labels: np.ndarray
            Labels of images
        hit_hist: list[list[bool]]
            List of results of prediction for each epoch models

        Notes
        -----
        Original codes seems to use only last two generation models to get deg and imp.

        """
        test_idx_deg = [
            i
            # TODO: this code get data from only last two generation.
            for i, (hit_last, hit_2nd_last) in enumerate(zip(hit_hist[-1], hit_hist[-2]))
            if not hit_last and hit_2nd_last
        ]
        test_idx_imp = [
            i
            # TODO: this code get data from only last two generation.
            for i, (hit_last, hit_2nd_last) in enumerate(zip(hit_hist[-1], hit_hist[-2]))
            if hit_last and not hit_2nd_last
        ]

        test_images_deg = test_images[test_idx_deg]
        test_images_imp = test_images[test_idx_imp]

        return (test_images_deg, test_images_imp)

    def _calc_weight_diff(self, model_hist):
        """Calculate weight diff."""
        weight_hist = []
        for model in model_hist:
            weight_hist.append(model.get_weights())
        num_layers = len(weight_hist[-1])

        weight_diff = []
        for layer_idx in range(num_layers):
            weight_diff.append(weight_hist[-1][layer_idx] - weight_hist[-2][layer_idx])

        weight_diff_idx_ordered = []
        for layer_idx in range(num_layers):
            weight_diff_idx_ordered.append(
                [
                    np.unravel_index(arr_idx, np.shape(weight_diff[layer_idx]))
                    for arr_idx in np.flipud(np.argsort(np.abs(weight_diff[layer_idx]), axis=None))
                ]
            )
        weight_diff_mean_by_layer = [np.mean(np.abs(w)) for w in weight_diff]
        weight_diff_norm_by_layer = weight_diff_mean_by_layer / np.sum(weight_diff_mean_by_layer)

        return weight_diff_idx_ordered, weight_diff_norm_by_layer, num_layers

    def _calc_weight_deg_and_imp(self, model_hist, test_images_deg, test_images_imp):
        """Calculate weight degrade index and improve index."""
        model_last = model_hist[-1]
        weight = model_last.weights

        with tf.GradientTape() as deg_tape:
            deg_tape.watch(weight)
            pred_deg = model_last(test_images_deg)
        grads_deg = deg_tape.gradient(pred_deg, weight)

        with tf.GradientTape() as imp_tape:
            imp_tape.watch(weight)
            pred_imp = model_last(test_images_imp)
        grads_imp = imp_tape.gradient(pred_imp, weight)

        weight_deg = self._calc_weight_indices(grads_deg)
        weight_imp = self._calc_weight_indices(grads_imp)

        return (
            weight_deg,
            weight_imp,
        )

    def _calc_weight_indices(self, grads):
        weights_indices = []
        for grad in grads:
            if grad is not None:
                grad_flat = tf.reshape(grad, [-1])
                sorted_indices = tf.argsort(tf.abs(grad_flat), direction="DESCENDING")
                weights_indices.append(
                    [np.unravel_index(idx, grad.shape) for idx in sorted_indices.numpy()]
                )
        return weights_indices

    def _calc_weight_by_fwd(self, model_last, test_images):
        """Calculate weight degrade index and improve index by forward impact."""
        layer_indexes_has_weights = [
            i
            for i in range(len(model_last.layers))
            if model_last.get_layer(index=i).count_params() != 0
        ]

        forward_impact = {}
        for layer_index in layer_indexes_has_weights:
            weights_layer_index = layer_indexes_has_weights.index(layer_index) * 2
            forward_impact[weights_layer_index] = self._compute_each_forward_impact(
                model_last, test_images, layer_index
            )
            forward_impact[layer_indexes_has_weights.index(layer_index) * 2 + 1] = []

        weight_forward = []
        for forward in forward_impact.values():
            if len(forward) == 0:
                weight_forward.append([])
            else:
                weight_forward.append(
                    [
                        np.unravel_index(arr_idx, np.shape(np.array(forward)))
                        for arr_idx in np.argsort(np.abs(np.array(forward)), axis=None)[::-1]
                    ]
                )

        forward_impact_mean_by_layer = []
        for _, forward in forward_impact.items():
            if len(forward) == 0:
                forward_impact_mean_by_layer.append(0)
            else:
                forward_impact_mean_by_layer.append(np.mean(np.abs(forward)))
        forward_impact_norm_by_layer = forward_impact_mean_by_layer / np.sum(
            forward_impact_mean_by_layer
        )
        return weight_forward, forward_impact_norm_by_layer

    def _compute_each_forward_impact(self, model, test_images, layer_index):
        """Compute forward impact of each weight.

        The product of the given weight
        and the activation value of the corresponding neuron
        in the previous layer.

        Parameters
        ----------
        model : tf.keras.Model
            Target model
        test_images : np.ndarray
            Test data
        layer_index : int
            Target layer_index

        Returns
        -------
        forward_impact
            computed forward impact

        """
        if len(test_images) == 0:
            return []

        if layer_index < 0:
            raise IndexError(f"Not found previous layer: {layer_index}")

        # Evaluate activation value of the corresponding neuron
        # in the previous layer
        if layer_index == 0:
            previous_layer_output = test_images
        else:
            previous_layer = model.get_layer(index=layer_index - 1)
            previous_layer_model = tf.keras.Model(inputs=model.input, outputs=previous_layer.output)
            previous_layer_model.compile(run_eagerly=True)
            previous_layer_output = previous_layer_model.predict(test_images)

        # Evaluate the neuron weight
        target_layer = model.get_layer(index=layer_index)
        if isinstance(target_layer, tf.keras.layers.Conv2D):
            return []

        elif isinstance(target_layer, tf.keras.layers.Dense):
            w_ij = target_layer.get_weights()[0]
            o_i_w_ij = np.array([o_i.reshape(-1, 1) * w_ij for o_i in previous_layer_output])
            forward_impact = np.mean(np.abs(o_i_w_ij), axis=0)

        else:
            raise TypeError(
                f"Unexpected layer: f{target_layer.name}.\n"
                "Avaiable layers are 'Conv2D' or 'Dense'."
            )

        return forward_impact

    def modify_target_index(self, weights, model):
        """Modify target index.

        Target layer indices calculated previous methods are based on weights
        obtained via `model.get_weights()`. `model.get_weights()` returns an array
        that flattens all the weights and biases of the layers of model.
        Thus target layer indices do not indicate real indices of layer.
        This method modifies given target layer indices to real indices.
        """
        # boolean list whether the layer has weights or not
        layer_masks = [len(layer.get_weights()) != 0 for layer in model.layers]

        modified_weights = [
            (self._get_real_layer_index(weight[0], layer_masks),) + weight[1:] for weight in weights
        ]

        return modified_weights

    def _get_real_layer_index(self, layer_index, mask):
        """Get real layer index.

        `layer.get_weights()` returns `[weights, biases]`, so `model.get_weights()` returns
        `[weights_of_layer1, biases_of_layer1, weights_of_layer2, biases_of_layer2, ...]`.
        """
        target_index = layer_index // 2
        if target_index > len(mask):
            raise IndexError("target_index exceeds number of layers having weights.")

        mask_indices = np.where(mask)[0]

        return mask_indices[target_index]

    def _criterion(self, model, location, input_pos, input_neg):
        orig_location = np.copy(location)
        orig_location = self._copy_weights_to_location(model, orig_location)
        model = self._copy_location_to_weights(location, model)

        # "N_{patched} is the number of inputs in I_{neg}
        # whose output is corrected by the current patch"
        loss_input_neg, _acc_input_neg, n_patched = model_evaluate(
            model, input_neg, verbose=0, batch_size=self.batch_size
        )

        # "N_{intact} is th number of inputs in I_{pos}
        # whose output is still correct"
        loss_input_pos, _acc_input_pos, n_intact = model_evaluate(
            model, input_pos, verbose=0, batch_size=self.batch_size
        )
        neg_term = (n_patched / len(input_neg) + 1) / (loss_input_neg + 1)
        pos_term = (n_intact / len(input_pos) + 1) / (loss_input_pos + 1)

        fitness = neg_term + self.alpha * pos_term

        model = self._copy_location_to_weights(orig_location, model)

        return fitness, n_patched, n_intact
