"""Kaminari Optimizer."""

import copy
import logging

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential

logger = logging.getLogger(__name__)


class KaminariUtils:
    """Kaminari utils.

    Module to reduce a model by removing its first self.depth layers
    using extract_tail. Also save_processed_images, changing them from the
    original value, to their output after layer self.depth Using
    translate_weights and de_translate_weights, we modify their first value
    (layer index) so repair approaches like Arachne can work with the models
    before and after the KaminariOptimization.
    """

    def __init__(self, depth):
        if depth < 0:
            errmsg = "Depth should be positive"
            raise ValueError(errmsg)
        self.depth = depth

    def _get_kaminari_inputs(self, model, inputs, batch_size=32):
        """Get the inputs to feed the reduced model.

        This computation done here is the part that is optimized and doesn't
        need to be done during all the evaluations of the repair approach.

        Parameters
        ----------
        model
            model to be reduced
        inputs : np.ndarray
            original images given to the model
        batch_size : int
            batch size to use for the images prediction

        Returns
        -------
        the outputs of layer self.depth when given these inputs

        """
        if self.depth == 0:
            return inputs.copy()

        if isinstance(model, Sequential):
            head_outputs = model.layers[self.depth - 1].output
        elif isinstance(model, tf.keras.Model):
            tail_input_layers = self._get_tail_layers_inputs(model)
            head_outputs = [layer.output for layer in tail_input_layers]
        else:
            errmsg = (
                f"Unknown model type: {type(model)!r}" "Use Functional or Sequential Keras models."
            )
            raise TypeError(errmsg)

        head_inputs = model.input

        head_model = tf.keras.Model(inputs=head_inputs, outputs=head_outputs, name="head")

        kaminari_inputs = head_model.predict(inputs[0], batch_size=batch_size)
        if len(head_model.outputs) > 1:
            kaminari_inputs = [np.array(kaminari_input) for kaminari_input in kaminari_inputs]
        else:
            kaminari_inputs = np.array(kaminari_inputs)
        # add back the labels that go with each image
        kaminari_inputs = [kaminari_inputs, inputs[1]]
        return kaminari_inputs

    def save_processed_images(
        self,
        model,
        input_neg,
        input_pos,
        output_path=None,
        input_path=None,
        batch_size=32,
    ):
        """

        Get the output from layer self.depth of the model when given this inputs.

        Given a
            - keras model
            - negative images
            - positive images
            - depth of the layer whose output you want to retrieve
        it feeds the images to the portion of the model
        defined by the depth attributed
        and stores the output in lists, then returns them.
        It is recommended that if the returned processed images will be
        predicted with batch size X, X is used as an input for this method

        Parameters
        ----------
        model
            the model that will give output of layer self.depth
        input_neg : np.ndarray
            images that are incorrectly classified by the model
        input_pos : np.ndarray
            images that are correctly classified by the model
        output_path : Path
            path where we want to save the processed images
        input_path : Path
            path to processed images instead of computing them if previously saved
        batch_size : int
            size of batch for each process


        Returns
        -------
        the outputs of layer self.depth when given these inputs

        """
        logger.info("Preprocessing images...")
        # If user specified an input path containing pre-computed data, load it
        if input_path is not None:
            hf = h5py.File(input_path)
            negative = hf.get("negative")
            positive = hf.get("positive")
            hf.close()
            return negative, positive

        if self.depth == 0:
            return copy.deepcopy(input_neg), copy.deepcopy(input_pos)

        new_neg = self._get_kaminari_inputs(model, input_neg, batch_size)
        new_pos = self._get_kaminari_inputs(model, input_pos, batch_size)

        # Optionally save the compressed dataset to some files
        if output_path is not None:
            hf = h5py.File(output_path, "w")
            hf.create_dataset(name="negative", data=new_neg)
            hf.create_dataset(name="positive", data=new_pos)
            hf.close()

        logger.info("DONE")
        return new_neg, new_pos

    def copy_layer(self, layer):
        """Copy layer.

        Return a copy of the layer.

        Returns a copy of the layer.
        based on https://github.com/keras-team/keras/issues/13140

        Parameters
        ----------
        layer:
            DNN layer to be copied

        Returns
        -------
        Set(layer):
            true deep copy of the layer

        """
        config = layer.get_config()
        cloned_layer = type(layer).from_config(config)
        cloned_layer.build(layer.input_shape)
        return cloned_layer

    def _get_tail_layers_inputs(self, model):
        tail_input_layers = set()
        for layer in model.layers[: self.depth]:
            for node in layer.outbound_nodes:
                if node.layer in model.layers[self.depth :]:
                    tail_input_layers.add(layer)
                    break
        return tail_input_layers

    def _get_reduced_sequential_model(self, model):
        """Get reduced model for sequential model.

        Extracts the layers between depth and the last from model,
        and copies them to a new smaller model.

        Parameters
        ----------
        model:
            the original model, that we will copy only the last layers

        Returns
        -------
        Reduced model.

        """
        input_shape = model.layers[self.depth].input_shape[1:]
        submodel = Sequential()

        if self.depth > 0:
            # if depth is 0, we don't need to modify the input
            submodel.add(Input(shape=input_shape, batch_size=None, name="input"))

        for layer in model.layers[self.depth :]:
            logger.debug(f"Adding layer {layer}")
            copy = self.copy_layer(layer)
            submodel.add(copy)
            copy.set_weights(layer.get_weights())

        submodel.build(input_shape)
        submodel.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return submodel

    def _get_reduced_functional_model(self, model):
        """Get reduces model for functional model.

        Extracts the layers between depth and the last from model,
        and copies them to a new smaller model.

        Parameters
        ----------
        model:
            the original model, that we will copy only the last layers

        Returns
        -------
        New reduced model, which is the copy of the last
        layers of the input model, removing the first self.depth layers

        """
        names_dict = {}
        tail_layers = model.layers[self.depth :]
        tail_input_layers = self._get_tail_layers_inputs(model)
        logger.debug(f"TAIL INPUT LAYERS: {tail_input_layers!r}")
        new_inputs = []
        for layer in tail_input_layers:
            if isinstance(layer.output_shape, list):
                new_inputs.append(Input(shape=layer.output_shape[0][1:], name=layer.name))
            else:
                new_inputs.append(Input(shape=layer.output_shape[1:], name=layer.name))
            names_dict[new_inputs[-1].name] = layer.name
        logger.debug(f"NEW INPUTS: {new_inputs!r}")
        new_layers = []
        for layer in tail_layers:
            logger.debug(f"LAYER: {layer!r}")
            inbound_layer_names = []
            for n in layer.inbound_nodes:
                if isinstance(n.inbound_layers, list):
                    for lay in n.inbound_layers:
                        inbound_layer_names.append(lay.name)
                else:
                    inbound_layer_names.append(n.inbound_layers.name)
            new_layer_inputs = [i for i in new_inputs if names_dict[i.name] in inbound_layer_names]
            new_layer_inputs.extend(
                [i for i in new_layers if names_dict[i.name] in inbound_layer_names]
            )
            config = layer.get_config()
            new_layer = type(layer).from_config(config)
            new_layer.build(layer.input_shape)
            new_layer.set_weights(layer.get_weights())
            if len(new_layer_inputs) == 1:
                new_layer = new_layer(new_layer_inputs[0])
            else:
                new_layer = new_layer(new_layer_inputs)
            logger.debug(f"NEW LAYER: {new_layer!r}")
            names_dict[new_layer.name] = layer.name
            new_layers.append(new_layer)
        return tf.keras.Model(inputs=new_inputs, outputs=new_layers[-1], name="reduced")

    def get_reduced_model(self, model):
        """Return the model without the first self.depth layers.

        Extracts the layers between depth and the last from model,
        and copies them to a new smaller model.

        Parameters
        ----------
        model:
            the original model, that we will copy only the last layers

        Returns
        -------
        model.Model:
            A new reduced model, which is the copy of the last layers of the input model,
            removing the first self.depth layers

        """
        if self.depth >= len(model.layers):
            errmsg = "Depth should be smaller than model size."
            raise ValueError(errmsg)

        if self.depth == 0:
            submodel = tf.keras.models.clone_model(model)
            submodel.set_weights(model.get_weights())
            return submodel

        logger.info("Extracting submodel...")

        if isinstance(model, Sequential):
            submodel = self._get_reduced_sequential_model(model)
        elif isinstance(model, tf.keras.Model):
            submodel = self._get_reduced_functional_model(model)
        else:
            errmsg = (
                f"Unknown model type: {type(model)!r}" "Use Functional or Sequential Keras models."
            )
            raise TypeError(errmsg)

        logger.debug("Submodel's summary:")
        logger.debug(submodel.summary())
        logger.info("DONE")
        return submodel
