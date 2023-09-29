"""Test for KaminariOpmization."""
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

import pytest

from repair.core.model import load_model_from_tf
from repair.methods.athena.athena import Athena
from repair.methods.kaminari_optimization import KaminariUtils

RESOURCE_DIR = Path("tests/resources/fashion-mnist")
MODEL_DIR = RESOURCE_DIR / "model"

# negative inputs directory
INPUT_NEG_DIR = RESOURCE_DIR / "negative/0"

# positive inputs directory
INPUT_POS_DIR = RESOURCE_DIR / "positive"


class NonFunNonSeq(tf.keras.Model):
    """Keras Model.

    This model is nether Functional nor Sequential.
    """

    def __init__(self):
        """Set layers."""
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(4, activation=tf.nn.relu)

    def call(self, inputs):
        """Calls model on inputs."""
        return self.dense_2(self.dense1(inputs))


class KaminariTest:
    """Unit testing class for kaminariOptimization."""

    batch_size = 1024

    @pytest.fixture
    def input_data(self):
        """Fixture that returns input data to check models."""
        common_settings = {"num_particles": 1, "num_iterations": 1}
        athena = Athena()
        athena.set_options(**common_settings)

        inputs_neg = athena.load_input_neg(INPUT_NEG_DIR)
        inputs_pos = athena.load_input_pos(INPUT_POS_DIR)

        in_ci = (
            os.getenv("GITHUB_ACTIONS")
            or os.getenv("TRAVIS")
            or os.getenv("CIRCLE_CI")
            or os.getenv("GITLAB_CI")
        )

        if not in_ci:
            # we sample positive inputs when running locally to make it faster
            rnd = np.random.default_rng(seed=42)
            inputs_pos_indexes = rnd.choice(
                range(len(inputs_pos[0])), size=20, replace=False
            )
            inputs_pos = (
                inputs_pos[0][inputs_pos_indexes],
                inputs_pos[1][inputs_pos_indexes],
            )

        return inputs_neg, inputs_pos

    @pytest.fixture
    def model(self):
        return load_model_from_tf(MODEL_DIR)

    @pytest.fixture
    def non_fun_non_seq(self):
        """Model that is neither Functional nor Sequential."""
        return NonFunNonSeq()

    def test_should_reject_negative_depth(self):
        """Check that negative depth raises an exception."""
        with pytest.raises(Exception, match="Depth should be positive"):
            KaminariUtils(-1)

    def test_should_reject_too_large_depth(self, model):
        """Check that depth too large raise an exception."""
        kaminari = KaminariUtils(len(model.layers))
        with pytest.raises(Exception, match="Depth should be smaller than model size."):
            kaminari.get_reduced_model(model)

    def test_should_reject_non_func(self, non_fun_non_seq):
        """Check that models not sequential nor functional raise an exception."""
        kaminari = KaminariUtils(1)
        with pytest.raises(Exception, match=r"Unknown model type: .*"):
            kaminari.get_reduced_model(non_fun_non_seq)
        with pytest.raises(Exception, match=r"Unknown model type: .*"):
            kaminari._get_kaminari_inputs(non_fun_non_seq, None)

    def test_get_reduced_model_d0(self, model):
        """Checks that kaminari returns model of same size when depth is 0."""
        kaminari = KaminariUtils(0)
        reduced_model = kaminari.get_reduced_model(model)
        assert len(reduced_model.layers) == len(model.layers)

    def test_get_reduced_model(self, model):
        """Checks that kaminari returns model of the rights size for non 0 depth."""
        depth = int(len(model.layers) / 2)
        kaminari = KaminariUtils(depth)
        reduced_model = kaminari.get_reduced_model(model)

        input_layers = len(
            [
                lay
                for lay in reduced_model.layers
                if isinstance(lay, tf.keras.layers.InputLayer)
            ]
        )
        # we have to add some inputs to represent the "head"
        assert len(reduced_model.layers) == len(model.layers) - depth + input_layers

    def test_get_reduced_model_multi_inputs(self, model):
        """
        Check that kaminari can return the reduced model when cutting
        Enet at a depth when multiple layers feed into the tail.
        """
        depth = len(model.layers) - 1
        kaminari = KaminariUtils(depth)
        reduced_model = kaminari.get_reduced_model(model)

        input_layers = len(
            [
                layer
                for layer in reduced_model.layers
                if isinstance(layer, tf.keras.layers.InputLayer)
            ]
        )
        # we have to add some inputs to represent the "head"
        assert len(reduced_model.layers) == len(model.layers) - depth + input_layers

    def test_save_processed_images_d0(self, model, input_data):
        """
        Check that the reduced (copy) model returns the same result on
        the original inputs as the original model on the new (copy) inputs
        for depth 0.
        """
        orig_neg, orig_pos = input_data

        kaminari = KaminariUtils(0)
        reduced_model = kaminari.get_reduced_model(model)
        new_neg, new_pos = kaminari.save_processed_images(
            model, orig_neg, orig_pos, batch_size=self.batch_size
        )

        # original model inference on original data
        orig_pos_pred = model.predict(new_pos[0], batch_size=self.batch_size)
        orig_neg_pred = model.predict(new_neg[0], batch_size=self.batch_size)

        # reduced model inference on features
        reduced_pos_pred = reduced_model.predict(orig_pos[0], batch_size=self.batch_size)
        reduced_neg_pred = reduced_model.predict(orig_neg[0], batch_size=self.batch_size)

        np.testing.assert_allclose(orig_pos_pred, reduced_pos_pred)
        np.testing.assert_allclose(orig_neg_pred, reduced_neg_pred)

    def test_save_processed_images(self, model, input_data):
        """
        Check that the reduced model returns the same result on the new
        inputs as the original model on the original inputs for a non-zero
        depth.
        """
        orig_neg, orig_pos = input_data

        kaminari = KaminariUtils(int(len(model.layers) / 2))
        reduced_model = kaminari.get_reduced_model(model)
        new_neg, new_pos = kaminari.save_processed_images(
            model, orig_neg, orig_pos, batch_size=self.batch_size
        )

        # original model inference on original data
        orig_pos_pred = model.predict(orig_pos[0], batch_size=self.batch_size)
        orig_neg_pred = model.predict(orig_neg[0], batch_size=self.batch_size)

        # reduced model inference on features
        reduced_pos_pred = reduced_model.predict(new_pos[0], batch_size=self.batch_size)
        reduced_neg_pred = reduced_model.predict(new_neg[0], batch_size=self.batch_size)

        np.testing.assert_allclose(orig_pos_pred, reduced_pos_pred)
        np.testing.assert_allclose(orig_neg_pred, reduced_neg_pred)

    def test_save_processed_images_multiple_inputs(self, model, input_data):
        """
        Check that the reduced model (copy model) returns the same result
        on the new inputs as the original model on the original inputs when
        the reduced model have multiple inputs.
        """
        orig_neg, orig_pos = input_data

        kaminari = KaminariUtils(len(model.layers) - 1)
        reduced_model = kaminari.get_reduced_model(model)
        new_neg, new_pos = kaminari.save_processed_images(
            model, orig_neg, orig_pos, batch_size=self.batch_size
        )

        # original model inference on original data
        orig_pos_pred = model.predict(orig_pos[0], batch_size=self.batch_size)
        orig_neg_pred = model.predict(orig_neg[0], batch_size=self.batch_size)

        # reduced model inference on features
        reduced_pos_pred = reduced_model.predict(new_pos[0], batch_size=self.batch_size)
        reduced_neg_pred = reduced_model.predict(new_neg[0], batch_size=self.batch_size)

        np.testing.assert_allclose(orig_pos_pred, reduced_pos_pred)
        np.testing.assert_allclose(orig_neg_pred, reduced_neg_pred)
