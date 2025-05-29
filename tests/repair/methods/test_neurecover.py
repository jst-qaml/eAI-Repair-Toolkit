"""Tests for NeuRecover.

Currently some test do regression test with snapshot output by current neurecover.

Todo:
- prepare smallest dataset
- prepare unit test as possible

"""

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

import pytest

from repair.core.model import load_model_from_tf
from repair.dataset.fashion_mnist import FashionMNIST
from repair.methods.neurecover import NeuRecover
from tests.repair.methods.test_arachne import assert_match_snapshot

NEURECOVER_RESOURCES = Path("tests/resources/neurecover")
MODEL = NEURECOVER_RESOURCES / "models"


tf.get_logger().setLevel(logging.WARNING)


@pytest.fixture
def dataset():
    return FashionMNIST()


@pytest.fixture
def init_neurecover():
    """Create initialized NeuRecover.

    Any options are not set to use patched Dataset.
    Required to set options later as necessary.
    """
    return NeuRecover()


@pytest.fixture
def neurecover(init_neurecover):
    """Create setup neurecover instance."""
    common_options = {
        "num_particles": 1,
        "num_iterations": 1,
        "num_input_pos_sampled": 100,
        "weights_dir": MODEL / "logs" / "model_check_points",
        "positive_data_dir": NEURECOVER_RESOURCES / "positive",
    }
    init_neurecover.set_options(**common_options)

    return init_neurecover


@pytest.fixture
def model():
    """Prepare model for test."""
    return load_model_from_tf(MODEL)


@pytest.fixture
def neg_input(neurecover):
    return neurecover.load_input_neg(NEURECOVER_RESOURCES / "negative/0")


@pytest.fixture
def pos_input(neurecover):
    return neurecover.load_input_pos(NEURECOVER_RESOURCES / "positive")


@pytest.fixture
def output_dir(tmp_path):
    """Prepare temporary output dir for test."""
    tmp_outputs_dir = tmp_path / "outputs"
    tmp_outputs_dir.mkdir()
    return tmp_outputs_dir


@pytest.fixture
def localized_weights():
    """Snapshot for neurecover."""
    return [
        [15, 16, 6],
        [15, 5, 5],
        [15, 5, 7],
    ]


@pytest.fixture
def optimized_weights():
    """Snapshot for arachne optimize."""
    return [
        [15, 16, 6, 0.2950529, -0.053359367],
        [15, 5, 5, 0.281062, -0.040111978],
        [15, 5, 7, 0.23827223, 0.079375975],
    ]


class LocalizeTest:
    """Test group for localize."""

    def test_localize(
        self,
        neurecover,
        model,
        neg_input,
        output_dir,
        localized_weights,
    ):
        result = neurecover.localize(model=model, input_neg=neg_input, output_dir=output_dir)

        assert Path.exists(output_dir / "weights.csv")
        np.testing.assert_allclose(result, localized_weights)


class OptimizeTest:
    """Test group for optimize."""

    def test_optimize(
        self,
        neurecover,
        model,
        neg_input,
        pos_input,
        output_dir,
        optimized_weights,
    ):
        test_weights = neurecover.load_weights(NEURECOVER_RESOURCES / "negative" / "0")

        neurecover.optimize(
            model=model,
            weights=test_weights,
            input_neg=neg_input,
            input_pos=pos_input,
            output_dir=output_dir,
            model_dir=None,
        )

        assert Path.exists(output_dir / "repair")
        assert Path.exists(output_dir / "weights.csv")

        assert_match_snapshot(output_dir / "weights.csv", optimized_weights)
