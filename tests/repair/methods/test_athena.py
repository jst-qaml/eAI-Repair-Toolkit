"""Test for Athena."""

import copy
import shutil
from pathlib import Path

import numpy as np

import pytest

from repair.core.dataset import RepairDataset
from repair.core.model import load_model_from_tf
from repair.methods.athena.athena import Athena

# TODO: For now these resources are preserved to keep test_criterion soundness.
#       Make appropriate dataset with fashion-mnist.
RESOURCE_DIR = Path("./tests/resources/outputs_bdd_objects/")
MODEL_DIR = RESOURCE_DIR / "model"

# labels.json in this dir has only one setting that "repair_priorty": 1
SINGLE_INPUT_NEG_DIR = RESOURCE_DIR / "negative"

# labels.json in this dir has sevelal settings that "repair_priorty": 1
MULTI_INPUT_NEG_DIR = RESOURCE_DIR / "negative" / "0"

# labels.json in this dir has weights
WEIGHTS_INPUT_NEG_DIR = RESOURCE_DIR / "negative" / "1"

# labels.json in this dir does not have any labels to be localized
NO_INPUT_NEG_DIR = RESOURCE_DIR / "negative" / "2"

DONE_INPUT_NEG_DIR = RESOURCE_DIR / "negative" / "3"

# positive inputs directory
INPUT_POS_DIR = RESOURCE_DIR / "positive"


RESOURCE_MNIST = Path("tests/resources/fashion-mnist")
MODEL_FASHION = RESOURCE_MNIST / "model"
NEGATIVE_FASHION = RESOURCE_MNIST / "negative"
NEGATIVE_FASHION_0 = NEGATIVE_FASHION / "0"
POSITIVE_FASHION = RESOURCE_MNIST / "positive"
FASHION_WEIGHTS_DIR = NEGATIVE_FASHION_0


class AthenaTest:
    """Test class for Athena."""

    @pytest.fixture
    def athena(self):
        """Create Athena module."""
        common_settings = {"num_particles": 1, "num_iterations": 1}

        athena = Athena()
        athena.set_options(**common_settings)
        return athena

    @pytest.fixture
    def weights(self, athena, use_weights_label):
        loaded_weights = athena.load_weights(NEGATIVE_FASHION_0)
        _weights = []
        for w in loaded_weights:
            _weights += loaded_weights[w]["weights"]
        return _weights

    @pytest.fixture
    def personal_best_scores(self, athena, weights):
        """Get several values for find patch."""
        model = load_model_from_tf(MODEL_FASHION)

        input_pos = athena.load_input_pos(POSITIVE_FASHION)
        input_pos_sampled = athena._sample_positive_inputs(input_pos)
        input_neg = athena.load_input_neg(NEGATIVE_FASHION_0)

        locations = athena._get_initial_particle_positions(
            weights, model, athena.num_particles
        )
        personal_best_scores = athena._initialize_personal_best_scores(
            locations, model, input_pos_sampled, input_neg
        )

        return personal_best_scores

    @pytest.fixture
    def best_particle(self, personal_best_scores):
        return np.argmax(np.array(personal_best_scores)[:, 0])

    @pytest.fixture
    def history(self, personal_best_scores, best_particle):
        return [personal_best_scores[best_particle]]

    @pytest.fixture
    def dump_restore_single_file(self, tmp_path):
        """Evacutate and restore the original file."""
        shutil.copyfile(NEGATIVE_FASHION_0 / "labels.json", tmp_path / "labels.json")

        yield

        shutil.move(tmp_path / "labels.json", NEGATIVE_FASHION_0 / "labels.json")

    @pytest.fixture(scope="function")
    def use_no_priority_label(self, tmp_path):
        shutil.copyfile(
            NEGATIVE_FASHION_0 / "labels.json",
            tmp_path / "labels.json",
        )
        shutil.copyfile(
            NEGATIVE_FASHION_0 / "labels_no_priority.json",
            NEGATIVE_FASHION_0 / "labels.json",
        )

        yield

        shutil.copyfile(
            tmp_path / "labels.json",
            NEGATIVE_FASHION_0 / "labels.json",
        )

    @pytest.fixture(scope="function")
    def use_weights_label(self, tmp_path):
        """Evacutate and restore the original file."""
        shutil.copyfile(
            NEGATIVE_FASHION_0 / "labels.json",
            tmp_path / "labels.json",
        )
        shutil.copyfile(
            NEGATIVE_FASHION_0 / "labels_with_weight.json",
            NEGATIVE_FASHION_0 / "labels.json",
        )

        yield

        shutil.move(
            tmp_path / "labels.json",
            NEGATIVE_FASHION_0 / "labels.json",
        )

    @pytest.fixture()
    def use_weights_csv(self, tmp_path):
        origin = NEGATIVE_FASHION_0 / "6" / "weights.csv"
        stashed = tmp_path / "weights.csv"
        shutil.copyfile(origin, stashed)

        yield

        shutil.move(stashed, origin)

    @pytest.fixture(scope="function")
    def use_optimized_label(self, tmp_path):
        shutil.copyfile(
            NEGATIVE_FASHION_0 / "labels.json",
            tmp_path / "labels.json",
        )
        shutil.copyfile(
            NEGATIVE_FASHION_0 / "labels_optimized.json",
            NEGATIVE_FASHION_0 / "labels.json",
        )

        yield

        shutil.move(
            tmp_path / "labels.json",
            NEGATIVE_FASHION_0 / "labels.json",
        )

    def test_load_input_neg_one(self, athena):
        """Load data of only one negative label.

        This test checks that the function loads a correct dataset.
        This test uses a directory whose labels.json has only one setting that
        "repair_priority: 1" which means "Reading only one data file".
        """
        imgs, labels = athena.load_input_neg(NEGATIVE_FASHION)
        # Check the type of loaded input
        assert type(imgs) == np.ndarray
        assert type(labels) == np.ndarray

        # Check whether the number of imgs and labels are the same
        assert len(imgs) == len(labels)

        # Check whther the number of imgs and labels are same as the target
        dataset = RepairDataset.load_repair_data(NEGATIVE_FASHION_0)
        expected_imgs = dataset[0]
        expected_labels = dataset[1]

        assert len(imgs) == len(expected_imgs)
        assert len(labels) == len(expected_labels)

    def test_load_input_neg_multi(self, athena):
        """Load data of only one negative label.

        This test checks that the function loads a correct dataset.
        This test uses a directory whose labels.json has two settings that
        "repair_priority: 1" which means "Reading two data file".

        """
        imgs, labels = athena.load_input_neg(NEGATIVE_FASHION / "2")

        # Check the type of loaded input
        assert type(imgs) == np.ndarray
        assert type(labels) == np.ndarray

        # Check whether the number of imgs and labels are the same
        assert len(imgs) == len(labels)

        dataset = RepairDataset.load_repair_data(NEGATIVE_FASHION / "2")
        expected_imgs = dataset[0]
        expected_labels = dataset[1]

        assert len(imgs) == len(expected_imgs)
        assert len(labels) == len(expected_labels)

    def test_load_input_neg_raise_error_when_empty(self, athena, use_no_priority_label):
        with pytest.raises(ValueError):
            athena.load_input_neg(NEGATIVE_FASHION_0)

    def test_load_input_pos(self, athena):
        """Load data of only one negative label.

        This test checks that the function loads a correct dataset.
        """
        imgs, labels = athena.load_input_pos(POSITIVE_FASHION)

        # Check the type of loaded input
        assert type(imgs) == np.ndarray
        assert type(labels) == np.ndarray

        # Check whether the number of imgs and labels are the same
        assert len(imgs) == len(labels)

        # Check whether loading all the positive inputs
        dataset = RepairDataset.load_repair_data(POSITIVE_FASHION)
        expected_imgs = dataset[0]
        expected_labels = dataset[1]
        assert len(imgs) == len(expected_imgs)
        assert len(labels) == len(expected_labels)

        # Check whether setting protected inputs with the designated rate
        target_data = POSITIVE_FASHION / "0"

        dataset = RepairDataset.load_repair_data(target_data)
        expected_imgs = dataset[0]
        expected_labels = dataset[1]

        protected_imgs, protected_labels = athena.input_protected
        assert len(protected_imgs) == len(expected_imgs)
        assert len(protected_labels) == len(expected_labels)

    def test_load_weights(self, athena, use_weights_label):
        """Load weights."""
        weights = athena.load_weights(NEGATIVE_FASHION_0)

        # Check whether loading the correct weights
        assert len(weights) == 1
        assert weights["6"]["weights"] == [
            [15, 126, 0],
            [15, 405, 6],
            [15, 419, 0],
            [15, 201, 6],
            [15, 419, 6],
        ]

    def test_sample_positive_inputs_designated(self, athena):
        """Sample positive inputs with the designated number."""
        kwargs = {"num_input_pos_sampled": 20}
        athena.set_options(**kwargs)
        input_pos = athena.load_input_pos(POSITIVE_FASHION)
        sampled = athena._sample_positive_inputs(input_pos)

        # Check the number of sampled inputs
        assert len(sampled[0]) == 20
        assert len(sampled[1]) == 20

    def test_sample_positive_inputs_none(self, athena):
        """Do not sample positive inputs."""
        kwargs = {"num_input_pos_sampled": None}
        athena.set_options(**kwargs)
        input_pos = athena.load_input_pos(POSITIVE_FASHION)
        sampled = athena._sample_positive_inputs(input_pos)

        np.testing.assert_allclose(input_pos[0], sampled[0])
        np.testing.assert_allclose(input_pos[1], sampled[1])

    def test_fail_to_find_better_patch(
        self, athena, personal_best_scores, best_particle, history
    ):
        """Tests for fail_to_find_better_patch with several settings.

        Tests for this method is buched because the initialize for this
        method is time consuming.
        """
        # Return false because the iteration num is 1
        assert not athena._fail_to_find_better_patch(1, history)
        # Return True because failured in finding the better score
        for _ in range(10):
            history.append(personal_best_scores[best_particle])
        personal_best_scores[best_particle][1] = 1
        assert athena._fail_to_find_better_patch(11, history)

        # Return False because succeeded in finding the better score
        new_score = copy.copy(personal_best_scores[best_particle])
        new_score[0] = 200
        history.append(new_score)
        assert not athena._fail_to_find_better_patch(12, history)

        # Return False because failured in finding the better patch
        personal_best_scores[best_particle][1] = -1
        history.remove(new_score)
        assert not athena._fail_to_find_better_patch(11, history)

    def test_criterion(self, athena):
        """Return correct type values."""
        model = load_model_from_tf(MODEL_DIR)
        weights = athena.load_weights(WEIGHTS_INPUT_NEG_DIR)
        _weights = []
        for w in weights:
            _weights = _weights + weights[w]["weights"]

        input_pos = athena.load_input_pos(INPUT_POS_DIR)
        input_pos_sampled = athena._sample_positive_inputs(input_pos)
        input_neg = athena.load_input_neg(WEIGHTS_INPUT_NEG_DIR)

        locations = athena._get_initial_particle_positions(
            _weights, model, athena.num_particles
        )

        print(len(athena.input_protected[0]))

        fitness, n_patched, n_intact = athena._criterion(
            model, locations[0], input_pos_sampled, input_neg
        )

        assert fitness == -1
        assert type(n_patched) == int
        assert type(n_intact) == int

    def test_localize(self, athena, dump_restore_single_file):
        """Execute localize soundly."""
        model = load_model_from_tf(MODEL_FASHION)
        input_neg = athena.load_input_neg(NEGATIVE_FASHION_0)

        athena.localize(model, input_neg, NEGATIVE_FASHION_0)

        req = athena._load_requirements(NEGATIVE_FASHION_0)

        for label in req:
            assert "weights" in req[label]
            assert type(req[label]["weights"]) == list

    def test_skipping_localize(self, athena, dump_restore_single_file):
        """Skip localize function with the already localied weights."""
        model = load_model_from_tf(MODEL_FASHION)
        input_neg = athena.load_input_neg(NEGATIVE_FASHION_0)

        athena.localize(model, input_neg, NEGATIVE_FASHION_0)

    def test_optimize(self, athena, use_weights_label, use_weights_csv):
        """Execute optimize soundly."""
        model = load_model_from_tf(MODEL_FASHION)
        weights = athena._load_requirements(NEGATIVE_FASHION_0)
        input_pos = athena.load_input_pos(POSITIVE_FASHION)

        athena.optimize(
            model,
            MODEL_FASHION,
            weights,
            NEGATIVE_FASHION_0,
            input_pos,
            NEGATIVE_FASHION_0,
        )

        req = athena._load_requirements(NEGATIVE_FASHION_0)

        for label in req:
            assert "repaired_values" in req[label]
            assert len(req[label]["repaired_values"]) == len(req[label]["weights"])

    def test_skip_optimize(self, athena, use_optimized_label):
        """Skip optimize function with already optimized weights."""
        model = load_model_from_tf(MODEL_FASHION)
        weights = athena._load_requirements(NEGATIVE_FASHION_0)
        input_pos = athena.load_input_pos(POSITIVE_FASHION)

        athena.optimize(
            model,
            MODEL_FASHION,
            weights,
            NEGATIVE_FASHION_0,
            input_pos,
            NEGATIVE_FASHION_0,
        )

    def test_evaluate(self, athena):
        """Do nothing because evaluate is not implemented."""
        athena.evaluate(None, None, None, None, None, None, None, None)
