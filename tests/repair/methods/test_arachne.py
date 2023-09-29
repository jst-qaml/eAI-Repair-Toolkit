"""Tests for Arachne.

Currently some test do regression test with snapshot output by current arachne.

Todo:
- prepare smaller model
- prepare smallest dataset
- prepare unit test as possible

"""

from pathlib import Path

import numpy as np
from tensorflow import keras

import pytest
import pytest_mock  # noqa

from repair.core.model import load_model_from_tf
from repair.dataset.fashion_mnist import FashionMNIST
from repair.methods.arachne.arachne import Arachne

RESOURCE_FASHION = Path("tests/resources/fashion-mnist")
MODEL_FASHION = RESOURCE_FASHION / "model"
NEGATIVE_FASHION = RESOURCE_FASHION / "negative"
POSITIVE_FASHION = RESOURCE_FASHION / "positive"


@pytest.fixture
def dataset():
    return FashionMNIST()


@pytest.fixture
def init_arachne():
    """Create initialized Arachne.

    Any options are not set to use patched Dataset.
    Required to set options later as necessary.
    """
    return Arachne()


@pytest.fixture
def arachne(init_arachne):
    """Create setup arachne instance."""
    common_options = {
        "num_particles": 1,
        "num_iterations": 1,
        "num_input_pos_sampled": 100,
    }
    init_arachne.set_options(**common_options)

    return init_arachne


@pytest.fixture
def model():
    """Prepare model for test."""
    return load_model_from_tf(MODEL_FASHION)


@pytest.fixture
def neg_input(arachne):
    """Prepare negative inputs data."""
    return arachne.load_input_neg(NEGATIVE_FASHION / "0")


@pytest.fixture
def pos_input(arachne):
    """Prepare positive inputs data."""
    return arachne.load_input_pos(POSITIVE_FASHION)


@pytest.fixture(scope="function")
def keras_model():
    """Create Sequential keras model fixtrue."""
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, kernel_initializer="ones"))
    model.add(keras.layers.Dense(1, kernel_initializer="ones"))

    return model


@pytest.fixture
def output_dir(tmp_path):
    """Prepare temporary output dir for test."""
    tmp_outputs_dir = tmp_path / "outputs"
    tmp_outputs_dir.mkdir()
    return tmp_outputs_dir


@pytest.fixture
def localized_weights():
    """Snapshot for arachne."""
    return [
        [15, 126, 0, 0.17492177],
        [15, 405, 6, 0.19923395],
        [15, 419, 0, 0.17959525],
        [15, 201, 6, 0.14921123],
        [15, 419, 6, 0.19604559],
    ]


@pytest.fixture
def optimized_weights():
    """Snapshot for arachne optimize."""
    return [
        [15, 126, 0, 0.17492177, 0.044712096],
        [15, 405, 6, 0.19923395, -0.12100043],
        [15, 419, 0, 0.17959525, 0.050145146],
        [15, 201, 6, 0.14921123, -0.023170875],
        [15, 419, 6, 0.19604559, -0.0034147718],
    ]


def assert_match_snapshot(weight_path, optimized_weights):
    """Assert output weights match snapshot."""
    import csv

    def parse_weights(value):
        try:
            return int(value)
        except ValueError:
            return float(value)

    with open(weight_path, newline="") as test_weights:
        result_raw = list(csv.reader(test_weights))
        result = list(map(lambda row: list(map(parse_weights, row))[:4], result_raw))

    assert np.allclose(result, list(map(lambda row: row[:4], optimized_weights)))


def test_load_input_neg(arachne, mocker):
    """Test load_input_neg.

    Test to arachne calles dataset's `load_repair_data()` actually.
    """
    dummy_dir = "dir"
    mocked_dataset = mocker.patch(
        "repair.core.dataset.RepairDataset.load_repair_data", return_value=mocker.Mock()
    )

    arachne.load_input_neg(dummy_dir)

    assert mocked_dataset.call_count == 1


def test_load_input_pos(arachne, mocker):
    """Test load_input_pos.

    Test to arachne calles dataset's `load_repair_data()` actually.
    """
    dummy_dir = "dir"
    mocked_dataset = mocker.patch(
        "repair.core.dataset.RepairDataset.load_repair_data", return_value=mocker.Mock()
    )

    arachne.load_input_neg(dummy_dir)

    assert mocked_dataset.call_count == 1


def test_load_weights(init_arachne, mocker):
    """Test load_weights.

    Test to read first 3 elements of each lines.
    Currently no type-converting is performed,
    then compare with string.

    `mocker` is a fixture pytest_mock provides.
    This wraps `unittest.mock` in standard library.

    :param init_arachne: Arachne fixture
    :param mocker: pytest mock fixtrue
    """
    test_data = (
        "2,6167,1,-0.028274775,0.00079285796\n"
        "2,5143,1,-0.031656027,0.014967415\n"
        "2,4119,1,-0.04418554,-0.0026078192"
    )
    mo = mocker.mock_open(read_data=test_data)
    mocker.patch("builtins.open", mo)
    result = init_arachne.load_weights(Path("dummy_dir"))

    assert len(result) == len(test_data.split())

    assert len(result[0]) == 3
    assert result[0][0] == "2"
    assert result[0][1] == "6167"
    assert result[0][2] == "1"


class LocalizeTest:
    """Test group for localize."""

    def test_localize(
        self,
        arachne,
        model,
        neg_input,
        output_dir,
        localized_weights,
    ):
        """Test localize.

        Currently test whether the result matches the `snapshot`.

        :param arachne: Arachne fixture
        :param model: Model fixtrue
        :param neg_input: Dataset fixture
        :param output_dir: Output dir fixture
        :param weights_snapshot: snapshot of weight
        """
        result = arachne.localize(model=model, input_neg=neg_input, output_dir=output_dir)

        assert Path.exists(output_dir / "weights.csv")
        np.testing.assert_allclose(result, localized_weights)

    class ReshapeTargetModelTest:
        """Test group for _reshape_target_model()."""

        def test_raise_exception_when_invalid_layer_index(self, init_arachne):
            """Test the method raises exception when layer is invalid.

            There are 2 cases in which this kind of exception is thrown.
            This tests in case that layer index designates output layer.

            :param init_arachne: Arachne fixture
            """
            test_model = keras.Sequential()
            test_model.add(keras.layers.Add())
            init_arachne.target_layer = len(test_model.layers) - 1

            # suppress ruff because reshape_target_model uses Exception
            with pytest.raises(Exception):  # noqa: B017
                init_arachne._reshape_target_model(test_model, None)

        def test_raise_exception_when_invalid_layer_type(
            self,
            init_arachne,
        ):
            """Test the method raises exception when layer is invalid.

            This test in case that layer type is invalid.

            :param init_arachne: Arachne fixture
            """
            test_model = keras.Sequential()
            test_model.add(keras.layers.Add())
            test_model.add(keras.layers.Add())
            init_arachne.target_layer = 0

            with pytest.raises(Exception) as e:
                init_arachne._reshape_target_model(test_model, None)

            assert "Invalid layer_index" in str(e.value)

        def test_search_target_layer_index_without_enter_loop(
            self,
            init_arachne,
            mocker,
        ):
            """Test the method searches target layer index.

            This tests in case that the latest layer is target layer.
            In this case, the method should return the given model without reshaping.

            :param init_arachne: Arachne fixture
            :param mocker: pytest mocker fixtrue
            """
            test_model = keras.Sequential()
            test_model.add(keras.Input(shape=(2, 1)))
            test_model.add(keras.layers.Dense(1))

            mocked_dataset = mocker.patch("repair.core.eai_dataset.EAIDataset", spec=True)
            mocked_dataset.shape = (2, 1)

            dummy_input_neg = [mocked_dataset] * 10

            result = init_arachne._reshape_target_model(
                model=test_model,
                input_neg=dummy_input_neg,
            )

            assert init_arachne.target_layer == len(test_model.layers) - 1
            assert result is test_model

    class ComputeEachForwardImpactTest:
        """Test group for compute_each_forward_impact."""

        def test_raise_exception_when_invalid_layer_index(self, init_arachne):
            """Test to raise exception when layer index of weight is invalid.

            :param init_arachne: Arachne fixtrue
            """
            test_weight = (-1, 0, 0)

            with pytest.raises(Exception) as e:
                init_arachne._compute_each_forward_impact(
                    None,
                    None,
                    test_weight,
                    None,
                    None,
                )

            assert "Not found" in str(e.value)

        def test_compute_forward_impact(self, init_arachne):
            """Test to compute forward impact.

            Check using index propperly and forward impact is not negative.

            :param init_arachne: Arachne fixtrue
            """
            test_weight = (1, 1, 2)
            test_activations = np.array([[1, 2]])
            test_neuron_weight = np.array([[0.5, 0.75, 0.9], [-0.2, -0.5, -0.75]])

            result = init_arachne._compute_each_forward_impact(
                None,
                None,
                test_weight,
                test_activations,
                test_neuron_weight,
            )

            assert result >= 0
            assert result == 1.5


class OptimizeTest:
    """Test group for optimize."""

    class FailToFindBetterPatchTest:
        """Test group for fail_to_find_better_patch."""

        def test_returns_false_if_still_in_iteration(self, init_arachne):
            """Test to return false if still in iteration.

            :param init_arachne: Arachne fixture
            """
            init_arachne.min_iteration_range = 10
            dummy_history = [(0.0, 0, 0)]

            result = init_arachne._fail_to_find_better_patch(
                t=0,
                history=dummy_history,
            )

            assert result is False

        def test_return_false_if_not_better_patch(self, init_arachne):
            """Test to return false if not better patch.

            :param init_arachne: Arachne fixture
            """
            init_arachne.min_iteration_range = 1
            test_history = [
                (
                    0.5,
                    0,
                    0,
                ),
                (
                    1.0,
                    0,
                    0,
                ),
            ]

            result = init_arachne._fail_to_find_better_patch(t=10, history=test_history)

            assert result is False

        def test_return_true_if_found_better_patch(self, init_arachne):
            """Test to return false if not better patch.

            :param init_arachne: Arachne fixture
            """
            init_arachne.min_iteration_range = 2
            test_history = [
                (
                    1.0,
                    0,
                    0,
                ),
                (
                    0.5,
                    0,
                    0,
                ),
                (
                    0.3,
                    0,
                    0,
                ),
            ]

            result = init_arachne._fail_to_find_better_patch(t=10, history=test_history)

            assert result is True

    def test_optimize(
        self,
        arachne,
        model,
        neg_input,
        pos_input,
        output_dir,
        optimized_weights,
    ):
        """Test optimize.

        :param arachne: Arachne fixture
        :param model: model fixture
        :param neg_input: negative input fixture
        :param pos_input: positive input fixture
        :param output_dir: output directory fixture
        :param optimized_weights: snapshot of weights by optimize
        """
        test_weights = arachne.load_weights(NEGATIVE_FASHION / "0")

        arachne.optimize(
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


class EvaluateTest:
    """Test group for evaluate."""

    def test_evaluate(
        self,
        arachne,
        model,
        dataset,
        neg_input,
        pos_input,
        output_dir,
        localized_weights,
        mocker,
    ):
        """Test evaluate.

        :param arachne: Arachne fixture
        :param model: model fixture
        :param dataset: dataset fixture
        :param neg_input: negative input fixture
        :param pos_input: positive input fixture
        :param output_dir: output directory fixture
        :param localized_weights: snapshot of weights by localize
        :param mocker: pytest mock fixture
        """
        test_num_runs = 1

        mocked_load_weights = mocker.patch(
            "repair.methods.arachne.arachne.Arachne.load_weights"
        )
        mocked_load_weights.return_value = list(
            map(lambda row: row[:3], localized_weights)
        )
        mocked_localize = mocker.patch("repair.methods.arachne.arachne.Arachne.localize")
        mocked_localize.return_value = None
        mocked_optimize = mocker.patch("repair.methods.arachne.arachne.Arachne.optimize")
        mocked_optimize.return_value = None
        mocker.patch(
            "repair.methods.arachne.arachne.load_model_from_tf", return_value=model
        )

        arachne.evaluate(
            dataset=dataset,
            model_dir=MODEL_FASHION,
            target_data=neg_input,
            target_data_dir=NEGATIVE_FASHION / "0",
            positive_inputs=pos_input,
            positive_inputs_dir=POSITIVE_FASHION,
            output_dir=output_dir,
            num_runs=test_num_runs,
        )

        assert Path.exists(output_dir / "result.txt")
        assert mocked_localize.call_count == test_num_runs
        assert mocked_optimize.call_count == test_num_runs
