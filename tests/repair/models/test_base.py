from repair.core.loader import load_repair_model
from repair.model.base import BaseCNNModel


def test_model_loadable():
    loaded_model = load_repair_model("base")

    assert loaded_model is BaseCNNModel


def test_model_structure():
    input_shape = (86, 86, 3)
    output_shape = 10

    model = BaseCNNModel().compile(input_shape=input_shape, output_shape=output_shape)

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, output_shape)


def test_model_compile():
    input_shape = (86, 86, 3)
    output_shape = 10

    model = BaseCNNModel().compile(input_shape=input_shape, output_shape=output_shape)

    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"
    assert "accuracy" in model.compiled_metrics._metrics
