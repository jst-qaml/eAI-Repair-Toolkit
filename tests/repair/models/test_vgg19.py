from repair.core.loader import load_repair_model
from repair.model.vgg19 import VGG19FineTuningModel


def test_model_loadable():
    loaded_model = load_repair_model("vgg19")

    assert loaded_model is VGG19FineTuningModel


def test_model_structure(imagenet_shape):
    input_shape, output_shape = imagenet_shape

    model = VGG19FineTuningModel().compile(input_shape=input_shape, output_shape=output_shape)

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, output_shape)


def test_model_compile(imagenet_shape):
    input_shape, output_shape = imagenet_shape

    model = VGG19FineTuningModel().compile(input_shape=input_shape, output_shape=output_shape)

    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"
    assert "accuracy" in model.compiled_metrics._metrics
