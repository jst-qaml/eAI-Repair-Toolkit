import json

import pytest

from repair.core.loader import load_repair_model
from repair.model.keras_app import KerasModel


@pytest.fixture
def non_compiled_kerasapp(tmp_path):
    with open(tmp_path / "model_settings.json", "w") as f:
        json.dump(
            {
                "model": "MobileNetV2",
                "augmentation": [
                    ["RandomFlip", {"mode": "horizontal"}],
                    ["RandomRotation", {"factor": 0.2}],
                ],
                "layers": [
                    [
                        "Conv2D",
                        {"filters": 32, "kernel_size": 3, "activation": "relu", "name": "Conv2D1"},
                    ],
                    ["Dense", {"units": 128, "activation": "relu", "name": "Dense1"}],
                ],
                "optimizer": ["Adam", {"learning_rate": 0.2}],
            },
            f,
            ensure_ascii=False,
        )

    kerasapp = KerasModel()
    kerasapp.set_extra_config(model_settings=str(tmp_path / "model_settings.json"))

    return kerasapp


@pytest.fixture
def non_compiled_kerasapp_wo_optimizer(tmp_path):
    with open(tmp_path / "model_settings.json", "w") as f:
        json.dump(
            {
                "model": "MobileNetV2",
                "augmentation": [
                    ["RandomFlip", {"mode": "horizontal"}],
                    ["RandomRotation", {"factor": 0.2}],
                ],
                "layers": [
                    [
                        "Conv2D",
                        {"filters": 32, "kernel_size": 3, "activation": "relu", "name": "Conv2D1"},
                    ],
                    ["Dense", {"units": 128, "activation": "relu", "name": "Dense1"}],
                ],
            },
            f,
            ensure_ascii=False,
        )

    kerasapp = KerasModel()
    kerasapp.set_extra_config(model_settings=str(tmp_path / "model_settings.json"))
    return kerasapp


def test_model_loadable():
    loaded_model = load_repair_model("keras_app")

    assert loaded_model is KerasModel


def test_compile_original_model(imagenet_shape, non_compiled_kerasapp):
    input_shape, output_shape = imagenet_shape
    model = non_compiled_kerasapp.compile(input_shape=input_shape, output_shape=output_shape)

    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"
    assert "accuracy" in model.compiled_metrics._metrics


def test_compile_original_model_without_optimizer(
    imagenet_shape, non_compiled_kerasapp_wo_optimizer
):
    input_shape, output_shape = imagenet_shape
    model = non_compiled_kerasapp_wo_optimizer.compile(
        input_shape=input_shape, output_shape=output_shape
    )

    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"
    assert "accuracy" in model.compiled_metrics._metrics
