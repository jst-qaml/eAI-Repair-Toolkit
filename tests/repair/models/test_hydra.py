import json

from tensorflow.keras import optimizers

import pytest

from repair.core.loader import load_repair_model
from repair.model.hydra import HydraModel


@pytest.fixture
def non_compiled_hydra():
    hydra = HydraModel()
    hydra.set_extra_config()

    return hydra


@pytest.fixture
def model_settings_json(tmp_path):
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

    hydra = HydraModel()
    hydra.set_extra_config(model_settings=str(tmp_path / "model_settings.json"), branch_num=3)
    return hydra


@pytest.fixture
def model_settings_json_without_optimizer(tmp_path):
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

    hydra = HydraModel()
    hydra.set_extra_config(model_settings=str(tmp_path / "model_settings.json"), branch_num=3)
    return hydra


def test_model_loadable():
    loaded_model = load_repair_model("hydra")

    assert loaded_model is HydraModel


def test_compile_original_model(imagenet_shape, non_compiled_hydra):
    input_shape, output_shape = imagenet_shape
    model = non_compiled_hydra.compile(input_shape=input_shape, output_shape=output_shape)

    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"
    assert "accuracy" in model.compiled_metrics._metrics


def test_composed_with_json_model(imagenet_shape, model_settings_json):
    input_shape, output_shape = imagenet_shape
    model = model_settings_json.compile(input_shape=input_shape, output_shape=output_shape)

    assert isinstance(model.optimizer, optimizers.Adam)
    assert model.loss == "categorical_crossentropy"
    assert "accuracy" in model.compiled_metrics._metrics


def test_composed_with_json_model_wo_optimizer(
    imagenet_shape, model_settings_json_without_optimizer
):
    input_shape, output_shape = imagenet_shape
    model = model_settings_json_without_optimizer.compile(
        input_shape=input_shape, output_shape=output_shape
    )

    assert isinstance(model.optimizer, optimizers.SGD)
    assert model.loss == "categorical_crossentropy"
    assert "accuracy" in model.compiled_metrics._metrics
