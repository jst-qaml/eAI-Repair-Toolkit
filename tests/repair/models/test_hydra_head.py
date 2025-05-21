import json
import shutil

import pytest

from repair.core.loader import load_repair_model
from repair.model.hydra_head import HydraHeadModel
from repair.utils import create_gate_class


@pytest.fixture
def prepare_gate_class(fashion_mnist_repair_data_dir, pretrained_keras_model_dir):
    hydra_dir = fashion_mnist_repair_data_dir
    with open(hydra_dir / "hydra_fmnist.json", "w") as f:
        json.dump([[0, 2, 4, 6], [1, 3], [5, 7, 8, 9]], f)

    create_gate_class.run(
        data_dir=hydra_dir, hydra_setting_file=str(hydra_dir / "hydra_fmnist.json")
    )
    for asset in pretrained_keras_model_dir.iterdir():
        dst = hydra_dir / "gate" / asset.name
        if asset.is_dir():
            shutil.copytree(asset, dst)
        else:
            shutil.copy(asset, dst)

    return hydra_dir


@pytest.fixture
def non_compiled_model(prepare_gate_class):
    gate_dir = prepare_gate_class

    hydra_head = HydraHeadModel()
    hydra_head.set_extra_config(hydra_head_dir=gate_dir)
    return hydra_head


def test_model_loadable():
    loaded_model = load_repair_model("hydra_head")

    assert loaded_model is HydraHeadModel


def test_model_structure(non_compiled_model, fashionmnist_shape):
    input_shape, output_shape = fashionmnist_shape

    model = non_compiled_model.compile(input_shape=input_shape, output_shape=output_shape)

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, output_shape)


def test_model_compile(non_compiled_model, fashionmnist_shape):
    input_shape, output_shape = fashionmnist_shape

    model = non_compiled_model.compile(input_shape=input_shape, output_shape=output_shape)

    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"
    assert "accuracy" in model.compiled_metrics._metrics
