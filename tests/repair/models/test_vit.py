import pytest

from repair.core.loader import load_repair_model
from repair.model.vit import VisionTransferModel
from repair.utils import create_vit_class


@pytest.fixture
def prepare_vit_class(fashion_mnist_repair_data_dir):
    create_vit_class.run(data_dir=fashion_mnist_repair_data_dir)

    return fashion_mnist_repair_data_dir / "vision_transformer"


@pytest.fixture
def non_compiled_model(prepare_vit_class):
    data_dir = prepare_vit_class
    vit = VisionTransferModel()
    vit.set_extra_config(data_dir=data_dir)
    return vit


def test_model_loadable():
    loaded_model = load_repair_model("vit")

    assert loaded_model is VisionTransferModel


def test_model_compile(non_compiled_model, fashionmnist_shape):
    input_shape, output_shape = fashionmnist_shape

    model = non_compiled_model.compile(input_shape=input_shape, output_shape=output_shape)

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, output_shape)
