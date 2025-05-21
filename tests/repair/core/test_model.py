import keras

import pytest

from repair.core.model import RepairModel, train


@pytest.fixture
def repair_model():
    """Tiny keras model for testing."""

    class DemoClass(RepairModel):
        def compile(self, input_shape, output_shape):
            model = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        1, kernel_size=(3, 3), activation="relu", input_shape=input_shape
                    ),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    keras.layers.Flatten(),
                    keras.layers.Dense(1, activation="relu"),
                    keras.layers.Dense(output_shape, activation="softmax"),
                ]
            )

            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            return model

    return DemoClass()


def test_train(repair_model, fashion_mnist_repair_data_dir, tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    train(repair_model, epochs=2, data_dir=fashion_mnist_repair_data_dir, output_dir=output_dir)

    assert (output_dir / "logs").exists()
    assert (
        len(list((output_dir / "logs/model_check_points").glob("weights.*.hdf5"))) > 0
    ), "Model checkpoints are not saved"
    assert len(list(output_dir.glob("*.pb"))) > 0, "Model is not saved in tensorflow format."
