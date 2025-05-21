"""Base CNN model."""
from __future__ import annotations

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay

from repair.core import model


class BaseCNNModel(model.RepairModel):
    """Base CNN model."""

    @classmethod
    def get_name(cls) -> str:
        """Return model name."""
        return "base"

    def compile(self, input_shape: tuple[int, int, int], output_shape: int):
        """Configure base CNN model.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
        output_shape : int

        Returns
        -------
        keras.Model
            A compiled keras model

        """
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu"))
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation="softmax"))

        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=1e-6,
        )

        opt = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        return model
