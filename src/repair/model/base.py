"""Base CNN model."""
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

from repair.core import model


class BaseCNNModel(model.RepairModel):
    """Base CNN model."""

    @classmethod
    def get_name(cls) -> str:
        """Return model name."""
        return "base"

    def compile(self, input_shape, output_shape):
        """Configure base CNN model.

        :param input_shape:
        :param output_shape:
        :return: model
        """
        model = Sequential()

        model.add(
            Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu")
        )
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

        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(
            optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model
