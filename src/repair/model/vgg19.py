"""VGG19 fine tuning model."""
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from repair.core import model


class VGG19FineTuningModel(model.RepairModel):
    """VGG19 fine tuning model."""

    @classmethod
    def get_name(cls) -> str:
        """Return model name."""
        return "vgg19"

    def compile(self, input_shape, output_shape):
        """Configure VGG19 model.

        :param input_shape:
        :param output_shape:
        :return: model
        """
        input_tensor = Input(shape=input_shape)

        vgg_model = VGG19(include_top=False, weights="imagenet", input_tensor=input_tensor)

        for layer in vgg_model.layers:
            layer.trainable = False

        x = Flatten(input_shape=vgg_model.output.shape)(vgg_model.output)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        predictions = Dense(output_shape, activation="softmax")(x)

        model = Model(inputs=vgg_model.input, outputs=predictions)

        opt = SGD(lr=1e-4, momentum=0.9)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        return model
