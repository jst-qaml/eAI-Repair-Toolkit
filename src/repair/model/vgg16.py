"""VGG16 fine tuning model."""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from repair.core import model


class VGG16FineTuningModel(model.RepairModel):
    """VGG16 tuned for Search-Based DNN Repair."""

    @classmethod
    def get_name(cls) -> str:
        """Return model name."""
        return "vgg16"

    def compile(self, input_shape, output_shape):
        """Configure VGG16 model.

        :param input_shape:
        :param output_shape:
        :return: model
        """
        # Use VGG16 as basis
        vgg_model = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            input_tensor=Input(shape=input_shape),
        )

        # cf. "SB-Repair of DNNs" says:
        # After the VGG16, we added two dense layers
        # (DENSE(4096), DENSE(4096)),
        # followed by the final layer for labels(DENSE(43)).
        fc = Flatten(input_shape=vgg_model.output.shape)(vgg_model.output)
        fc = Dense(4096, activation="relu")(fc)
        fc = Dense(4096, activation="relu")(fc)
        predictions = Dense(output_shape, activation="softmax")(fc)

        model = Model(inputs=vgg_model.input, outputs=predictions)

        # Compile model
        model.compile(
            optimizer=SGD(lr=1e-4, momentum=0.9),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
