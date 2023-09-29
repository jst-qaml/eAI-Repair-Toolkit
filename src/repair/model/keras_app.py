"""DNN models in keras.applications."""

import importlib
import json
from pathlib import Path

from tensorflow.keras import Sequential, applications, layers, optimizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from repair.core import model


class KerasModel(model.RepairModel):
    """keras.applications default model for Search-Based DNN Repair."""

    @classmethod
    def get_name(cls) -> str:
        """Return model name."""
        return "keras_app"

    def compile(self, input_shape, output_shape):
        """Configure ResNet model.

        :param input_shape:
        :param output_shape:
        :return: model
        """
        input_img = Input(shape=input_shape)

        if "augmentation" in self.tuning:
            augmentation_list = []
            for augment in self.tuning["augmentation"]:
                aug_instance = getattr(layers, augment[0])(**augment[1])
                augmentation_list.append(aug_instance)
            data_augmentation = Sequential(augmentation_list)
            input_img = data_augmentation(input_img)

        importlib.invalidate_caches()
        instance = getattr(applications, self.tuning["model"])(
            include_top=False,
            weights="imagenet",
            input_tensor=input_img,
            input_shape=input_shape,
            pooling=None,
        )
        out = instance.output

        if "layers" in self.tuning:
            for tuning in self.tuning["layers"]:
                try:
                    out = getattr(layers, tuning[0])(**tuning[1])(out)
                except AttributeError as ae:
                    raise AttributeError("Cannot import " + tuning[0]) from ae
                except TypeError as te:
                    raise TypeError("Wrong inputs for " + tuning[0]) from te

        predictions = Dense(output_shape, activation="softmax")(out)

        model = Model(inputs=instance.input, outputs=predictions)

        # Compile model
        if "optimizer" in self.tuning:
            opt_name = self.tuning["optimizer"][0]
            opt_kwargs = self.tuning["optimizer"][1]
            opt_instance = getattr(optimizers, opt_name)(**opt_kwargs)
        else:
            opt_instance = optimizers.SGD(lr=1e-4, momentum=0.9)

        model.compile(
            optimizer=opt_instance, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    def set_extra_config(self, **kwargs):
        """Designate keras.applications."""
        if "model_settings" in kwargs:
            tuning_path = Path(kwargs["model_settings"])
            with open(tuning_path) as json_file:
                self.tuning = json.load(json_file)
        else:
            raise TypeError("Require --model_settings")
