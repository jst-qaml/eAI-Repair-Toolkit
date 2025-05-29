"""DNN models of HydraNet.

Source paper:
"HydraNets: Specialized Dynamic Architectures for Efficient Inference"
"""
from pathlib import Path

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from repair.core import model
from repair.model.hydra import Combine


class HydraHeadModel(model.RepairModel):
    """Combining fine-tuned DNN models by hydra."""

    @classmethod
    def get_name(cls) -> str:
        """Return model name."""
        return "hydra_head"

    def compile(self, input_shape, output_shape):
        """Load and combine fine tuned model.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
        output_shape : int

        Returns
        -------
        keras.Model
            A compiled model

        """
        input_img = Input(shape=input_shape)
        gate_model, branch_models = self._load_hydra_materials(self.hydra_head_dir)

        if len(branch_models) > 0:
            branch_outs = []
            for idx in range(len(branch_models)):
                branch = branch_models[idx]
                branch_outs.append(self._copy_layers(branch, input_img, f"branch{str(idx)}"))
            gate_out = self._copy_layers(gate_model, input_img, "gate")
            branch_outs.append(gate_out)

        else:
            stem_out = self._copy_layers(gate_model, input_img, "gate", True)
            gate_out = gate_model.layers[-1](stem_out)

            branch_outs = []

            for idx in range(gate_out.shape[1]):
                b_out = Dense(
                    output_shape,
                    activation="softmax",
                    name=f"Branch{str(idx)}_Dense",
                )(stem_out)
                branch_outs.append(b_out)
            branch_outs.append(gate_out)

        out = Combine()(branch_outs)
        model = Model(inputs=input_img, outputs=out)

        model.compile(
            optimizer=Adam(learning_rate=2.5e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def set_extra_config(self, **kwargs):
        """Set extra config for Combine layer."""
        if "hydra_head_dir" in kwargs:
            self.hydra_head_dir = Path(kwargs["hydra_head_dir"])
        else:
            raise TypeError("Require --hydra_head_dir")

    def _load_model_with_weights(self, model_dir: Path):
        """Load model from SavedModel.

        Parameters
        ----------
        model_dir: str | Path
            Path to directory containing model files

        Returns
        -------
        model : tf.keras.Model
            Loaded model

        """
        reconstructed_model = keras.models.load_model(Path(model_dir))
        return reconstructed_model

    def _load_hydra_materials(self, hydra_head_dir: Path):
        gate_dir = hydra_head_dir / "gate"
        gate_model = self._load_model_with_weights(gate_dir)

        branch_dir = hydra_head_dir / "branch"
        if not branch_dir.exists():
            return gate_model, []

        branch_models = []
        for b_dir in branch_dir.iterdir():
            model_dir = branch_dir / b_dir
            if model_dir.is_dir():
                model = self._load_model_with_weights(model_dir)
                branch_models.append(model)

        return gate_model, branch_models

    def _copy_layers(self, model, layer_input, name, exclude_final=False):
        """Copy the hidden layers and the output layer.

        Parameters
        ----------
        model : keras.Model
            A copy target model
        layer_input :
            input layer
        name : str
            name to distinguish the layers
        exclude_final : bool
            Whether ignoring the final layer

        Returns
        -------
            the final output of copied layers

        """
        copy_start_idx = 0

        first_layer = model.layers[0]
        if isinstance(first_layer, keras.layers.InputLayer):
            copy_start_idx = 1

        out = layer_input
        num_layers = len(model.layers)
        if exclude_final:
            num_layers -= 1
        for idx in range(copy_start_idx, num_layers):
            layer = model.layers[idx]
            layer._name = f"{name}_{layer.name}"
            out = model.layers[idx](out)
        return out
