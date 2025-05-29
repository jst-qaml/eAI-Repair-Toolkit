"""Vision transformer model."""

from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

from repair.core import model


class VisionTransferModel(model.RepairModel):
    """keras.applications default model for Search-Based DNN Repair."""

    @classmethod
    def get_name(cls) -> str:
        """Return model name."""
        return "vit"

    def compile(self, input_shape, output_shape):
        """Configure ResNet model.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            A shape of input data
        output_shape : int
            A number of classes

        Returns
        -------
        keras.Model
            A compiled keras model

        """
        data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(self.image_size, self.image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )
        # Compute the mean and the variance of the training data
        # for normalization
        data_augmentation.layers[0].adapt(self.x_train)

        optimizer = keras.optimizers.AdamW(learning_rate=1e-2, weight_decay=1e-3)
        vit_model = self.create_vit_classifier(data_augmentation, input_shape)
        fc = Flatten(input_shape=vit_model.output.shape)(vit_model.output)
        model = Model(inputs=vit_model.input, outputs=fc)
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        )

        return model

    def mlp(self, x, hidden_units, dropout_rate):
        """Multi layer perceptions."""
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def set_extra_config(self, **kwargs):
        """Set extra config for vision transformer using data_dir in kwargs."""
        pre_train_dir = Path(kwargs["data_dir"])
        train_path = pre_train_dir / "train.h5"
        hf = h5py.File(train_path)
        self.x_train = hf["images"]
        self.y_train = hf["labels"]

        # We'll resize input images to this size
        self.image_size = 72
        # Size of the patches to be extract from the input images
        self.patch_size = 6
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        # Size of the transformer layers
        self.transformer_layers = 8
        # Size of the dense layers of the final classifier
        self.mlp_head_units = [2048, 1024]

    def create_vit_classifier(self, data_augmentation, input_shape):
        """Create vision transformer classifier.

        Parameters
        ----------
        data_augmentation : tensorflow.Tensor

        input_shape : tuple[int, int, int]
            A shape of input dataset

        Returns
        -------
        keras.Model
            A raw keras model.

        """
        inputs = layers.Input(shape=input_shape)
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(self.patch_size)(augmented)
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(np.max(self.y_train) + 1)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model


class Patches(layers.Layer):
    """Patches as layers."""

    def __init__(self, patch_size, **kwargs):
        """Patche initialization."""
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        """Return image patches as layer outputs.

        Parameters
        ----------
        images : tensorflow.Tensor

        Returns
        -------
        tensorflow.Tensor
            Same as return type of `tf.image.extract_patches`.
            4D tensor indexed by batch, row, and column.

        """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        """Meaningless override for saving model."""
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    """Patch encoder."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        """Initialize patch encorder."""
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        """Encoding image patches.

        Parameters
        ----------
        patch : tensorflow.Tensor

        Returns
        -------
        tensorflow.Tensor
            Encoded patches

        """
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        """Meaningless override for saving model."""
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
            }
        )
        return config
