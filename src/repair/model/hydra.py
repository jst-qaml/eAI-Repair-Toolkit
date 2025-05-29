"""DNN models of HydraNet.

Source paper:
"HydraNets: Specialized Dynamic Architectures for Efficient Inference"
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers, optimizers
from tensorflow.keras.layers import (
    Add,
    AveragePooling2D,
    CategoryEncoding,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    SeparableConv2D,
)
from tensorflow.keras.models import Model

from repair.core import model
from repair.core.exceptions import RepairModelError


class HydraModel(model.RepairModel):
    """HydraNet DNN model."""

    @classmethod
    def get_name(cls) -> str:
        """Return model name."""
        return 'hydra'

    def compile(self, input_shape: tuple[int, int, int], output_shape: int):
        """Configure ResNet model.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
        output_shape : int

        Returns
        -------
        keras.Model
            A compiled keras model.

        """
        if self.tuning is None:
            return self._get_original(input_shape, output_shape)

        input_img = Input(shape=input_shape)

        if 'augmentation' in self.tuning:
            augmentation_list = []
            for augment in self.tuning['augmentation']:
                aug_instance = getattr(layers, augment[0])(**augment[1])
                augmentation_list.append(aug_instance)
            data_augmentation = keras.Sequential(augmentation_list)
            input_img = data_augmentation(input_img)

        importlib.invalidate_caches()
        instance = getattr(applications, self.tuning['model'])(
            include_top=False,
            weights='imagenet',
            input_tensor=input_img,
            input_shape=input_shape,
            pooling=None,
        )
        out = instance.output

        branch_list = []

        self._add_branch_layer(out, branch_list, output_shape)

        self._add_gate_layer(out, branch_list)

        predictions = Combine()(branch_list)

        if 'optimizer' in self.tuning:
            opt_name = self.tuning['optimizer'][0]
            opt_kwargs = self.tuning['optimizer'][1]
            opt_instance = getattr(optimizers, opt_name)(**opt_kwargs)
        else:
            opt_instance = optimizers.SGD(learning_rate=1e-4, momentum=0.9)

        model = Model(inputs=instance.input, outputs=predictions)

        model.compile(optimizer=opt_instance, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def _add_branch_layer(self, out, branch_list, output_shape):
        for idx in range(self.branch_num):
            b_out = out
            if 'layers' in self.tuning:
                for tuning in self.tuning['layers']:
                    branch_tuning = tuning[1].copy()
                    branch_tuning['name'] = f"Branch{str(idx)}{branch_tuning['name']}"
                    try:
                        b_out = getattr(layers, tuning[0])(**branch_tuning)(b_out)
                    except AttributeError as ae:
                        raise RepairModelError(f'Cannot import {tuning[0]}') from ae
                    except TypeError as te:
                        raise RepairModelError(f'Wrong inputs for {tuning[0]}') from te

            b_out = Dense(output_shape, activation='softmax', name=f'Branch{str(idx)}_Dense')(b_out)
            branch_list.append(b_out)

    def _add_gate_layer(self, g_out, branch_list):
        if 'layers' in self.tuning:
            for tuning in self.tuning['layers']:
                try:
                    g_out = getattr(layers, tuning[0])(**tuning[1])(g_out)
                except AttributeError as ae:
                    raise RepairModelError(f'Cannot import {tuning[0]}') from ae
                except TypeError as te:
                    raise RepairModelError(f'Wrong inputs for {tuning[0]}') from te
        g_out = Dense(self.branch_num, activation='softmax', name='Gate_Dense')(g_out)
        branch_list.append(g_out)

    def _get_original(self, input_shape, output_shape):
        """Get a model from the original paper.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            A shape of input dataset
        output_shape : int
            The number of classes

        Returns
        -------
        keras.Model
            A compiled keras model.

        """
        d_num = 2
        ws_val = 0.5
        wb_val = 0.125
        wg_val = 0.125
        input_img = Input(shape=input_shape)
        out = Conv2D(
            64,
            (3, 3),
            padding='same',
            data_format='channels_last',
            strides=2,
            name='Input_Conv2D',
        )(input_img)
        out = MaxPooling2D(
            pool_size=(3, 3), data_format='channels_last', strides=2, name='Input_MaxPool'
        )(out)
        for idx in range(3):
            out = self.add_block(out, d_num, ws_val, 128 * 2**idx, name=f'Block{str(128 * 2**idx)}')
        branches = []
        for idx in range(self.branch_num):
            branch = self.add_block(out, d_num, wb_val, 1024, name=f'Branch{str(idx)}')
            branch = AveragePooling2D(
                pool_size=(7, 7),
                data_format='channels_last',
                strides=1,
                name=f'Branch{str(idx)}_AvgPool',
            )(branch)
            branch = Flatten(name=f'Branch{str(idx)}_Flatten')(branch)
            branch = Dense(output_shape, activation='softmax', name=f'Branch{str(idx)}_Dense')(
                branch
            )
            branches.append(branch)

        gate = self.add_block(out, d_num, wg_val, 1024, name='Gate')
        gate = Flatten(name='Gate_Flatten')(gate)
        gate = Dense(self.branch_num, activation='softmax', name='Gate_Dense')(gate)
        branches.append(gate)

        predictions = Combine()(branches)

        model = Model(inputs=input_img, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        return model

    def add_block(self, out, d_num, w_val, channels, shortcut=True, name=None):
        """Add layers block."""
        sc = SeparableConv2D(
            channels * w_val,
            3,
            padding='same',
            data_format='channels_last',
            strides=1,
            name=f'{name}_Shortcut1',
        )(out)
        sc = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=f'{name}_Shortcut2')(sc)
        for n in range(d_num):
            b_name = f'{name}_{str(n)}'
            out = SeparableConv2D(
                channels * w_val,
                3,
                padding='same',
                data_format='channels_last',
                strides=1,
                name=f'{b_name}_SepConv2D1',
            )(out)
            out = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=f'{b_name}_BatchNorm1')(
                out
            )
            out = SeparableConv2D(
                channels * w_val,
                (3, 3),
                padding='same',
                data_format='channels_last',
                strides=1,
                name=f'{b_name}_SepConv2D2',
            )(out)
            out = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=f'{b_name}_BatchNorm2')(
                out
            )

        return Add(name=f'{name}_Add')([sc, out])

    def set_extra_config(self, **kwargs):
        """Set extra config for Combine layer."""
        if 'branch_num' in kwargs:
            self.branch_num = kwargs['branch_num']
        else:
            self.branch_num = 3

        if 'model_settings' in kwargs:
            tuning_path = Path(kwargs['model_settings'])
            with open(tuning_path) as json_file:
                self.tuning = json.load(json_file)
        else:
            self.tuning = None


class Combine(layers.Layer):
    """Combine outputs of branches based on gate output."""

    def __init__(self, **kwargs):
        """Initializing CombineLayer with the shape of the split."""
        super().__init__()

    def get_config(self):
        """Config information of this layer."""
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Concrete process of combine.

        branch_list = tf.split(input, self.split_shape, 1)
        combined = branch_list[self.inner_predict(branch_list[-1])]
        """
        gate_categories = self.inner_predict(inputs[-1], len(inputs) - 1)
        out_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for i in range(len(gate_categories)):
            out = inputs[0][i] * gate_categories[i][0]
            for j in range(1, len(inputs) - 1):
                out = out + inputs[j][i] * gate_categories[i][j]
            out_list = out_list.write(out_list.size(), out)

        final_outs = out_list.stack()

        return final_outs

    @tf.function
    def inner_predict(self, data, num):
        """Gating branch based on gating layer."""
        tf_list = tf.math.argmax(data, 1)
        # Ensure the shape is one-hot
        tf_list = tf.reshape(tf_list, (-1, 1))
        gate_out = CategoryEncoding(num_tokens=num, output_mode='one_hot')(tf_list)
        return gate_out
