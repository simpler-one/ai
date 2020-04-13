import numpy as np
import tensorflow as tf
import keras
from keras import backend as BE
import math


class LocalDense2D(keras.layers.Layer):
    def __init__(self, channels, size, **kwargs):
        """

        :param int channels:
        :param (int, int) size: (height, width)
        """
        super().__init__(**kwargs)
        self._channels = channels
        self._size = size

        self._unit_wid = 0
        self._unit_hgt = 0
        self._weights = None

    def build(self, input_shape):
        super().build(input_shape)
        self._unit_wid = math.ceil(input_shape[2] / self._size[1])
        self._unit_hgt = math.ceil(input_shape[1] / self._size[0])

        self._weights = self.add_weight("W", (
            self._unit_hgt, self._unit_wid, input_shape[3],
            self._size[0], self._size[1], self._channels
        ))

    def call(self, inputs, **kwargs):
        in_tensor = inputs if isinstance(inputs, tf.Tensor) else inputs[0]

    def compute_output_shape(self, input_shape):
        return self._size + (self._channels,)

    def get_config(self):
        return {
            **super().get_config(),
            "channels": self._channels,
            "size": self._size,
        }
