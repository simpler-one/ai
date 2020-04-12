import numpy as np
import keras
from keras import backend as BE
from abc import ABC

INT32 = "int32"


class BaseFocus2D(keras.layers.Layer, ABC):
    def __init__(self, *, threshold, padding, ignore_size, **kwargs):
        """
        :param float threshold: (0.0-1.0)
        :param int padding: (0-)
        :param (int, int) ignore_size:
        """
        super().__init__(**kwargs)
        self._shift = threshold - 0.5
        self._pad = padding
        self._ignore_size = ignore_size

        self._wid = 0
        self._hgt = 0
        self._x_min_weight = None
        self._y_min_weight = None
        self._x_max_weight = None
        self._y_max_weight = None

    def build(self, input_shape):
        super().build(input_shape)
        # self._shift = self.add_weight("shift", shape=(1,), dtype="float32")
        self._wid = np.int32(input_shape[2])
        self._hgt = np.int32(input_shape[1])
        self._x_min_weight = BE.constant(self._reshape_weight(np.arange(self._wid - 1, -1, -1)), dtype=INT32)
        self._y_min_weight = BE.constant(self._reshape_weight(np.arange(self._hgt - 1, -1, -1)), dtype=INT32)
        self._x_max_weight = BE.constant(self._reshape_weight(np.arange(self._wid)), dtype=INT32)
        self._y_max_weight = BE.constant(self._reshape_weight(np.arange(self._hgt)), dtype=INT32)

    def get_config(self):
        return {
            "threshold": self._shift + 0.5,
            "padding": self._pad,
            "ignore_size": self._ignore_size
        }

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_weight(self, weight):
        """

        :param np.ndarray[float] weight:
        :rtype: np.ndarray[float]
        :return:
        """
        raise NotImplementedError()
