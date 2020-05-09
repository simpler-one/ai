import numpy as np
from tensorflow import keras
from tensorflow.keras import backend
from abc import ABC

INT = "int32"

_SOFTMAX_INTERCEPT = 0.5
_TO_ROUND = 0.5
_EXCLUDE_EQUAL = 1e-7


class BaseFocus2D(keras.layers.Layer, ABC):
    def __init__(self, *, threshold, padding, ignore_small, ignore_large, **kwargs):
        """
        :param float threshold: (0.0-1.0)
        :param int padding: (0-)
        :param (float, float) ignore_small:
        :param (float, float) ignore_large:
        """
        super().__init__(**kwargs)
        self._shift = threshold - _SOFTMAX_INTERCEPT + _TO_ROUND - _EXCLUDE_EQUAL
        self._pad = padding
        self._ignore_small = ignore_small
        self._ignore_large = ignore_large

        self._wid = 0
        self._hgt = 0
        self._min_size = (0, 0)
        self._max_size = (0, 0)
        self._x_min_weight = None
        self._y_min_weight = None
        self._x_max_weight = None
        self._y_max_weight = None

    def build(self, input_shape):
        super().build(input_shape)
        # self._shift = self.add_weight("shift", shape=(1,), dtype="float32")
        self._wid = np.int32(input_shape[2])
        self._hgt = np.int32(input_shape[1])
        self._min_size = (self._ignore_small[0] * self._hgt, self._ignore_small[1] * self._wid)
        self._max_size = (self._ignore_large[0] * self._hgt, self._ignore_large[1] * self._wid)
        self._x_min_weight = backend.constant(self._reshape_weight(np.arange(self._wid - 1, -1, -1)), dtype=INT)
        self._y_min_weight = backend.constant(self._reshape_weight(np.arange(self._hgt - 1, -1, -1)), dtype=INT)
        self._x_max_weight = backend.constant(self._reshape_weight(np.arange(self._wid)), dtype=INT)
        self._y_max_weight = backend.constant(self._reshape_weight(np.arange(self._hgt)), dtype=INT)

    def get_config(self):
        return {
            "threshold": self._shift + 0.5,
            "padding": self._pad,
            "ignore_small": self._ignore_small,
            "ignore_large": self._ignore_large,
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
