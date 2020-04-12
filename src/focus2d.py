import numpy as np
import tensorflow as tf
import keras
from keras import backend as BE

INT32 = "int32"


class Focus2D(keras.layers.Layer):
    def __init__(self, *, threshold=0.5, padding=3, **kwargs):
        """
        :param float threshold: (0.0-1.0)
        :param int padding: (0-)
        """
        super().__init__(**kwargs)
        self._shift = threshold - 0.5
        self._pad = padding

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
        self._x_min_weight = BE.constant(np.arange(self._wid - 1, -1, -1)[None, :, None], dtype=INT32)
        self._y_min_weight = BE.constant(np.arange(self._hgt - 1, -1, -1)[None, :, None], dtype=INT32)
        self._x_max_weight = BE.constant(np.arange(self._wid)[None, :, None], dtype=INT32)
        self._y_max_weight = BE.constant(np.arange(self._hgt)[None, :, None], dtype=INT32)

    def call(self, inputs, **kwargs):
        in_tensor = inputs if isinstance(inputs, tf.Tensor) else inputs[0]
        (x_min, y_min), (x_max, y_max) = self._detect_rect(in_tensor)
        # TODO:

        return in_tensor

    def get_config(self):
        return {
            "threshold": self._shift + 0.5,
            "padding": self._pad,
        }

    def compute_output_shape(self, input_shape):
        return input_shape

    def _detect_rect(self, tensor):
        # (n, Y, X, C)
        x_scan = tf.reduce_max(tensor, axis=1)
        y_scan = tf.reduce_max(tensor, axis=2)
        x_scan = tf.cast(tf.math.round(tf.math.sigmoid(x_scan) + self._shift), INT32)  # 0 or 1
        y_scan = tf.cast(tf.math.round(tf.math.sigmoid(y_scan) + self._shift), INT32)  # 0 or 1

        x_min = tf.argmax(x_scan * self._x_min_weight, axis=1, output_type=INT32)
        y_min = tf.argmax(y_scan * self._y_min_weight, axis=1, output_type=INT32)
        x_max = tf.argmax(x_scan * self._x_max_weight, axis=1, output_type=INT32)
        y_max = tf.argmax(y_scan * self._y_max_weight, axis=1, output_type=INT32)

        return (
            (tf.maximum(0, x_min - self._pad), tf.maximum(0, y_min - self._pad)),
            (tf.minimum(x_max + self._pad, self._wid - 1), tf.minimum(y_max + self._pad, self._hgt - 1))
        )
