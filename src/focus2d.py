import numpy as np
import tensorflow as tf
import keras
from keras import backend as BE

INT32 = "int32"


class Focus2D(keras.layers.Layer):
    def __init__(self, *, threshold=0.5, padding=3, ignore_size=(5, 5), **kwargs):
        """
        :param float threshold: (0.0-1.0)
        :param int padding: (0-)
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
        self._x_min_weight = BE.constant(np.arange(self._wid - 1, -1, -1)[None, :, None], dtype=INT32)
        self._y_min_weight = BE.constant(np.arange(self._hgt - 1, -1, -1)[None, :, None], dtype=INT32)
        self._x_max_weight = BE.constant(np.arange(self._wid)[None, :, None], dtype=INT32)
        self._y_max_weight = BE.constant(np.arange(self._hgt)[None, :, None], dtype=INT32)

    def call(self, inputs, **kwargs):
        in_tensor = inputs if isinstance(inputs, tf.Tensor) else inputs[0]
        (x_min, y_min), (x_max, y_max) = self._detect_rect(in_tensor)

        # TODO: avoid all zero
        normal_zoom_wid = self._wid - self._pad * 2
        normal_zoom_hgt = self._hgt - self._pad * 2

        batch_out = [None] * in_tensor.shape[0]
        for b in range(in_tensor.shape[0]):
            channel_out = [None] * in_tensor.shape[3]
            for c in range(in_tensor.shape[3]):
                trimmed = in_tensor[b:b+1, y_min[b, c]:y_max[b, c], x_min[b, c]:x_max[b, c], c:c+1]
                zoom_wid = max(normal_zoom_wid, trimmed.shape[2])
                zoom_hgt = max(normal_zoom_hgt, trimmed.shape[1])

                resized = tf.image.resize_images(trimmed, (zoom_hgt, zoom_wid), preserve_aspect_ratio=True)
                resized = tf.image.resize_with_crop_or_pad(resized, self._hgt, self._wid)
                channel_out[c] = resized

            batch_out[b] = tf.concat(channel_out, axis=3)

        return tf.concat(batch_out, axis=0)

    def get_config(self):
        return {
            "threshold": self._shift + 0.5,
            "padding": self._pad,
            "ignore_size": self._ignore_size
        }

    def compute_output_shape(self, input_shape):
        return input_shape

    def _detect_rect(self, tensor):
        # (Batch, Y, X, Channel)
        x_scan = tf.reduce_max(tensor, axis=1)
        y_scan = tf.reduce_max(tensor, axis=2)
        x_scan = tf.cast(tf.math.round(tf.math.sigmoid(x_scan) + self._shift), INT32)  # 0 or 1
        y_scan = tf.cast(tf.math.round(tf.math.sigmoid(y_scan) + self._shift), INT32)  # 0 or 1

        x_min = tf.argmax(x_scan * self._x_min_weight, axis=1, output_type=INT32)
        y_min = tf.argmax(y_scan * self._y_min_weight, axis=1, output_type=INT32)
        x_max = tf.argmax(x_scan * self._x_max_weight, axis=1, output_type=INT32)
        y_max = tf.argmax(y_scan * self._y_max_weight, axis=1, output_type=INT32)

        return (x_min, y_min), (x_max, y_max)
