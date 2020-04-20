import tensorflow as tf
import keras


class MinorPool2D(keras.layers.Layer):
    def __init__(self, kernel_size=(2, 2), strides=2, padding="VALID", **kwargs):
        """
        :param (int, int) kernel_size:
        :param int or (int, int) strides:
        :param padding:
        """
        super().__init__(**kwargs)
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding

    def call(self, inputs, **kwargs):
        max_pool = tf.nn.max_pool2d(inputs, self._kernel_size, self._strides, self._padding)
        min_pool = -tf.nn.max_pool2d(-inputs, self._kernel_size, self._strides, self._padding)
        avg_pool = tf.nn.avg_pool2d(inputs, self._kernel_size, self._strides, self._padding)

        max_diff = max_pool - avg_pool
        min_diff = min_pool - avg_pool

        diff_excess = max_diff + min_diff
        abs_excess = tf.abs(diff_excess)
        greater_diff = tf.maximum(max_diff, -min_diff)
        factor = tf.math.divide_no_nan(greater_diff, abs_excess)

        return min_pool
        return avg_pool + diff_excess * factor
