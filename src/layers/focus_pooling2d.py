import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as BE


class FocusPooling2D(keras.layers.Layer):
    def __init__(self, kernel_size=(2, 2), conserved_ratio=0.3, **kwargs):
        """
        :param (int, int) kernel_size:
        :param float conserved_ratio: (0.0-1.0)
        """
        super().__init__(**kwargs)
        self._kernel_size = kernel_size
        self._strides = kernel_size
        self._conserved_ratio = conserved_ratio

        self._input_size = (-1, -1)

    def build(self, input_shape):
        self._input_size = input_shape[1:3]

    def call(self, inputs, **kwargs):
        keep_w = round(self._input_size[1] * self._conserved_ratio)
        keep_h = round(self._input_size[0] * self._conserved_ratio)
        kh, kw = self._kernel_size
        sh, sw = self._strides

        def for_batch(tensor_yxc):
            def for_channel(tensor_yx):
                x_pool = tf.nn.max_pool2d(tensor_yx[None, :, :, None], (1, kw), (1, sw), "VALID")
                x_scr = tf.math.reduce_std(tensor_yx, axis=1)
                x_scr = tf.nn.max_pool1d(x_scr[None, :, None], kw, sw, "VALID")
                x_scr = tf.argsort(x_scr[0, :, 0], direction="DESCENDING")

                def for_x(x):
                    x0 = x * sw
                    return tf.cond(x_scr[x] < keep_w, lambda: tensor_yx[:, x0:x0 + kw], lambda: x_pool[:, x:x + 1])

                tensor_yx = tf.stack([for_x(x) for x in range(x_scr.shape[0])])

                y_pool = tf.nn.max_pool2d(tensor_yx[None, :, :, None], (kh, 1), (sh, 1), "VALID")
                y_scr = tf.math.reduce_std(tensor_yx[None, :, None], axis=2)
                y_scr = tf.nn.max_pool1d(y_scr, kw, sw, "VALID")
                y_scr = tf.argsort(y_scr[0, :, 0], direction="DESCENDING")

                def for_y(y):
                    y0 = y * sh
                    return tf.cond(y_scr[y] < keep_h, lambda: tensor_yx[y0:y0 + kh, :], lambda: y_pool[y:y + 1, :])

                return tf.stack([for_y(y) for y in range(y_scr.shape[0])])
            return tf.stack([for_channel(tensor_yxc[..., c]) for c in range(tensor_yxc.shape[-1])], axis=-1)
        return tf.map_fn(for_batch, inputs)

    def compute_output_shape(self, input_shape):
        conserved_w = round(input_shape[2] * self._conserved_ratio)
        conserved_h = round(input_shape[1] * self._conserved_ratio)
        w = input_shape[1] * (1 - self._conserved_ratio) // self._strides[1] + conserved_w
        h = input_shape[0] * (1 - self._conserved_ratio) // self._strides[0] + conserved_h
        return None, h, w, input_shape[-1]

    def get_config(self):
        return {
            **super().get_config(),
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "conserved_ratio": self._conserved_ratio,
        }

