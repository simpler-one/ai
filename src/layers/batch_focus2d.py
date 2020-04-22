import tensorflow as tf
import keras
from layers.base_focus2d import BaseFocus2D, INT
from images.tf_affine import transform, base_transform_map, translation_matrix, zoom_matrix

FLOAT = "float32"


class BatchFocus2D(BaseFocus2D):
    def __init__(self, *, threshold=0.5, padding=3, ignore_small=(0.1, 0.1), ignore_large=(0.8, 0.8), **kwargs):
        """
        :param float threshold: (0.0-1.0) sigmoid threshold
        :param int padding: (0-)
        :param (float, float) ignore_small:
        :param (float, float) ignore_large:
        """
        super().__init__(
            threshold=threshold, padding=padding,
            ignore_small=ignore_small, ignore_large=ignore_large,
            **kwargs
        )
        self._base_affine_map = None

    def build(self, input_shape):
        super().build(input_shape)
        self._base_affine_map = base_transform_map(input_shape[1:3])

    def call(self, inputs, **kwargs):
        in_tensor = inputs[0] if isinstance(inputs, list) else inputs

        zoom_w = self._wid - self._pad * 2
        zoom_h = self._hgt - self._pad * 2

        def batch_loop(batch_tensor):
            (x_min, y_min), (x_max, y_max) = self._detect_rect(batch_tensor)
            trim_w = tf.cast(x_max - x_min, FLOAT)
            trim_h = tf.cast(y_max - y_min, FLOAT)

            def focus():
                zoom = tf.minimum(zoom_w / trim_w, zoom_h / trim_h)
                shift_x = tf.cast(x_min, FLOAT) - (zoom_w / zoom - trim_w) / 2
                shift_y = tf.cast(y_min, FLOAT) - (zoom_h / zoom - trim_h) / 2
                affine = zoom_matrix((zoom, zoom)) @ translation_matrix((-shift_x, -shift_y))
                return transform(batch_tensor, self._base_affine_map, affine)

            return tf.cond(
                (
                    tf.math.logical_or(
                        tf.math.logical_and(trim_w <= self._min_size[1], trim_h <= self._min_size[0]),  # Too small to focus
                        tf.math.logical_or(self._max_size[1] < trim_w, self._max_size[0] < trim_h)
                    )
                ),
                lambda: batch_tensor,  # Skip
                focus
            )

        return tf.map_fn(batch_loop, in_tensor, parallel_iterations=10)

    def _detect_rect(self, batch_tensor):
        # (*, Y, X, Channel)
        x_scan = tf.reduce_max(tf.reduce_max(batch_tensor, axis=0), axis=-1)
        y_scan = tf.reduce_max(tf.reduce_max(batch_tensor, axis=1), axis=-1)
        x_scan = tf.cast(tf.math.sigmoid(x_scan) + self._shift, INT)  # 0 or 1
        y_scan = tf.cast(tf.math.sigmoid(y_scan) + self._shift, INT)  # 0 or 1

        x_min = tf.argmax(x_scan * self._x_min_weight, axis=0, output_type=INT)
        y_min = tf.argmax(y_scan * self._y_min_weight, axis=0, output_type=INT)
        x_max = tf.reduce_max(x_scan * self._x_max_weight, axis=0)
        y_max = tf.reduce_max(y_scan * self._y_max_weight, axis=0)

        return (x_min, y_min), (x_max, y_max)

    def _reshape_weight(self, weight):
        return weight
