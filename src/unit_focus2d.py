import tensorflow as tf
from base_focus2d import BaseFocus2D

INT32 = "int32"


class UnitFocus2D(BaseFocus2D):
    def __init__(self, *, threshold=0.5, padding=3, ignore_size=(5, 5), **kwargs):
        """
        :param float threshold: (0.0-1.0)
        :param int padding: (0-)
        :param (int, int) ignore_size:
        """
        super().__init__(threshold=threshold, padding=padding, ignore_size=ignore_size, **kwargs)

    def call(self, inputs, **kwargs):
        in_tensor = inputs if isinstance(inputs, tf.Tensor) else inputs[0]

        normal_zoom_wid = self._wid - self._pad * 2
        normal_zoom_hgt = self._hgt - self._pad * 2

        def unit_loop(unit_tensor):
            # TODO: avoid zero size
            (x_min, y_min), (x_max, y_max) = self._detect_rect(unit_tensor)

            trimmed = unit_tensor[y_min:y_max, x_min:x_max, :]
            zoom_wid = max(normal_zoom_wid, trimmed.shape[2])
            zoom_hgt = max(normal_zoom_hgt, trimmed.shape[1])

            resized = tf.image.resize_images(trimmed, (zoom_hgt, zoom_wid), preserve_aspect_ratio=True)
            resized = tf.image.resize_with_crop_or_pad(resized, self._hgt, self._wid)
            return resized

        return tf.map_fn(unit_loop, in_tensor)

    def _detect_rect(self, unit_tensor):
        # (Y, X, Channel)
        x_scan = tf.reduce_max(tf.reduce_max(unit_tensor, axis=0), axis=-1)
        y_scan = tf.reduce_max(tf.reduce_max(unit_tensor, axis=1), axis=-1)
        x_scan = tf.cast(tf.math.round(tf.math.sigmoid(x_scan) + self._shift), INT32)  # 0 or 1
        y_scan = tf.cast(tf.math.round(tf.math.sigmoid(y_scan) + self._shift), INT32)  # 0 or 1

        x_min = tf.argmax(x_scan * self._x_min_weight, axis=0, output_type=INT32)
        y_min = tf.argmax(y_scan * self._y_min_weight, axis=0, output_type=INT32)
        x_max = tf.argmax(x_scan * self._x_max_weight, axis=0, output_type=INT32)
        y_max = tf.argmax(y_scan * self._y_max_weight, axis=0, output_type=INT32)

        return (x_min, y_min), (x_max, y_max)

    def _reshape_weight(self, weight):
        return weight
