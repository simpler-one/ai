import tensorflow as tf
import keras
from keras import backend as BE


class FocusLayer(keras.layers.Layer):
    def __init__(self, *, threshold=0.5, padding=3):
        self._shift = threshold - 0.5
        self._padding = padding

    def call(self, inputs, **kwargs):
        in_tensor = inputs if isinstance(inputs, tf.Tensor) else inputs[0]

    def get_config(self):
        return {
            "threshold": self._shift + 0.5,
            "padding": self._padding,
        }

    def _detect_rect(self, tensor):
        pass
