import tensorflow as tf
import keras
import keras.backend as BE


class ArcFace(keras.layers.Layer):
    def __init__(self, categories, *, margin_penalty=0.50, softmax_factor=30.0, **kwargs):
        """
        :param int categories:
        :param float softmax_factor:
        :param float margin_penalty:
        """
        super().__init__(**kwargs)
        self._categories = categories
        self._margin_penalty = margin_penalty
        self._softmax_factor = softmax_factor

    def call(self, inputs, *kwargs):
        feat_vector, targets = inputs

        feat_vector = tf.nn.l2_normalize(feat_vector, axis=-1)
        centroid = tf.reduce_mean(feat_vector[:, :, None] * targets[:, None, :], axis=0)

        cos_sim = feat_vector @ centroid

        # add margin
        # clip cos-sim to prevent zero division when backward
        theta = tf.acos(BE.clip(cos_sim, -1.0 + BE.epsilon(), 1.0 - BE.epsilon()))
        target_cos_sim = tf.cos(theta + self._margin_penalty)

        cos_sim = cos_sim * (1 - targets) + target_cos_sim * targets
        return tf.nn.softmax(cos_sim * self._softmax_factor)

    def compute_output_shape(self, input_shape):
        return None, self._categories

    def get_config(self):
        return {
            **super().get_config(),
            "categories": self._categories,
            "margin_penalty": self._margin_penalty,
            "softmax_factor": self._softmax_factor,
        }
