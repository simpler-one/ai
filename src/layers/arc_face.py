import tensorflow as tf
import keras
import keras.backend as BE
from abc import ABC, abstractmethod


class _ArcFaceBase(keras.layers.Layer, ABC):
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
        centroid = self._get_centroid(feat_vector, targets)

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

    @abstractmethod
    def _get_centroid(self, feat_vector, targets):
        return tf.reduce_sum(feat_vector[:, :, None] * targets[:, None, :], axis=0) / tf.reduce_sum(targets, axis=0)


class ArcFace(_ArcFaceBase):
    def __init__(self, categories, *, margin_penalty=0.50, softmax_factor=30.0, **kwargs):
        super().__init__(categories, margin_penalty=margin_penalty, softmax_factor=softmax_factor, **kwargs)
        self._centroid = tf.zeros((0,))

    def build(self, input_shape):
        self._centroid = self.add_weight(
            "centroid", (input_shape[0][-1], self._categories), initializer='glorot_uniform'
        )

    def _get_centroid(self, feat_vector, targets):
        return tf.nn.l2_normalize(self._centroid, axis=0)


class CentroidArcFace(_ArcFaceBase):
    def _get_centroid(self, feat_vector, targets):
        return tf.reduce_sum(feat_vector[:, :, None] * targets[:, None, :], axis=0) / tf.reduce_sum(targets, axis=0)
