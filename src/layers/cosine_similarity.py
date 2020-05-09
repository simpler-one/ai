import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend


class StaticCosineSimilarity(keras.layers.Layer):
    def __init__(self, centroids, **kwargs):
        super().__init__(**kwargs)
        self._org_centroids = centroids
        self._centroids = backend.constant(centroids / np.linalg.norm(centroids, axis=0))

    def call(self, inputs, *kwargs):
        feat_vector = tf.nn.l2_normalize(inputs, axis=-1)
        return feat_vector @ self._centroids

    def compute_output_shape(self, input_shape):
        return None, self._centroids.shape[-1]

    def get_config(self):
        return {
            **super().get_config(),
            "centroids": np.array(self._org_centroids).tolist(),
        }
