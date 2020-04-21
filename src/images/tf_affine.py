import numpy as np
import tensorflow as tf


def base_transform_map(size, dtype="float32"):
    """

    :param (int, int) size: (height, width)
    :param Any dtype:
    :return:
    """
    height, width = size
    x = np.tile(np.linspace(0, width, width).reshape(1, -1), (height, 1))
    y = np.tile(np.linspace(0, height, height).reshape(-1, 1), (1, width))
    return tf.constant([[x, y, np.ones(x.shape)]], dtype)


def transform(image, base_map, affine_matrix):
    """

    :param tf.Tensor[float] image: [X/Y, X/Y] or [X/Y, X/Y, C]
    :param tf.Tensor[float] base_map:
    :param tf.Tensor[float] affine_matrix:
    :return:
    """
    height, width = image.shape[:2]
    mat = np.linalg.inv(affine_matrix)[..., None, None]
    dx, dy, _ = np.sum(base_map * mat, axis=1)
    x_index = tf.clip_by_value(dx, 0, width - 1).astype('i')
    y_index = tf.clip_by_value(dy, 0, height - 1).astype('i')
    return image[y_index, x_index]


def translation_matrix(delta):
    """

    :param (int, int) delta: (x, y)
    :return:
    """
    dx, dy = delta
    return tf.constant([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])


def rotation_matrix(radian, anchor=(0, 0)):
    """

    :param float radian:
    :param (int, int) anchor: (x, y)
    :return:
    """
    ax, ay = anchor
    sin = tf.math.sin(radian)
    cos = tf.math.cos(radian)
    return tf.constant([
        [cos, -sin, ax - ax * cos + ay * sin],
        [sin, cos,  ay - ax * sin - ay * cos],
        [0,   0,    1]
    ])


def shear_matrix(mx, my):
    return tf.constant([
        [1,   -mx, 0],
        [-my, 1,   0],
        [0,   0,   1]
    ])


def zoom_matrix(zoom, anchor=(0, 0)):
    """

    :param (float, float) zoom: (x, y)
    :param (int, int) anchor: (x, y)
    :return:
    """
    zx, zy = zoom
    ax, ay = anchor
    return tf.constant([
        [zx, 0,      -ax * zx + ax],
        [0,      zy, -ay * zy + ay],
        [0,      0,  1]
    ])
