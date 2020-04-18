import numpy as np


def transform(image, affine_matrix):
    length = np.max(image.shape)
    half_len = length / 2
    x = np.tile(np.linspace(-half_len, half_len, length).reshape(1, -1), (length, 1))
    y = np.tile(np.linspace(-half_len, half_len, length).reshape(-1, 1), (1, length))
    xy = np.array([[x, y, np.ones(x.shape)]])

    dx, dy, _ = np.sum(xy * affine_matrix[..., None, None], axis=1)
    x_index = np.clip(dx + half_len, 0, length - 1).astype('i')
    y_index = np.clip(dy + half_len, 0, length - 1).astype('i')
    return image[y_index, x_index]


def translation_matrix(delta_x, delta_y):
    """

    :param int delta_x:
    :param int delta_y:
    :return:
    """
    return np.array([
        [1, 0, -delta_x],
        [0, 1, -delta_y],
        [0, 0, 1]
    ])


def rotation_matrix(radian):
    return np.array([
        [np.cos(radian), -np.sin(radian), 0],
        [np.sin(radian), np.cos(radian),  0],
        [0,              0,               1]
    ])


def shear_matrix(mx, my):
    return np.array([
        [1,   -mx, 0],
        [-my, 1,   0],
        [0,   0,   1]
    ])


def zoom_matrix(zoom_x, zoom_y):
    """

    :param float zoom_x:
    :param float zoom_y:
    :return:
    """
    return np.array([
        [1 / zoom_x, 0,          0],
        [0,          1 / zoom_y, 0],
        [0,          0,          1]
    ])
