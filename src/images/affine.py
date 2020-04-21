import numpy as np


def transform(image, affine_matrix, image_x_axis=1):
    """

    :param np.ndarray[float or int] image: [X/Y, X/Y] or [X/Y, X/Y, C]
    :param np.ndarray[float] affine_matrix:
    :param int image_x_axis: (0 or 1)
    :return:
    """
    height, width = image.shape[:2]
    x = np.tile(np.linspace(0, width, width).reshape(1, -1), (height, 1))
    y = np.tile(np.linspace(0, height, height).reshape(-1, 1), (1, width))
    xy = np.array([[x, y, np.ones(x.shape)]])

    mat = np.linalg.inv(affine_matrix)[..., None, None]
    dx, dy, _ = np.sum(xy * mat, axis=1)
    index_map = (
        np.clip(dx, 0, width - 1).astype('i'),
        np.clip(dy, 0, height - 1).astype('i')
    )
    return image[index_map[image_x_axis], index_map[1 - image_x_axis]]


def translation_matrix(delta):
    """

    :param (int, int) delta: (x, y)
    :return:
    """
    dx, dy = delta
    return np.array([
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
    return np.array([
        [np.cos(radian), -np.sin(radian), ax - ax * np.cos(radian) + ay * np.sin(radian)],
        [np.sin(radian), np.cos(radian),  ay - ax * np.sin(radian) - ay * np.cos(radian)],
        [0,              0,               1]
    ])


def shear_matrix(mx, my):
    return np.array([
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
    return np.array([
        [zx, 0,      -ax * zx + ax],
        [0,      zy, -ay * zy + ay],
        [0,      0,  1]
    ])
