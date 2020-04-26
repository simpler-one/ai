import numpy as np
import random
import math
from typing import Iterable
from .affine import transform, rotation_matrix, translation_matrix, zoom_matrix, shear_matrix


class IrregularSymbolGenerator:
    def __init__(
        self, *,
        n_mix_range=(3, 6),
        max_color_value=1.0,
        transparent_color_range=0.1,
        x_shift_range=(0.1, 0.5), y_shift_range=(0.1, 0.5),
        rotate_range=(45, 315),
        target_value=0.2, noise_patterns=(),
    ):
        """
        :param (int, int) n_mix_range:
        :param float max_color_value: eg: 255, 1.0
        :param float transparent_color_range:
        :param (int, int) x_shift_range: (0 - width, 0 - width)
        :param (int, int) y_shift_range: (0 - height, 0 - height)
        :param (int, int) rotate_range: (0 - 360, 0 - 360)
        :param float target_value: (0 - 1.0)
        :param noise_patterns:
        """
        self._n_mix_range = n_mix_range
        self._max_color_value = max_color_value
        self.transparent_color_range = transparent_color_range
        self._x_shift_range = x_shift_range
        self._y_shift_range = y_shift_range
        self._rotate_range = rotate_range
        self._target_value = target_value
        self._noises = noise_patterns

    def generate_from(self, images, labels):
        """

        :param np.ndarray images: fg > bg >=0
        :param np.ndarray labels: one hot
        :return:
        """
        n_mix = random.randrange(self._n_mix_range[0], self._n_mix_range[1])
        indices = [random.randrange(images.shape[0]) for _ in range(n_mix)]

        return self.generate_one_from_all(
            (images[i] for i in indices),
            (labels[i] for i in indices),
            images.shape[1:3],
            labels.shape[-1]
        )

    def generate_one_from_all(self, images, labels, size, n_categories):
        """

        :param Iterable[np.ndarray] images: fg > bg >=0
        :param Iterable[np.ndarray] labels: one hot
        :param (int, int) size: (height, width)
        :param int n_categories:
        :return:
        """
        height, width = size
        out_img = np.zeros((height, width, 1))
        out_label = np.zeros((n_categories,))

        center = (size[1] // 2, size[0] // 2)

        x_shift_range = np.round(np.array(self._x_shift_range) * width).tolist()
        y_shift_range = np.round(np.array(self._y_shift_range) * width).tolist()

        for img, l in zip(images, labels):
            x_shift = random.randrange(*x_shift_range) * random.choice((-1, 1))
            y_shift = random.randrange(*y_shift_range) * random.choice((-1, 1))
            rotation = random.randrange(*self._rotate_range)
            matrix = rotation_matrix(rotation / 180 * math.pi, center) @ translation_matrix((x_shift, y_shift))
            img[img < self.transparent_color_range] = 0
            out_img += transform(img, matrix)
            out_label[np.argmax(l)] = self._target_value

        return np.clip(out_img, 0, self._max_color_value), out_label
