import numpy as np
import keras
from layers.unit_focus2d import UnitFocus2D

W = 3
H = 3

# slash="-" * 0.7 + "|" * 0.7
CATEGORIES = (
    "o", "^", "-", "|", "x"
)


def create():
    data = ["./data/symbols/0.png", "./data/symbols/1.png", "./data/symbols/2.png"]
    target = np.array([
        [
            *[
                # o
                *[0, 1, 0] +
                [1, 0, 1] +
                [0, 1, 0]
            ] + [
                # ^
                *[0, 0, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ] + [
                # -
                *[0, 0, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ] + [
                # |
                *[0, 0, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ] + [
                # x
                *[0, 0, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ]
        ],
        [
            *[
                # o
                *[0, 0, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ] + [
                # ^
                *[0, 1, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ] + [
                # -
                *[0, 0, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ] + [
                # |
                *[0, 0, 0] +
                [0, 1, 0] +
                [0, 1, 0]
            ] + [
                # x
                *[0, 0, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ]
        ],
        [
            *
            [
                *  # o
                [0, 1, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ] + [
                 *  # ^
                 [0, 1, 0] +
                 [0, 0, 0] +
                 [1, 0, 0]
            ] + [
                *  # -
                [0, 0, 0] +
                [0, 0.7, 0] +
                [0, 1, 0]
            ] + [
                *  # |
                [0, 0, 0] +
                [0, 1, 0] +
                [0, 1, 0]
            ] + [
                *  # x
                [0, 0, 0] +
                [0, 0, 0] +
                [0, 0, 0]
            ]
        ],
    ])

    model = keras.models.Sequential([
        keras.layers.Conv2D(8, (3, 3)),
        keras.layers.MaxPooling2D(),
        UnitFocus2D(),
        keras.layers.Conv2D(16, (3, 3)),
        keras.layers.MaxPooling2D(),
        UnitFocus2D(),
        keras.layers.Dense(32),
        keras.layers.Dense(W * H * len(CATEGORIES)),
    ])
