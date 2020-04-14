import numpy as np
import PIL
from PIL.Image import Image
import tensorflow as tf
import keras
from layers.channel_focus2d import ChannelFocus2D

FILE_PATHS = ("./data/1.png", "./data/2.png")
IMG_MAP = ["_", " "]


def main():
    images = []
    for path in FILE_PATHS:
        pil_image: Image = PIL.Image.open(path)
        img = np.asarray(pil_image.convert("RGB"))
        images.append(img)

        (min_x, max_x), (min_y, max_y) = detect_rect(img[:, :, 0])
        print((min_x, max_x), (min_y, max_y))

        x_weights, y_weights = create_focus_weights((img.shape[1], img.shape[0]), (min_x, max_x), (min_y, max_y))

    img_in = np.array(images)
    model = keras.models.Sequential([
        ChannelFocus2D(),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["acc"])
    img_out = model.predict(img_in)

    for b in img_out:
        out_img = b[:, :, 0]

        print("--image--")
        for r in out_img:
            print("".join([IMG_MAP[v] for v in np.round(r / 255).astype("int32")]))
        print()


def detect_rect(img, padding=3):
    """

    :param np.ndarray[float] img:
    :param int padding:
    :return:
    """
    x_scan = np.max(img, axis=0) / 255.0
    y_scan = np.max(img, axis=1) / 255.0

    x_scan = np.round(x_scan)
    y_scan = np.round(y_scan)

    x_min_weight = np.arange(x_scan.shape[0] - 1, -1, -1)
    y_min_weight = np.arange(y_scan.shape[0] - 1, -1, -1)

    x_max_weight = np.arange(x_scan.shape[0])
    y_max_weight = np.arange(y_scan.shape[0])

    x_min = np.argmax(x_scan * x_min_weight)
    y_min = np.argmax(y_scan * y_min_weight)

    x_max = np.argmax(x_scan * x_max_weight)
    y_max = np.argmax(y_scan * y_max_weight)

    return (max(0, x_min - padding), min(x_scan.shape[0], x_max + padding)), \
           (max(0, y_min - padding), min(y_scan.shape[0], y_max + padding))


def create_focus_weights(org_size, x_range, y_range, padding=3):
    """

    :param (int, int) org_size:
    :param (int, int) x_range:
    :param (int, int) y_range:
    :param int padding:
    :return:
    """
    org_w, org_h = org_size
    min_x, max_x = x_range
    min_y, max_y = y_range

    trim_w = max_x - min_x
    trim_h = max_y - min_y
    zoom = min(org_w / trim_w, org_h / trim_h)

    margin_x = round((org_w / zoom - trim_w) / 2) + padding
    margin_y = round((org_h / zoom - trim_h) / 2) + padding

    x_weights = np.zeros((org_w, org_w))
    for x in range(margin_x, org_w - margin_x):
        x_weights[x, min_x + round(x * zoom)] = 1.0  # one hot

    y_weights = np.zeros((org_h, org_h))
    for y in range(margin_y, org_h - margin_y):
        y_weights[y, min_y + round(y * zoom)] = 1.0  # one hot

    return x_weights, y_weights


# ----------
main()
