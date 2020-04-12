import numpy as np
import PIL
from PIL.Image import Image
import tensorflow as tf
import keras
import keras.backend as BE
from channel_focus2d import ChannelFocus2D
from unit_focus2d import UnitFocus2D

FILE_PATHS = ("./data/1.png", "./data/2.png")
IMG_MAP = ["_", " "]


def main():
    images = []
    for path in FILE_PATHS:
        pil_image: Image = PIL.Image.open(path)
        img = np.asarray(pil_image.convert("RGB"))
        images.append(img)

        # (min_x, min_y), (max_x, max_y) = detect_rect(img)
        # print((min_x, min_y), (max_x, max_y))

    img_in = np.array(images)
    # img_tensor = BE.constant(img_in)
    # focus = UnitFocus2D()
    # focus.build(img_in.shape)
    # out_tensor = focus.call(img_tensor)
    # session = tf.Session()
    #
    # img_out = session.run(out_tensor)

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

    return (max(0, x_min - padding), max(0, y_min - padding)), \
           (min(x_scan.shape[0], x_max + padding), min(y_scan.shape[0], y_max + padding))


# ----------
main()
