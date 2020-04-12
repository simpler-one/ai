import numpy as np
import PIL
from PIL.Image import Image
import tensorflow as tf
import keras
import keras.backend as BE
from focus2d import Focus2D

FILE_PATH = "./data/1.png"


def main():
    pil_image: Image = PIL.Image.open(FILE_PATH)
    img = np.asarray(pil_image)[:, :, 0]
    img = img.reshape(img.shape[:2])

    (min_x, min_y), (max_x, max_y) = detect_rect(img)
    print((min_x, min_y), (max_x, max_y))

    img_tensor = BE.constant(img[np.newaxis, :, :, np.newaxis])
    focus = Focus2D()
    focus.build((-1,) + img.shape + (1,))
    out_tensor = focus.call(img_tensor)
    session = tf.Session()
    out_img = session.run(out_tensor)[0, :, :, 0]
    for r in out_img:
        print(np.round(r / 255).astype("int32").tolist())

    # model = keras.models.Sequential([
    #     Focus2D()
    # ])
    # model.compile(optimizer="adam", loss="mse", metrics=["acc"])
    # model.predict(img[None, :, :, None])


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
