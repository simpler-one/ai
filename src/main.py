import numpy as np
import PIL
from PIL.Image import Image
import tensorflow as tf
from tensorflow import keras
from images import transform, rotation_matrix, translation_matrix, zoom_matrix, load_image
from images.irregular_symbol_generator import IrregularSymbolGenerator
from layers.focus_pooling2d import FocusPooling2D
from layers.batch_focus2d import BatchFocus2D
from etl8g import read_files_etl8g
import math

FILE_PATHS = ("./data/1.png", "./data/2.png", "./data/4.png", "./data/4-2.png")
IMG_MAP = ["_", " "]


def main():
    session = tf.Session()

    # irr_data_list, irr_target_list = read_files_etl8g(["./data/ETL8G/ETL8G_01"], (28, 28))
    # show_img(irr_data_list[0])

    images = []
    for path in FILE_PATHS:
        images.append(load_image(path))

    img_in = np.array(images, dtype=np.float32)[:, :, :, None]

    irr_gen = IrregularSymbolGenerator(max_color_value=255)
    for _ in range(round(3)):
        irr_data, irr_target = irr_gen.generate_from(img_in, np.zeros((img_in.shape[0], 1)))
        show_img(irr_data[..., 0])

    return

    model = keras.models.Sequential([
        BatchFocus2D(),
    ])
    img_out = model.predict(img_in)

    # focus = BatchFocus2D()
    # focus.build(img_in.shape)
    # img_out = focus.call(img_in)
    # img_out = session.run(img_out)

    for im in img_out:
        show_img(im[..., 0].astype("i"))


def show_img(img):
    img = PIL.Image.fromarray(np.uint8(img))
    img.show()


def save_img(img, dir_name, name):
    import PIL.Image
    img = PIL.Image.fromarray(np.uint8(img))
    img.save(f"./output/{dir_name}/{name}.png")


# ----------
main()
