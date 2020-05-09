import numpy as np
import os
import PIL.Image
from tensorflow.keras.datasets import mnist

EXPORT_DIR = "./output/mnist/"
EXPORT_SIZE = 200


def main():
    (train_data, train_target), _ = mnist.load_data()
    for i, (d, t) in enumerate(zip(train_data[:EXPORT_SIZE], train_target[:EXPORT_SIZE])):
        save_img(d, t, i)


def save_img(img, category, name):
    dir_path = f"{EXPORT_DIR}/{category}"
    os.makedirs(dir_path, exist_ok=True)

    img = PIL.Image.fromarray(np.uint8(img))
    img.save(f"{dir_path}/{name}.png")


# -----
main()
