import os
import re
import numpy as np
import PIL.Image
from PIL.Image import Image


def load_image_dir(dir_path):
    return [load_image(p) for p in list_pictures(dir_path)]


def load_image(file_path):
    pil_image: Image = PIL.Image.open(file_path)
    return np.asarray(pil_image.convert("L"))[..., None]


def list_pictures(dir_path, ext="jpg|jpeg|bmp|png|ppm"):
    return [os.path.join(root, f)
            for root, _, files in os.walk(dir_path) for f in files
            if re.match(r"([\w]+\.(?:" + ext + "))", f.lower())]
