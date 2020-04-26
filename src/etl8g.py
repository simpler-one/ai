import numpy as np
import struct
from typing import NamedTuple
import PIL.Image

DATA_SET_LENGTH = 5
CATEGORY_VARIATION = 956
RECORD_SIZE = 8199

GRAY_LEVEL = 16
BASIC_RESCALE = 256 // 16

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 127

KANA_VARIATION = 72


def read_files_etl8g(file_path_list, output_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    data = []
    target = []

    for path in file_path_list:
        cur_data, cur_target = read_file_etl8g(path, output_size)
        data.extend(cur_data)
        target.extend(cur_target)

    return data, target


def read_file_etl8g(file_path, output_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    data = []
    target = []

    with open(file_path, "rb") as file:
        for i in range(DATA_SET_LENGTH):
            cur_data, cur_target = read_category_etl8g(file, output_size)
            data.extend(cur_data)
            target.extend(target)

    return data, target


def read_category_etl8g(file_stream, output_size=(IMAGE_WIDTH, IMAGE_HEIGHT), rescale=4.0):
    data = []

    for i in range(CATEGORY_VARIATION):
        cur_data, label = read_record_etl8g(file_stream, output_size)
        if b".HIRA" in label:
            data.append(np.clip(np.array(cur_data) * BASIC_RESCALE * rescale, 0, 255))

    return data, list(range(KANA_VARIATION))


def read_record_etl8g(file_stream, output_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    data = file_stream.read(RECORD_SIZE)
    record = Record(*struct.unpack('>2H8sI4B4H2B30x8128s11x', data))
    img = PIL.Image.frombytes('F', (IMAGE_WIDTH, IMAGE_HEIGHT), record.data, 'bit', 4)
    img = img.convert('L')
    img = img.resize(output_size, PIL.Image.BILINEAR)
    return img, record.label


class Record(NamedTuple):
    unknown0: int
    unknown1: int
    label: bytes
    unknown3: int
    unknown4: int
    unknown5: int
    unknown6: int
    unknown7: int
    unknown8: int
    unknown9: int
    unknown10: int
    unknown11: int
    unknown12: int
    unknown13: int
    data: bytes
