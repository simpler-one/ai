import numpy as np
import tensorflow as tf
from tensorflow import keras
from images import load_image_dir

TRAIN_DATA_DIR = "./data/termination/data"
TRAIN_TARGET_DIR = "./data/termination/target"
SAVE_MODEL_PATH = "./output/t-detector.h5"


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[d], True)

    preprocessor = create_preprocess_model()

    train_data_list = load_image_dir(TRAIN_DATA_DIR)
    train_target_list = load_image_dir(TRAIN_TARGET_DIR)
    train_data = np.array(train_data_list, np.float32) / 255
    train_target = np.array(train_target_list, np.float32) / 255

    train_target = preprocessor.predict(train_target, verbose=0, batch_size=128)
    # show_img(train_target[0, ..., 0] * 255)

    model = create_model()
    model.fit(
        train_data, train_target,
        epochs=50,
        batch_size=32,
        verbose=2,
        callbacks=[],
    )

    show_img(model.predict(train_data, batch_size=128, verbose=0)[0, ..., 0] * 255)

    model.save(SAVE_MODEL_PATH)

    # loss = model.evaluate(test_data, test_target, verbose=0)
    # print(f"loss: {loss:.4g}")


def create_model():
    main_in = keras.layers.Input((None, None, 1))

    tensor = keras.layers.Conv2D(8, (9, 9), activation="relu", padding="same")(main_in)
    tensor = keras.layers.Conv2D(16, (9, 9), activation="relu", padding="same")(tensor)
    tensor = keras.layers.Conv2D(32, (9, 9), activation="relu", padding="same")(tensor)
    tensor = keras.layers.Conv2D(16, (9, 9), activation="relu", padding="same")(tensor)
    tensor = keras.layers.Conv2D(1, (3, 3), padding="same")(tensor)
    tensor = allowance(tensor)
    tensor = keras.layers.Activation("relu")(tensor)

    model = keras.models.Model(main_in, tensor)
    model.compile(optimizer="adam", loss="mse", metrics=[])
    return model


def create_preprocess_model():
    main_in = keras.layers.Input((None, None, 1))
    tensor = allowance(main_in)

    model = keras.models.Model(main_in, tensor)
    model.compile(optimizer="adam", loss="mse", metrics=[])
    return model


def allowance(tensor):
    tensor = keras.layers.MaxPool2D()(tensor)
    tensor = Binarize()(tensor)
    return tensor


class Binarize(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return keras.backend.tanh(inputs * 1024)


def show_img(img):
    import PIL.Image
    img = PIL.Image.fromarray(np.uint8(img))
    img.show()


# -----
main()
