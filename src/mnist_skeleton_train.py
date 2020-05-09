import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.datasets import mnist

SKELETON_ITERATIONS = 4
SKELETON_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
CATEGORIES = 10

# SAVE_MODEL_PATH = "./output/mnist-dense.h5"
SAVE_MODEL_PATH = "./output/mnist-skeleton.h5"


def main():
    (train_data, train_target), (test_data, test_target) = mnist.load_data()

    # train_data = train_data[..., None] / 255
    # test_data = test_data[..., None] / 255
    train_data = np.array([[d, preprocess(d)] for d in train_data]).transpose((0, 2, 3, 1)) / 255
    test_data = np.array([[d, preprocess(d)] for d in test_data]).transpose((0, 2, 3, 1)) / 255
    # train_data = np.array([preprocess(d) for d in train_data])[..., None] / 255
    # test_data = np.array([preprocess(d) for d in test_data])[..., None] / 255

    train_target = keras.utils.to_categorical(train_target, CATEGORIES)
    test_target = keras.utils.to_categorical(test_target, CATEGORIES)

    model = create_model(train_data.shape[1:])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()

    model.fit(
        train_data, train_target,
        validation_data=(test_data, test_target),
        epochs=50,
        callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)],
        batch_size=128,
        verbose=2
    )

    model.save(SAVE_MODEL_PATH)
    loss, acc = model.evaluate(test_data, test_target, verbose=0)
    print(f"acc: {acc:.2%}, loss: {loss:.4g}")


def create_model(input_shape):
    main_in = keras.layers.Input(input_shape)

    tensor = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv-1")(main_in)
    tensor = keras.layers.MaxPool2D((3, 3), strides=2)(tensor)
    tensor = keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv-2")(tensor)
    tensor = keras.layers.Flatten()(tensor)
    tensor = keras.layers.Dropout(0.25)(tensor)
    tensor = keras.layers.Dense(128, activation="relu")(tensor)
    tensor = keras.layers.BatchNormalization()(tensor)
    tensor = keras.layers.Dense(CATEGORIES, activation="softmax")(tensor)

    return keras.models.Model(main_in, tensor)


def preprocess(image):
    return np.clip(to_skeleton(image, SKELETON_KERNEL, SKELETON_ITERATIONS), 0, 255)


def to_skeleton(image, kernel, iteration=1):
    img = image
    # img = cv2.threshold(image, 63, 255, cv2.THRESH_BINARY)[1]

    for i in range(iteration):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        tophat = img - opened
        img = eroded | tophat

    # return cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((2, 2)))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# -----
main()
