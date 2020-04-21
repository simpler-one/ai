import numpy as np
import keras
from images.irregular_symbol_generator import IrregularSymbolGenerator
from layers import ArcFace
import os
from keras.datasets import mnist

BASE_MODEL_PATH = "./output/mnist.h5"
SAVE_MODEL_PATH = "./output/mnist-base.h5"

CATEGORIES = 10
LAST_FEAT = "last-feat"


def main():
    (train_data, train_target), (test_data, test_target) = mnist.load_data()
    train_data = train_data[..., None]
    test_data = test_data[..., None]
    train_target = keras.utils.to_categorical(train_target)
    test_target = keras.utils.to_categorical(test_target)

    main_in = keras.layers.Input(train_data.shape[1:], name="main-in")
    label_in = keras.layers.Input((CATEGORIES,), name="label-in")

    tensor = keras.layers.Conv2D(16, (5, 5), activation="relu", padding="same", name="conv-1")(main_in)
    tensor = keras.layers.MaxPool2D((3, 3), strides=2)(tensor)
    tensor = keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv-2")(tensor)
    tensor = keras.layers.Flatten()(tensor)
    tensor = keras.layers.Dropout(0.25)(tensor)
    tensor = keras.layers.Dense(64, activation="relu", name="a")(tensor)
    tensor = keras.layers.BatchNormalization(name=LAST_FEAT)(tensor)
    tensor = ArcFace(CATEGORIES)([tensor, label_in])

    model = keras.models.Model([main_in, label_in], tensor)

    if not os.path.exists(BASE_MODEL_PATH):
        print("Base model: not found")
    else:
        print("Base model: found")
        base_model = keras.models.load_model(SAVE_MODEL_PATH)
        layers = ("conv-1", "conv-2")
        for l in layers:
            model.get_layer(l).set_weights(base_model.get_layer(l).get_weights())
            model.get_layer(l).trainable = False

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()

    model.fit(
        [train_data, train_target], train_target,
        validation_data=([test_data, test_target], test_target),
        epochs=50,
        callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)],
        batch_size=128,
        verbose=2
    )

    centroid = model.layers[-1].get_weights()[0]
    centroid = centroid / np.linalg.norm(centroid, axis=0)

    predict_model = keras.models.Model(
        main_in,
        keras.layers.Dense(CATEGORIES)(model.get_layer(LAST_FEAT).output)  # cos-sim
    )
    predict_model.layers[-1].set_weights([centroid, np.zeros((CATEGORIES,))])

    predict_model.save(SAVE_MODEL_PATH)

    loss, acc = model.evaluate([test_data, test_target], test_target, verbose=0)
    print(f"acc: {acc:.2%}, loss: {loss:.4g}")


# -----
main()
