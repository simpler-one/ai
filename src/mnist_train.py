import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from images.irregular_symbol_generator import IrregularSymbolGenerator
from layers import ArcFace, StaticCosineSimilarity
import os

BASE_MODEL_PATH = "./output/mnist-base.h5"
SAVE_MODEL_PATH = "./output/mnist.h5"

CATEGORIES = 10
MAIN_IN = "main-in"
LAST_FEAT = "last-feat"


def main():
    (train_data, train_target), (test_data, test_target) = mnist.load_data()
    train_data = train_data[..., None] / 255
    test_data = test_data[..., None] / 255
    train_target = keras.utils.to_categorical(train_target, CATEGORIES)
    test_target = keras.utils.to_categorical(test_target, CATEGORIES)

    irr_gen = IrregularSymbolGenerator()

    model = create_model(train_data.shape[1:])
    apply_transfer_learning(model)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()

    for i in range(10):
        irr_data_list = []
        irr_target_list = []

        for _ in range(round(train_data.shape[0] * 0.05)):
            irr_data, irr_target = irr_gen.generate_from(train_data, train_target)
            irr_data_list.append(irr_data)
            irr_target_list.append(irr_target)

        train_data = np.concatenate([train_data, np.stack(irr_data_list)])
        train_target = np.concatenate([train_target, np.stack(irr_target_list)])

        model.fit(
            [train_data, train_target], train_target,
            validation_data=([test_data, test_target], test_target),
            initial_epoch=i * 3, epochs=(i + 1) * 3,
            callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)],
            batch_size=128,
            verbose=2
        )

    centroid = model.layers[-1].get_weights()[0]

    predict_model = keras.models.Model(
        model.get_layer(MAIN_IN).input,
        StaticCosineSimilarity(centroid)(model.get_layer(LAST_FEAT).output)
    )
    predict_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

    predict_model.save(SAVE_MODEL_PATH)

    loss, acc = predict_model.evaluate(test_data, test_target, verbose=0)
    print(f"acc: {acc:.2%}, loss: {loss:.4g}")


def create_model(input_shape):
    main_in = keras.layers.Input(input_shape, name=MAIN_IN)
    label_in = keras.layers.Input((CATEGORIES,), name="label-in")

    tensor = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv-1")(main_in)
    tensor = keras.layers.MaxPool2D((3, 3), strides=2)(tensor)
    tensor = keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv-2")(tensor)
    tensor = keras.layers.Flatten()(tensor)
    tensor = keras.layers.Dropout(0.25)(tensor)
    tensor = keras.layers.Dense(128, activation="relu")(tensor)
    tensor = keras.layers.BatchNormalization(name=LAST_FEAT)(tensor)
    tensor = ArcFace(CATEGORIES)([tensor, label_in])
    # tensor = keras.layers.Dense(CATEGORIES, activation="softmax")(tensor)

    return keras.models.Model([main_in, label_in], tensor)


def apply_transfer_learning(model):
    """

    :param keras.models.Model model:
    :return:
    """
    if not os.path.exists(BASE_MODEL_PATH):
        print("Base model: not found")
    else:
        print("Base model: found")
        base_model = keras.models.load_model(BASE_MODEL_PATH)
        layers = ("conv-1", "conv-2")
        for l in layers:
            model.get_layer(l).set_weights(base_model.get_layer(l).get_weights())
            model.get_layer(l).trainable = False


# -----
main()
