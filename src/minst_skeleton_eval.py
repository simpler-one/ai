import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.datasets import mnist

SKELETON_ITERATIONS = 4
SKELETON_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# MODEL_PATH = "./output/mnist-dense.h5"
MODEL_PATH = "./output/mnist-skeleton.h5"


def main():
    _, (test_data, test_target) = mnist.load_data()
    irr_data = np.array([cv2.dilate(d, np.ones((3, 3)), iterations=2) for d in test_data])

    # test_data = test_data[..., None] / 255
    # irr_data = irr_data[..., None] / 255
    test_data = np.array([[d, preprocess(d)] for d in test_data]).transpose((0, 2, 3, 1)) / 255
    irr_data = np.array([[d, preprocess(d)] for d in irr_data]).transpose((0, 2, 3, 1)) / 255
    # test_data = np.array([preprocess(d) for d in test_data])[..., None] / 255
    # irr_data = np.array([preprocess(d) for d in irr_data])[..., None] / 255

    for i, d in enumerate(irr_data[:100, :, :, 1]):
        label = test_target[i]
        save_img(d * 255, "mnist-skeleton-bold", f"{label}-{i}")

    test_target = keras.utils.to_categorical(test_target)

    model = keras.models.load_model(MODEL_PATH)

    print("\n## Normal")
    loss, acc = model.evaluate(test_data, test_target, batch_size=128, verbose=0)
    print(f"acc: {acc:.2%}, loss: {loss:.4g}")

    predictions = model.predict(test_data, batch_size=128, verbose=0)
    recall_prb_list = [prediction[np.argmax(actual)] for prediction, actual in zip(predictions, test_target)]

    recall_prb_avg = np.average(recall_prb_list)
    recall_prb_overview = [f"{p:.4g}" for p in np.percentile(recall_prb_list, np.arange(95, 0, -5))]
    print("\n-- Normal recall prob --")
    print(f"avg: {recall_prb_avg}")
    print(f"percentiles(95-5, 5): {recall_prb_overview}")

    print("\n## Irr")
    loss, acc = model.evaluate(irr_data, test_target, batch_size=128, verbose=0)
    print(f"acc: {acc:.2%}, loss: {loss:.4g}")

    irr_predictions = model.predict(irr_data, batch_size=128, verbose=0)
    irr_prb_list = [prd[np.argmax(actual)] for prd, actual in zip(irr_predictions, test_target)]

    irr_prb_avg = np.average(irr_prb_list)
    irr_prb_overview = [f"{p:.4g}" for p in np.percentile(irr_prb_list, np.arange(95, 0, -5))]
    print("\n-- Irregular recall prob --")
    print(f"avg: {irr_prb_avg}")
    print(f"percentiles(95-5, 5): {irr_prb_overview}")


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


def save_img(img, dir_name, name):
    import PIL.Image
    img = PIL.Image.fromarray(np.uint8(img))
    img.save(f"./output/{dir_name}/{name}.png")


# -----
main()
