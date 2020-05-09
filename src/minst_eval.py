import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from etl8g import read_files_etl8g
from layers import StaticCosineSimilarity

NORMAL_MODEL_PATH = "./output/mnist-normal.h5"
PLUS_MODEL_PATH = "./output/mnist-plus.h5"

CUSTOM_LAYERS = {
    "StaticCosineSimilarity": StaticCosineSimilarity
}


def main():
    _, (test_data, test_target) = mnist.load_data()
    test_data = test_data[..., None] / 255
    test_target = keras.utils.to_categorical(test_target)

    irr_data_list, irr_label_list = read_files_etl8g(["./data/ETL8G/ETL8G_01"], (28, 28))
    irr_data = np.array(irr_data_list)[..., None] / 255

    model = keras.models.load_model(PLUS_MODEL_PATH, custom_objects=CUSTOM_LAYERS)

    loss, acc = model.evaluate(test_data, test_target, batch_size=128, verbose=0)
    print(f"acc: {acc:.2%}, loss: {loss:.4g}")

    predictions = model.predict(test_data, batch_size=128, verbose=0)
    recall_prb_list = [prediction[np.argmax(actual)] for prediction, actual in zip(predictions, test_target)]

    recall_prb_avg = np.average(recall_prb_list)
    recall_prb_overview = [f"{p:.4g}" for p in np.percentile(recall_prb_list, np.arange(95, 0, -5))]
    print("\n-- Normal recall cosine similarity --")
    print(f"avg: {recall_prb_avg}")
    print(f"percentiles(95-5, 5): {recall_prb_overview}")

    irr_predictions = model.predict(irr_data, batch_size=128, verbose=0)
    irr_prb_list = [prd[np.argmax(prd)] for prd in irr_predictions]

    irr_prb_avg = np.average(irr_prb_list)
    irr_prb_overview = [f"{p:.4g}" for p in np.percentile(irr_prb_list, np.arange(95, 0, -5))]
    print("\n-- Irregular cosine similarity --")
    print(f"avg: {irr_prb_avg}")
    print(f"percentiles(95-5, 5): {irr_prb_overview}")

    irr_prob_index_list = sorted(enumerate(irr_prb_list), key=lambda item: item[1], reverse=True)
    for irr_prob_i in irr_prob_index_list:
        i, _ = irr_prob_i
        # d = irr_data_list[i]
        label: str = irr_label_list[i]
        prediction = irr_predictions[i]
        prb_category: int = np.argmax(prediction)
        print(f"{label:.10} -> {prb_category} ({prediction[prb_category]:.2g})")


# -----
main()
