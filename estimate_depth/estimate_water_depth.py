import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt

import generate_data.segmentation as segmentation
import estimate_depth.models.cnn as cnn

IMG_WIDTH = 512
IMG_HEIGHT = 512


def _get_csv_key_from_file(filename, round_to=15):
    """
    The labels exist only for every :15 minutes, we store a picture every 5 minutes.
    Thus we round to the nearest :15 minutes to get the closest ground truth for the current image
    """
    # remove .png
    filename = filename[:-4]
    date, time = filename.split("_")
    hour, minutes = time.split(":")

    next_quarter = round_to * round(int(minutes) / round_to)
    next_quarter = "0" + str(next_quarter) if next_quarter < 10 else str(next_quarter)
    if next_quarter == "60":
        hour = str(int(hour) + 1)
        next_quarter = "00"

    return "{d}_{h}:{m}".format(d=date, h=hour, m=next_quarter)


def _load_dataset(path, round_to=15):
    x_train, y_train = [], []
    x_val, y_val = [], []

    heights = dict(pd.read_csv(path + "labels.csv").to_numpy())
    files = os.listdir(path)

    for i, f in enumerate(files):
        if f == "labels.csv":
            continue

        img_path = path + f
        img = cv2.imread(img_path)
        img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255

        # remove extension from image
        key = _get_csv_key_from_file(f, round_to=round_to)
        gauge_height = np.float32(heights[key])

        # the size of the validation set is 10% of the original dataset
        if i % 8 == 0:
            x_val.append(img)
            y_val.append(gauge_height)
        else:
            x_train.append(img)
            y_train.append(gauge_height)

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val)


def _plot_model(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


def train_and_predict(dataset_path, round_to=15):
    x_train, y_train, x_test, y_test = _load_dataset(dataset_path, round_to=round_to)

    x_train = segmentation.predict_on_learned_model(x_train)
    x_test = segmentation.predict_on_learned_model(x_test)
    print(len(x_train), len(x_test))

    model: Model = cnn.create_cnn_model(x_train[0].shape)
    model.summary()

    history = model.fit(
        x_train, y_train,
        epochs=100,
        shuffle=True,
    )

    result = model.evaluate(x_test, y_test)
    performance = dict(zip(model.metrics_names, result))
    print("Result:", performance)

    test_set_size = len(y_test)
    predictions = model.predict(x_test)

    compare = pd.DataFrame(data={
        "original": y_test.reshape((test_set_size,)),
        "predictions": predictions.reshape((test_set_size,)),
    })
    print(compare)

    relative_errors = [abs(pred - truth) / truth for pred, truth in zip(predictions, y_test)]
    avg_relative_error = sum(relative_errors) / test_set_size
    median_relative_error = sorted(relative_errors)[test_set_size // 2]
    max_relative_error = max(relative_errors)

    print("Avg relative error:", avg_relative_error)
    print("Median relative error:", median_relative_error)
    print("Max relative error:", max_relative_error)

    # _plot_model(history)
