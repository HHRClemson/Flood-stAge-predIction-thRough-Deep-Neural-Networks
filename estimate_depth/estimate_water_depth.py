import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import Model
from sklearn.model_selection import KFold

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
    x, y = [], []

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

        x.append(img)
        y.append(gauge_height)

    return np.asarray(x), np.asarray(y)


def _plot_model(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


def train_and_predict(dataset_path, n_folds=10, round_to=15):
    x, y = _load_dataset(dataset_path, round_to=round_to)
    x = segmentation.predict_on_learned_model(x)
    input_shape = x[0].shape

    kf = KFold(n_splits=n_folds, shuffle=True)

    best_loss = float("inf")
    best_performance = None

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("len train:", len(x_train))
        print("len test:", len(x_test))

        model: Model = cnn.create_cnn_model(input_shape)
        model.fit(
            x_train, y_train,
            epochs=100,
        )

        result = model.evaluate(x_test, y_test)
        performance = dict(zip(model.metrics_names, result))

        if performance["loss"] < best_loss:
            best_loss = performance["loss"]
            best_performance = performance

    print("Best performance:", best_performance)
