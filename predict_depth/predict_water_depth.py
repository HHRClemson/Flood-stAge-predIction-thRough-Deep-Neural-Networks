import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt

import predict_depth.cnn as cnn
import predict_depth.bcnn as bcnn


def _get_csv_key_from_file(filename):
    """
    The labels exist only for every :15 minutes, we store a picture every 5 minutes.
    Thus we round to the nearest :15 minutes to get the closest ground truth for the current image
    """
    # remove .png extension
    filename = filename[:-4]
    # remove filename prefix
    filename = filename.split("-", 1)[1]

    date, time = filename.split("_")
    hour, minutes = time.split(":")

    next_quarter = 15 * round(int(minutes) / 15)
    next_quarter = "0" + str(next_quarter) if next_quarter < 10 else str(next_quarter)

    if next_quarter == "60":
        hour = str(int(hour) + 1)
        next_quarter = "00"

    return "{d}_{h}:{m}".format(d=date, h=hour, m=next_quarter)


def _load_dataset(path):
    x_train, y_train = [], []
    x_val, y_val = [], []

    heights = dict(pd.read_csv(path + "labels.csv").to_numpy())

    files = os.listdir(path)

    for i, file in enumerate(files):
        if file == "labels.csv":
            continue

        img = cv2.imread(path + file).astype(np.float32)

        # remove extension from image
        key = _get_csv_key_from_file(file)

        # the size of the validation set is 20% of the original dataset
        if i % 5 == 0:
            x_val.append(img)
            y_val.append(np.float32(heights[key]))
        else:
            x_train.append(img)
            y_train.append(np.float32(heights[key]))
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val)


def _plot_model(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


def train_and_predict(dataset_path, bayesian=False):
    x_train, y_train, x_val, y_val = _load_dataset(dataset_path)

    if bayesian:
        model: Model = bcnn.create_bcnn_model(len(y_train), x_train[0].shape)
    else:
        model: Model = cnn.create_cnn_model(x_train[0].shape)
    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    history = model.fit(
        x_train, y_train,
        epochs=30,
        validation_data=(x_val, y_val),
        callbacks=[callback]
    )

    val_set_size = len(y_val)
    predictions = model.predict(x_val)

    if bayesian:
        """
        If we trained our model on a bayesian CNN, we can now predict various statistical measurements.
        Here we calculate the mean and the min and max of one standard deviation from the mean
        """
        prediction_mean = np.mean(predictions, axis=1)
        prediction_min = np.min(predictions, axis=1)
        prediction_max = np.max(predictions, axis=1)
        prediction_range = np.max(predictions, axis=1) - np.min(predictions, axis=1)
    else:
        prediction_mean = np.repeat(np.NaN, val_set_size, axis=0)
        prediction_min = np.repeat(np.NaN, val_set_size, axis=0)
        prediction_max = np.repeat(np.NaN, val_set_size, axis=0)
        prediction_range = np.repeat(np.NaN, val_set_size, axis=0)

    compare = pd.DataFrame(data={
        "original": y_val.reshape((val_set_size,)),
        "predictions": predictions.reshape((val_set_size,)),
        "mean": prediction_mean.reshape((val_set_size,)),
        "min": prediction_min.reshape((val_set_size,)),
        "max": prediction_max.reshape((val_set_size,)),
        "range": prediction_range.reshape((val_set_size,)),
    })
    print(compare)

    avg_error = sum([abs(pred - truth) for pred, truth in zip(predictions, y_val)]) / val_set_size
    print("Avg error: ", avg_error)

    _plot_model(history)
