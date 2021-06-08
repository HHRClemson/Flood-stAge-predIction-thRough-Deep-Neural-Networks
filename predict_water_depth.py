import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import models, Model, layers

import matplotlib.pyplot as plt


def _get_csv_key_from_file(filename):
    """
    The labels exist only for every :15 minutes, we store a picture every 5 minutes.
    Thus we round to the nearest :15 minutes to get the closest ground truth for the current image
    """
    # remove .png extension
    filename = filename[:-4]

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


def _create_model() -> Model:
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(32, 3, 3, activation="relu", input_shape=(450, 800, 3)))
    cnn.add(layers.MaxPool2D((2, 2)))
    cnn.add(layers.Conv2D(64, 3, 3, activation="relu"))
    cnn.add(layers.MaxPooling2D(2, 2))
    cnn.add(layers.Conv2D(64, 3, 3, activation="relu"))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(128, activation="relu"))
    cnn.add(layers.Dense(64, activation="relu"))
    cnn.add(layers.Dense(1, activation="linear"))

    cnn.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError()
    )

    return cnn


def _plot_model(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


def train_and_predict():
    x_train, y_train, x_val, y_val = _load_dataset("./datasets/RockyCreek/")

    model: Model = _create_model()
    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    history = model.fit(
        x_train, y_train,
        epochs=30,
        validation_data=(x_val, y_val),
        callbacks=[callback]
    )

    predictions_val = model.predict(x_val)
    compare = pd.DataFrame(data={
        "original": y_val.reshape((len(y_val),)),
        "predictions": predictions_val.reshape((len(predictions_val),))
    })
    print(compare)

    _plot_model(history)
