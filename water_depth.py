import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import models, Model, layers

import matplotlib.pyplot as plt


def load_dataset(path):
    x_train, y_train = [], []
    x_val, y_val = [], []

    heights = dict(pd.read_csv(path + "labels.csv").to_numpy())

    files = os.listdir(path)

    for i, file in enumerate(files):
        if file == "labels.csv":
            continue
        img = cv2.imread(path + file).astype(np.float32)

        # remove extension from image
        key = file[:-4]

        # validation set is around 20%
        if i % 5 == 0:
            x_val.append(img)
            y_val.append(np.float32(heights[key]))
        else:
            x_train.append(img)
            y_train.append(np.float32(heights[key]))
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val)


def create_model() -> Model:
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(32, 3, 3, activation="relu", input_shape=(450, 800, 3)))
    cnn.add(layers.MaxPool2D((2, 2)))
    cnn.add(layers.Conv2D(64, 3, 3, activation="relu"))
    cnn.add(layers.MaxPooling2D(2, 2))
    cnn.add(layers.Conv2D(64, 3, 3, activation="relu"))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(64, activation="relu"))
    cnn.add(layers.Dense(10, activation="linear"))

    cnn.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError()
    )

    return cnn


def plot_model(history):
    pass


if __name__ == "__main__":
    x_train, y_train, x_val, y_val = load_dataset("./datasets/RockyCreek/")

    model: Model = create_model()
    model.summary()

    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_val, y_val)
    )

    plot_model(history)
