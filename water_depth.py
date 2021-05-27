import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models, Model, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


def append_ext(fn):
    """Append the extensions to the .csv labels"""
    return fn + ".png"


def load_dataset():
    dataset_path = "./datasets/RockyCreek/"

    traindf = pd.read_csv(dataset_path + "labels.csv", dtype=str, sep=", ")
    print(traindf)
    testdf = pd.read_csv(dataset_path + "labels.csv", dtype=str, sep=", ")

    traindf["time"] = traindf["time"].apply(append_ext)
    testdf["time"] = testdf["time"].apply(append_ext)

    datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=dataset_path,
        x_col="time",
        y_col="height",
        subset="training",
        batch_size=32,
        shuffle=True,
        class_mode="sparse",
        target_size=(800, 450, 3)
    )

    valid_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=dataset_path,
        x_col="time",
        y_col="height",
        subset="validation",
        batch_size=32,
        shuffle=True,
        class_mode="sparse",
        target_size=(800, 450, 3)
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255.)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=dataset_path,
        x_col="time",
        y_col="height",
        batch_size=32,
        shuffle=False,
        class_mode=None,
        target_size=(800, 450, 3)
    )

    return train_generator, valid_generator, test_generator


def create_model() -> Model:
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(32, 3, 3, activation="relu", input_shape=(800, 450, 3)))
    cnn.add(layers.MaxPool2D((2, 2)))
    cnn.add(layers.Conv2D(64, 3, 3, activation="relu"))
    cnn.add(layers.MaxPooling2D(2, 2))
    cnn.add(layers.Conv2D(64, 3, 3, activation="relu"))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(64, activation="relu"))
    cnn.add(layers.Dense(10))

    cnn.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError,
        metrics=["accuracy"]
    )

    return cnn


def plot_model(history):
    pass


if __name__ == "__main__":
    train_gen, valid_gen, test_gen = load_dataset()

    model: Model = create_model()
    model.summary()

    STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
    STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size
    STEP_SIZE_TEST = test_gen.n // test_gen.batch_size

    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_gen,
        validation_steps=STEP_SIZE_VALID,
        epochs=10
    )

    plot_model(history)
