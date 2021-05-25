import tensorflow as tf
from tensorflow.keras import models, Model, layers
import matplotlib.pyplot as plt


def load_dataset():
    return (), ()


def create_conv_base() -> Model:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    return model


def add_dense_layers(model: Model) -> Model:
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10))
    return model


def create_model() -> Model:
    model = add_dense_layers(create_conv_base())
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.MeanSquaredError,
                  metrics=["accuracy"])
    return model


def plot_model(history):
    pass


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_dataset()

    model: Model = create_model()
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    plot_model(history)
