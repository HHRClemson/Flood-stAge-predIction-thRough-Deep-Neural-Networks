import tensorflow as tf
from tensorflow.keras import models, Model, layers


# Create a standard CNN regression model
def create_cnn_model(img_shapes) -> Model:
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(32, 3, 3, activation="relu", input_shape=img_shapes))
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
