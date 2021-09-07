import tensorflow as tf
from tensorflow.keras import models, Model, layers


def _wape(y, y_pred):
    """Weighted Average Percentage Error metric in the interval [0; 100]"""
    nominator = tf.reduce_sum(tf.abs(tf.subtract(y, y_pred)))
    denominator = tf.add(tf.reduce_sum(tf.abs(y)), K.epsilon())
    wape = tf.scalar_mul(100.0, tf.divide(nominator, denominator))
    return wape


def create_cnn_model(img_shapes) -> Model:
    """Create a standard CNN regression model"""
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
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError(name="MAE"),
                 tf.metrics.RootMeanSquaredError(name="RMSE"),
                 _wape,
                 tf.metrics.MeanAbsolutePercentageError(name="MAPE")],
    )

    return cnn
