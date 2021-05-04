import numpy as np
from tensorflow import keras
from sve.sve_model import SVE


def train_mnist():
    # Train the SVE on the classical MNIST dataset
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    sve = SVE()
    sve.compile(optimizer=keras.optimizers.Adam())
    sve.fit(mnist_digits, epochs=30, batch_size=128)


if __name__ == "__main__":
    train_mnist()
