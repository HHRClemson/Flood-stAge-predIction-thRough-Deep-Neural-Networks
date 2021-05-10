import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from sve.sve_model import SVE


def train_mnist() -> keras.Model:
    # Train the SVE on the classical MNIST dataset
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = mnist_digits.astype("float32") / 255

    print(x_train.shape, x_test.shape, mnist_digits.shape)

    sve = SVE()
    sve.summary()
    sve.compile(optimizer=keras.optimizers.Adam())
    sve.fit(
        mnist_digits,
        epochs=30,
        batch_size=128,
    )

    return sve


def plot_latent_space(sve, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))

    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = sve.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


if __name__ == "__main__":
    print("Start training the SVE...")
    trained_sve: keras.Model = train_mnist()
    print(trained_sve)

    plot_latent_space(trained_sve)
