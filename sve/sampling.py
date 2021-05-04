import tensorflow as tf
from tensorflow import keras


class Sampling(keras.Layer):
    """
    Using the reparameterization trick to obtain the encoded sample vector z by the
    mean and the log of the variance of the latent space distribution.
    """

    def __call__(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
