from typing import Dict
import tensorflow as tf
from tensorflow import keras

from sve.bi_interaction_pooling import BiInteractionPooling
from sve.sampling_layer import Sampling


class SVE(keras.Model):

    def __init__(self, latent_dim=2, **kwargs):
        super(SVE, self).__init__(**kwargs)
        self.encoder: keras.Model = self._build_encoder(latent_dim)
        self.decoder: keras.Model = self._build_decoder(latent_dim)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @staticmethod
    def _build_encoder(latent_dim) -> keras.Model:
        """
        Generate the encoding FFNN consisting out of the following 5 layers:
        1) Input Layer
        2) Embedding Layer
        3) Bi-Interaction Pooling Layer
        4) Normal dense (hidden) Layer
        5) Sampling Layer with the mean and the log var of the latent space distribution
        """
        inputs = keras.Input(shape=(784,))
        x = keras.layers.Embedding(input_dim=784, output_dim=64)(inputs)
        x = BiInteractionPooling()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    @staticmethod
    def _build_decoder(latent_dim) -> keras.Model:
        """
        Generate the decoding FFNN.
        """
        inputs = keras.Input(shape=(latent_dim,))
        x = keras.layers.Dense(64, activation="relu")(inputs)
        outputs = keras.layers.Dense(784, activation="sigmoid")(x)

        return keras.Model(inputs, outputs, name="decoder")

    def train_step(self, data) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            print("Encoded:", z_mean, z_log_var, z)
            print("Decoded:", reconstruction)
            print("Input shape:", data.shape, "Reconstruction shape:", reconstruction.shape)

            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction)))
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
