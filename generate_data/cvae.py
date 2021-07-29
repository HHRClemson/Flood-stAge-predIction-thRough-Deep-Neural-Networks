import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, Model


class CVAE(Model):
    """
    Creates a Convolutional Variational Autoencoder to learn a latent space
    of the webcam images of the rivers.
    """

    def __init__(self,
                 image_size=256,
                 channels=3,
                 latent_dim=512,
                 kernel_size=3,
                 stride_size=2,
                 **kwargs):
        super(CVAE, self).__init__(name="VAE", **kwargs)

        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride_size = stride_size

        self.filters = [8, 16, 32, 64, 128]

        self.encoder: Model = self._create_encoder()
        self.enc_optimizer = tf.keras.optimizers.Adam(1e-4, loss="binary_crossentropy")

        self.decoder: Model = self._create_decoder()
        self.dec_optimizer = tf.keras.optimizers.Adam(1e-4, loss="binary_crossentropy")

        self.loss_function = tf.keras.losses.BinaryCrossentropy()

    def _create_encoder(self) -> Model:
        """
        Create an encoder to train a proper latent space in which we will
        modify the gauge height.
        The latent space will be the input for the generator
        """
        encoder = models.Sequential(name="Encoder")
        encoder.add(layers.InputLayer(input_shape=(self.image_size, self.image_size, self.channels)))

        for f in self.filters:
            encoder.add(layers.Conv2D(
                filters=f, kernel_size=self.kernel_size,
                strides=self.stride_size, padding="same"))
            encoder.add(layers.Activation("relu"))

        encoder.add(layers.Flatten())
        encoder.add(layers.Dense(self.latent_dim))
        return encoder

    def _create_decoder(self) -> Model:
        """Create a decoder as an discriminator for the encoder training"""
        decoder = models.Sequential(name="Decoder")
        decoder.add(layers.InputLayer(input_shape=self.latent_dim))
        decoder.add(layers.Dense(16 * 16 * 64))
        decoder.add(layers.Activation("relu"))
        decoder.add(layers.Reshape(target_shape=(16, 16, 64)))

        for f in reversed(self.filters):
            decoder.add(layers.Conv2DTranspose(
                filters=f, kernel_size=self.kernel_size,
                strides=self.stride_size, padding="same"))
            decoder.add(layers.Activation("relu"))

        decoder.add(layers.Conv2DTranspose(
            filters=1, kernel_size=self.kernel_size,
            strides=1, padding="same"))
        return decoder

    def train(self) -> Model:
        """Train the encoder"""
        return self.encoder

    @tf.function
    def train_step(self, image):
        """
        We train the encoder by maximizing the evidence lower bound (ELBO) on the
        marginal log-likelihood.
        """
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            reconstructed = self.decoder(self.encoder(image))
            loss = self.loss_function(image, reconstructed)

        enc_grads = enc_tape.gradient(loss, self.encoder.trainable_variables)
        self.encoder.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))

        dec_grads = dec_tape.gradient(loss, self.decoder.trainable_variables)
        self.decoder.apply_gradients(zip(dec_grads, self.decoder.trainable_variables))
        return loss

    def call(self, inputs, training=None, mask=None):
        for img in inputs:
            yield self.encoder(img)

    def get_config(self):
        config = super(CVAE, self).get_config()
        config.update({
            "image_size": self.image_size,
            "channels": self.channels,
            "latent_dim": self.latent_dim
        })
        return config

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.encoder.summary(line_length, positions, print_fn)
        self.decoder.summary(line_length, positions, print_fn)


if __name__ == "__main__":
    cvae = CVAE()
    cvae.summary()

