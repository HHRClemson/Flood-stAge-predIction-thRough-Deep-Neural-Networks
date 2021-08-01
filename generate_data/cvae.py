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
        super(CVAE, self).__init__(name="CVAE", **kwargs)

        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride_size = stride_size

        self.filters = [8, 16, 32, 64, 128]
        adam_learning_rate = 1e-4

        self.encoder: Model = self._create_encoder()
        self.enc_optimizer = tf.keras.optimizers.Adam(adam_learning_rate)

        self.decoder: Model = self._create_decoder()
        self.dec_optimizer = tf.keras.optimizers.Adam(adam_learning_rate)

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker]

    def _create_encoder(self) -> Model:
        """
        Create an encoder to train a proper latent space in which we will
        modify the gauge height.
        The latent space will be the input for the generator
        """
        image_input = layers.Input(shape=(self.image_size, self.image_size, self.channels))
        x = image_input

        for f in self.filters:
            x = layers.Conv2D(
                filters=f, kernel_size=self.kernel_size,
                strides=self.stride_size, padding="same")(x)
            x = layers.Activation("relu")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim)(x)

        return Model(inputs=image_input, outputs=x, name="Encoder")

    def _create_decoder(self) -> Model:
        """Create a decoder as an discriminator for the encoder training"""
        latent_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(8 * 8 * 64)(latent_input)
        x = layers.Activation("relu")(x)
        x = layers.Reshape(target_shape=(8, 8, 64))(x)

        for f in reversed(self.filters):
            x = layers.Conv2DTranspose(
                filters=f, kernel_size=self.kernel_size,
                strides=self.stride_size, padding="same")(x)
            x = layers.Activation("relu")(x)

        x = layers.Conv2DTranspose(
            filters=1, kernel_size=self.kernel_size,
            strides=1, padding="same")(x)
        return Model(inputs=latent_input, outputs=x, name="Decoder")

    @tf.function
    def train_step(self, image):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            reconstructed = self.decoder(self.encoder(image, training=True), training=True)

            loss = tf.reduce_mean(tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(image, reconstructed)))

        enc_grads = enc_tape.gradient(loss, self.encoder.trainable_variables)
        self.enc_optimizer.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))

        dec_grads = dec_tape.gradient(loss, self.decoder.trainable_variables)
        self.dec_optimizer.apply_gradients(zip(dec_grads, self.decoder.trainable_variables))

        self.reconstruction_loss_tracker.update_state(loss)
        return {"reconstruction_loss": self.reconstruction_loss_tracker.result()}

    def call(self, inputs, training=None, mask=None):
        return self.encoder(inputs)

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
