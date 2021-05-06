from typing import Dict
import tensorflow as tf
from tensorflow import keras
from sve.bi_interaction_pooling import BiInteractionPooling
from sve.sampling_layer import Sampling


class SVE(keras.Model):

    def __init__(self, latent_dim=5, **kwargs):
        super(SVE, self).__init__(**kwargs)
        self.encoder: keras.Model = self._build_encoder()
        self.decoder: keras.Model = self._build_decoder(latent_dim)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @staticmethod
    def _build_encoder() -> keras.Model:
        """
        Generate the encoding FFNN consisting out of 5 layers:
        1) Embedding Layer
        2) Bi-Interaction Layer
        3) Normal dense (hidden) Layer
        4) Mean and the Variance of the latent state distributions
        5) Sampling Layer
        :return:
        """
        encoder: keras.Model = keras.Sequential(name="Encoder")
        encoder.add(keras.Input(shape=(28, 28)))
        encoder.add(keras.layers.Embedding())
        encoder.add(BiInteractionPooling())
        encoder.add(keras.layers.Dense(activation="sigmoid"))

        z_mean = keras.layers.Dense(name="z_mean")
        z_log_var = keras.layers.Dense(name="z_log_var")
        sampling = Sampling()([z_mean, z_log_var])
        encoder.add(sampling)

        return encoder

    @staticmethod
    def _build_decoder(latent_dim) -> keras.Model:
        """
        Generate the decoding FFNN.
        :return:
        """
        decoder: keras.Model = keras.Sequential(name="Decoder")
        decoder.add(keras.Input(shape=(latent_dim,)))
        decoder.add(keras.layers.LSTM(return_sequence=True))

        return decoder

    def train_step(self, data) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

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

    def __str__(self):
        return "{encoder}\n{decoder}".format(
            encoder=self.model_to_str(self.encoder),
            decoder=self.model_to_str(self.decoder)
        )

    @staticmethod
    def model_to_str(model: keras.Model) -> str:
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        return "\n".join(summary)
