from tensorflow import keras
from sve.biInteraction import BiInteractionPooling


class SVE(keras.Model):

    def __init__(self, **kwargs):
        super(SVE, self).__init__(**kwargs)
        self.encoder: keras.Model = self._build_encoder()
        self.decoder: keras.Model = self._build_decoder()

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
        encoder: keras.Model = keras.Sequential()
        encoder.add(keras.Input(shape=(28, 28)))
        encoder.add(keras.layers.Embedding())
        encoder.add(BiInteractionPooling())
        encoder.add(keras.layers.Dense(activation="sigmoid"))

        x_mean = keras.layers.Dense(name="x_mean")
        x_var = keras.layers.Dense(name="x_var")
        encoder.add([x_mean, x_var])
        encoder.add(keras.layers.Sampling())

        return encoder

    @staticmethod
    def _build_decoder() -> keras.Model:
        """
        Generate the decoding FFNN.
        :return:
        """
        decoder: keras.Model = keras.Sequential()
        return decoder

    def train_step(self, data):
        pass

    def call(self, inputs):
        pass

    def __str__(self):
        return "Encoder: {encoder}\nDecoder: \n{decoder}".format(
            encoder=self.encoder.summary(),
            decoder=self.decoder.summary())
