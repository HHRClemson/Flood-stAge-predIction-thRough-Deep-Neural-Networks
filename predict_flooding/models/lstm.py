from tensorflow.keras import models, Model, layers


class LSTM(Model):

    def __init__(self, **kwargs):
        super(LSTM, self).__init__(name="LSTM", **kwargs)
        self.model: Model = self._create_model()

    @staticmethod
    def _create_model() -> Model:
        model = models.Sequential()
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.Dense(1))
        return model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super(LSTM, self).get_config()
