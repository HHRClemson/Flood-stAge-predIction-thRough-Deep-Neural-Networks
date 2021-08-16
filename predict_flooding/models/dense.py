from tensorflow.keras import models, Model, layers


class Dense(Model):

    def __init__(self, **kwargs):
        super(Dense, self).__init__(name="Dense", **kwargs)
        self.model: Model = self._create_model()

    @staticmethod
    def _create_model() -> Model:
        model = models.Sequential()
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1))
        return model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super(Dense, self).get_config()
