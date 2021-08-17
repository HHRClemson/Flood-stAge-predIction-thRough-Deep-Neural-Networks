from tensorflow.keras import models, Model, layers


class CNN(Model):

    def __init__(self, out_steps, conv_width=3, **kwargs):
        super(CNN, self).__init__(name="CNN", **kwargs)
        # number of steps to predict into the future
        self.out_steps = out_steps
        self.conv_width = conv_width

        self.model: Model = self._create_model()

    def _create_model(self) -> Model:
        model = models.Sequential()
        model.add(layers.Conv1D(filters=32,
                                kernel_size=(self.conv_width,),
                                activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(1))
        return model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super(CNN, self).get_config()
