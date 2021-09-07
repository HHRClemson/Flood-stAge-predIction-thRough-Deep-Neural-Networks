import tensorflow as tf
from tensorflow.keras import models, Model, layers


class LSTM(Model):

    def __init__(self, out_steps, **kwargs):
        super(LSTM, self).__init__(name="LSTM", **kwargs)
        self.out_steps = out_steps
        self.model: Model = self._create_model()

    def _create_model(self) -> Model:
        model = models.Sequential()
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.LSTM(32, return_sequences=False))
        model.add(layers.Dense(self.out_steps,
                               kernel_initializer=tf.initializers.zeros()))
        model.add(layers.Reshape([self.out_steps, 1]))
        return model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super(LSTM, self).get_config()


if __name__ == "__main__":
    model = LSTM(48)
    m = model.model
    m.build((None, *(100, 2)))
    m.summary()
    tf.keras.utils.plot_model(m, to_file="lstm.png", show_shapes=True)
