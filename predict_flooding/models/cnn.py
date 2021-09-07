import tensorflow as tf
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
        model.add(layers.Lambda(lambda x: x[:, -self.conv_width:, :]))
        model.add(layers.Conv1D(512, activation="relu", kernel_size=self.conv_width))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(self.out_steps,
                               kernel_initializer=tf.initializers.zeros()))
        model.add(layers.Reshape([self.out_steps, 1]))
        return model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super(CNN, self).get_config()


if __name__ == "__main__":
    model = CNN(48)
    m = model.model
    m.build((None, *(100, 2)))
    m.summary()
    tf.keras.utils.plot_model(m, to_file="cnn.png", show_shapes=True)
