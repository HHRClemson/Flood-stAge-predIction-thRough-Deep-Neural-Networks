import tensorflow as tf
from tensorflow.keras import models, Model, layers


class Dense(Model):

    def __init__(self, out_steps, **kwargs):
        super(Dense, self).__init__(name="Dense", **kwargs)
        self.out_steps = out_steps
        self.model: Model = self._create_model()

    def _create_model(self) -> Model:
        model = models.Sequential()
        model.add(layers.Lambda(lambda x: x[:, -1:, :]))
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(self.out_steps,
                               kernel_initializer=tf.initializers.zeros()))
        model.add(layers.Reshape([self.out_steps, 1]))
        return model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super(Dense, self).get_config()


if __name__ == "__main__":
    model = Dense(48)
    m = model.model
    m.build((None, *(100, 2)))
    m.summary()
    tf.keras.utils.plot_model(m, to_file="dense.png", show_shapes=True)
