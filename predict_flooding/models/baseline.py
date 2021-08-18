import tensorflow as tf
from tensorflow.keras import Model


class Baseline(Model):
    """
    Simple Baseline model which only returns the given input without making any prediction.
    Since the gauge height raises slowly, this is a reasonable assumption over a short
    period of time and will function as a performance baseline for other models.
    """

    def __init__(self, out_steps, **kwargs):
        super(Baseline, self).__init__(name="Baseline", **kwargs)
        self.out_steps = out_steps

    def call(self, inputs, training=None, mask=None):
        return tf.tile(inputs[:, -1:, :], [1, self.out_steps, 1])

    def get_config(self):
        return super(Baseline, self).get_config()
