import tensorflow as tf
from tensorflow.keras import Model


class Baseline(Model):
    """
    Simple Baseline model which only returns the given input without making any prediction.
    Since the gauge height raises slowly, this is a reasonable assumption over a short
    period of time and will function as a performance baseline for other models.
    """

    def __init__(self, label_index=None, **kwargs):
        super(Baseline, self).__init__(name="Baseline", **kwargs)
        self.label_index = label_index

    def call(self, inputs, training=None, mask=None):
        if self.label_index is None:
            return inputs
        prediction = inputs[:, :, self.label_index]
        return prediction[:, :, tf.newaxis]

    def get_config(self):
        return super(Baseline, self).get_config()
