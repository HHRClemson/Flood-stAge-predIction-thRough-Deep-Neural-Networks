"""
Bi-Interaction Layer from the paper Neural Factorization Machines
for Sparse Predictive Analytics by He et al. (https://arxiv.org/pdf/1708.05027.pdf)
Pooling operation to convert a set of embedding vectors to one vector.

Reference:
 - https://deepctr-doc.readthedocs.io/en/latest/deepctr.layers.interaction.html#deepctr.layers.interaction.BiInteractionPooling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class BiInteractionPooling(keras.Layer):

    def __init__(self, dimension=3, **kwargs):
        super(BiInteractionPooling, self).__init__(**kwargs)
        self.dimension = dimension

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions {given}, expected {expected} dimensions".format(
                    given=len(input_shape), expected=self.dimension))

        super(BiInteractionPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions {given}, expected {expected} dimensions".format(
                    given=K.ndim(inputs), expected=self.dimension))

        concated_embeds_value = inputs
        square_of_sum = tf.square(
            tf.reduce_sum(concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term

    @staticmethod
    def compute_output_shape(input_shape):
        return None, 1, input_shape[-1]
