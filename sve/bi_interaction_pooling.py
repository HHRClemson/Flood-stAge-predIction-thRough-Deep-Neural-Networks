"""
Bi-Interaction Pooling Layer from the paper Neural Factorization Machines
for Sparse Predictive Analytics by He et al. (https://arxiv.org/abs/1708.05027)
Pooling operation to convert a set of embedding vectors to one vector.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class BiInteractionPooling(layers.Layer):
    """
    Reference:
    https://deepctr-doc.readthedocs.io/en/latest/deepctr.layers.interaction.html#deepctr.layers.interaction.BiInteractionPooling
    """

    def __init__(self, dimension, **kwargs):
        super(BiInteractionPooling, self).__init__(**kwargs)
        self.dimension = dimension

    def build(self, input_shape):
        if len(input_shape) != self.dimension:
            raise ValueError(
                "Build: Unexpected inputs dimensions {given}, expected {expected} dimensions".format(
                    given=len(input_shape), expected=self.dimension))

        super(BiInteractionPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != self.dimension:
            raise ValueError(
                "Call: Unexpected inputs dimensions {given}, expected {expected} dimensions".format(
                    given=K.ndim(inputs), expected=self.dimension))

        concated_embeds_value = inputs
        square_of_sum = tf.square(
            tf.reduce_sum(concated_embeds_value, axis=1, keepdims=False))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keepdims=False)
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term

    def compute_output_shape(self, input_shape):
        return None, input_shape[-1]
