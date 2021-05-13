import tensorflow as tf
from tensorflow.keras import layers


class BiInteractionPooling(layers.Layer):
    """
    Bi-Interaction Pooling Layer from the paper Neural Factorization Machines
    for Sparse Predictive Analytics by He et al. (https://arxiv.org/abs/1708.05027)
    Pooling operation to convert a set of embedding vectors to one vector.
    """

    def __init__(self, **kwargs):
        super(BiInteractionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BiInteractionPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        embeddings = inputs

        square_of_sum = tf.square(tf.reduce_sum(embeddings, axis=1, keepdims=False))
        squared_sum = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=False)

        cross_term = 0.5 * tf.subtract(square_of_sum, squared_sum)
        print("Cross term: ", cross_term)
        return cross_term
