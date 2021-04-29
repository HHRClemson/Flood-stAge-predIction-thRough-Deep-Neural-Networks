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

    def __init__(self, **kwargs):
        super(BiInteractionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions {given}, expected 3 dimensions"
                    .format(given=len(input_shape)))

        super(BiInteractionPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions {given}, expected 3 dimensions"
                    .format(given=K.ndim(inputs)))

        dim = int(inputs.get_shape()[-1])
        hidden_nn_layers = [inputs]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)

            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)

            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])

            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID')

            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = self.reduce_sum(result, -1, keep_dims=False)

        return result

    def compute_output_shape(self, input_shape):
        pass

    @staticmethod
    def reduce_sum(input_tensor,
                   axis=None,
                   keep_dims=False,
                   name=None,
                   reduction_indices=None):
        try:
            return tf.reduce_sum(input_tensor,
                                 axis=axis,
                                 keep_dims=keep_dims,
                                 name=name,
                                 reduction_indices=reduction_indices)
        except TypeError:
            return tf.reduce_sum(input_tensor,
                                 axis=axis,
                                 keepdims=keep_dims,
                                 name=name)
