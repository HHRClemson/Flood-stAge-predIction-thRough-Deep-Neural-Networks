from tensorflow import keras


class BiInteraction(keras.Layer):

    def __init__(self, **kwargs):
        super(BiInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass
