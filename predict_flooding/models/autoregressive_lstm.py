import tensorflow as tf
from tensorflow.keras import Model, layers


class AutoRegressiveLSTM(Model):

    def __init__(self, out_steps, **kwargs):
        super(AutoRegressiveLSTM, self).__init__(name="AutoRegressiveLSTM", **kwargs)
        self.out_steps = out_steps
        self.lstm_cell = layers.LSTMCell(64)
        self.lstm_rnn = layers.RNN(self.lstm_cell, return_state=True)
        self.dense = layers.Dense(2)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)

        return prediction, state

    def call(self, inputs, training=None, mask=None):
        predictions = []
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)

            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def get_config(self):
        return super(AutoRegressiveLSTM, self).get_config()
