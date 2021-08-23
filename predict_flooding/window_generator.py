import numpy as np
from typing import Optional

import matplotlib.pyplot as plt
import tensorflow as tf


class SlidingWindowGenerator:
    """
    Class for controlling the window for time series learning as in controlling the
    1) number of steps of the input and label windows
    2) time offset between input and label windows
    3) which features are used as inputs, labels, or both

    We adapt the TensorFlow tutorial about time series forecasting:
    https://www.tensorflow.org/tutorials/structured_data/time_series
    """

    def __init__(self, input_width, label_width, shift,
                 train_df, test_df, label_columns=None):

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift

        self.train_df = train_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    @property
    def train_dataset(self):
        """Generate the training dataset."""
        return self.make_dataset(self.train_df)

    @property
    def test_dataset(self):
        """Generate the testing dataset."""
        return self.make_dataset(self.test_df)

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes manually again.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            batch_size=32)

        return ds.map(self.split_window)

    def plot(self, model: Optional[tf.keras.Model], max_subplots=5):
        """Plot batches of the training dataset for visual results."""
        plot_data = iter(self.train_dataset)
        plt.figure(figsize=(12, 8))

        plot_col = "height"
        plot_col_index = self.column_indices[plot_col]

        for i in range(max_subplots):
            inputs, labels = next(plot_data)
            plt.subplot(max_subplots, 1, i + 1)
            plt.ylabel("{} [normed]".format(plot_col))
            plt.plot(self.input_indices, inputs[i, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            plt.scatter(self.label_indices, labels[i, :, label_col_index],
                        edgecolors='k', label="Labels", c="#2ca02c", s=64)

            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[i, :, label_col_index],
                            marker='X', edgecolors='k', label="Predictions",
                            c="#ff7f0e", s=64)

            if i == 0:
                plt.legend()

        plt.xlabel('Time [h]')

        if model is not None:
            plt.savefig("flooding_results/{}-model.png".format(model.name))
        plt.show()
        plt.close()

    def __str__(self):
        return "\n".join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
