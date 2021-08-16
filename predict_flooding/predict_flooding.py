import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt

from predict_flooding.window_generator import SlidingWindowGenerator
from predict_flooding.models import *

OUT_STEPS = 10
NUM_EPOCHS = 100


def evaulate_baseline(_, test_ds):
    # since the baseline model does not predict anything, we do not need the training ds
    model: Model = baseline.Baseline()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    performance = model.evaluate(test_ds)
    return performance


def evaluate_dense(train_ds, test_ds):
    model: Model = dense.Dense()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    _ = model.fit(train_ds, epochs=NUM_EPOCHS)
    performance = model.evaluate(test_ds)
    return performance


def evaluate_cnn(train_ds, test_ds):
    model: Model = cnn.CNN(out_steps=OUT_STEPS)
    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    _ = model.fit(train_ds, epochs=NUM_EPOCHS)
    performance = model.evaluate(test_ds)
    return performance


def evaluate_lstm(train_ds, test_ds):
    model: Model = lstm.LSTM()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    _ = model.fit(train_ds, epochs=NUM_EPOCHS)
    performance = model.evaluate(test_ds)
    return performance


def train_and_predict(path):
    df = pd.read_csv(path)
    train_test_split = int(len(df) * 0.8)
    train_df = df[0:train_test_split]
    test_df = df[train_test_split:]
    #print(train_df.head())

    """Normalize the data by using the training mean and standard deviation."""
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    window_generator: SlidingWindowGenerator = SlidingWindowGenerator(
        input_width=100, label_width=1, shift=OUT_STEPS,
        train_df=train_df, test_df=test_df)

    train_ds = window_generator.train_dataset
    test_ds = window_generator.test_dataset

    baseline_performance = evaulate_baseline(train_ds, test_ds)
    dense_performance = evaluate_dense(train_ds, test_ds)
    cnn_performance = evaluate_cnn(train_ds, test_ds)
    lstm_performance = evaluate_lstm(train_ds, test_ds)

    print("\n".join([
        "Baseline performance: {}".format(baseline_performance),
        "Dense performance: {}".format(dense_performance),
        "CNN performance: {}".format(cnn_performance),
        "LSTM performance: {}".format(lstm_performance)
    ]))


if __name__ == "__main__":
    train_and_predict("datasets/time_series/chattahoochee.csv")
