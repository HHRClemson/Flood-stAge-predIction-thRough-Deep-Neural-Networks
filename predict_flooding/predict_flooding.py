import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt

from predict_flooding.window_generator import SlidingWindowGenerator
from predict_flooding.models import *

OUT_STEPS = 10


def _evaluate_baseline(_, test_ds, show_summary=False):
    # Since the baseline model does not predict anything, we do not need the training ds
    model: Model = baseline.Baseline(label_index=1)
    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    performance = model.evaluate(test_ds, show_summary)
    return performance


def _evaluate_dense(train_ds, test_ds, show_summary=False):
    model: Model = dense.Dense()
    model.summary()

    _, performance = _run_model(model, train_ds, test_ds, show_summary)
    return performance


def _evaluate_cnn(train_ds, test_ds, show_summary=False):
    model: Model = cnn.CNN(out_steps=OUT_STEPS)
    print("\n\nCREATED MODEL\n\n")
    _, performance = _run_model(model, train_ds, test_ds, show_summary)
    return performance


def _evaluate_lstm(train_ds, test_ds, show_summary=False):
    model: Model = lstm.LSTM()
    _, performance = _run_model(model, train_ds, test_ds, show_summary)
    return performance


def _run_model(model: Model, train_ds, test_ds,
               num_epochs=50, patience=5, show_summary=False):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=patience, mode="min")

    if show_summary:
        model.summary()

    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(train_ds, epochs=num_epochs, callbacks=[early_stopping])
    performance = model.evaluate(test_ds)
    return history, performance


def _plot_results(results):
    pass


def train_and_predict(path):
    df = pd.read_csv(path)
    df = df.drop("time", axis=1)
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["perception"] = pd.to_numeric(df["perception"], errors="coerce")

    train_test_split = int(len(df) * 0.8)
    train_df = df[0:train_test_split]
    test_df = df[train_test_split:]

    # Normalize the data by using the training mean and standard deviation.
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0)
    print(train_mean)
    print(train_std)

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    print(train_df.head())
    print(test_df.head())

    window_generator: SlidingWindowGenerator = SlidingWindowGenerator(
        input_width=100, label_width=1, shift=OUT_STEPS,
        train_df=train_df, test_df=test_df,
        label_columns=["height"])
    print(window_generator)

    train_ds = window_generator.train_dataset
    print(train_ds)
    test_ds = window_generator.test_dataset
    print(test_ds)

    #baseline_performance = _evaluate_baseline(train_ds, test_ds, show_summary=True)
    #print("EVALUATED BASELINE:", baseline_performance)

    #dense_performance = _evaluate_dense(train_ds, test_ds, show_summary=True)
    #print("EVALUATED DENSE:", dense_performance)

    cnn_performance = _evaluate_cnn(train_ds, test_ds, show_summary=True)
    print("EVALUATED CNN:", cnn_performance)

    lstm_performance = _evaluate_lstm(train_ds, test_ds, show_summary=True)
    print("EVALUATED LSTM:", lstm_performance)

    print("\n".join([
        "Baseline performance: {}".format(baseline_performance),
        "Dense performance: {}".format(dense_performance),
        "CNN performance: {}".format(cnn_performance),
        "LSTM performance: {}".format(lstm_performance)
    ]))

    _plot_results([baseline_performance, dense_performance,
                   cnn_performance, lstm_performance])


if __name__ == "__main__":
    train_and_predict("datasets/time_series/chattahoochee.csv")
