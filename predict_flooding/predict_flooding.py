import numpy as np
import pandas as pd
from pprint import pprint

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model

from predict_flooding.models import *
from predict_flooding.window_generator import SlidingWindowGenerator

FUTURE_PREDICTIONS = 48
EPOCHS = 0
PATIENCE = max(5, EPOCHS // 5)


def _evaluate_baseline(window: SlidingWindowGenerator, visualize):
    model: Model = baseline.Baseline(FUTURE_PREDICTIONS)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError(name="MAE"),
                           r_square,
                           tf.metrics.RootMeanSquaredError(name="RMSE"),
                           tf.metrics.MeanAbsolutePercentageError(name="MAPE")])
    performance = model.evaluate(window.test_dataset)

    if visualize:
        window.plot(model)

    return performance


def _evaluate_dense(window: SlidingWindowGenerator, visualize):
    model: Model = dense.Dense(FUTURE_PREDICTIONS)
    _, performance = _run_model(model, window, visualize)
    return performance


def _evaluate_cnn(window: SlidingWindowGenerator, visualize):
    model: Model = cnn.CNN(FUTURE_PREDICTIONS)
    _, performance = _run_model(model, window, visualize)
    return performance


def _evaluate_lstm(window: SlidingWindowGenerator, visualize):
    model: Model = lstm.LSTM(FUTURE_PREDICTIONS)
    _, performance = _run_model(model, window, visualize)
    return performance


def _evaluate_autoregressive_lstm(window: SlidingWindowGenerator, visualize):
    model: Model = autoregressive_lstm.AutoRegressiveLSTM(FUTURE_PREDICTIONS)
    _, performance = _run_model(model, window, visualize)
    return performance


def _r_square(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2


def _run_model(model: Model, window: SlidingWindowGenerator,
               visualize, num_epochs=EPOCHS, patience=PATIENCE):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=patience, mode="min")

    model.compile(loss=tf.losses.MeanSquaredError(name="MSE"),
                  metrics=[tf.metrics.MeanAbsoluteError(name="MAE"),
                           _r_square,
                           tf.metrics.RootMeanSquaredError(name="RMSE"),
                           tf.metrics.MeanAbsolutePercentageError(name="MAPE")],
                  optimizer=tf.optimizers.Adam())
    history = model.fit(window.train_dataset, epochs=num_epochs, callbacks=[early_stopping])
    performance = model.evaluate(window.test_dataset)

    if visualize:
        window.plot(model)

    return history, performance


def _plot_performance(performances):
    x = np.arange(len(performances))
    width = 0.15

    # sort the models by metric performance descending
    results = sorted(performances.items(), key=lambda x: x[1][1], reverse=True)

    mae = [m[1][0] for m in results]
    r2 = [m[1][1] for m in results]
    #rmse = [m[1][2] for m in results]
    rmse = [1 for m in results]
    mape = [m[1][3] for m in results]

    model_names = [m[0].upper() for m in results]

    plt.bar(x - 0.3, mae, width, label="Mean Absolute Error")
    plt.bar(x - 0.15, r2, width, label="R Squared Error")
    plt.bar(x + 0.0, rmse, width, label="Root Mean Square Error")
    plt.bar(x + 0.15, mape, width, label="Mean Absolute Percentage Error")
    plt.xticks(ticks=x, labels=model_names, rotation=45)
    plt.ylabel("Metrics")
    plt.legend()

    plt.savefig("flooding_results/performance.png")
    plt.show()
    plt.close()


def train_and_predict(path, visualize=False):
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
    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    window: SlidingWindowGenerator = SlidingWindowGenerator(
        input_width=100, label_width=FUTURE_PREDICTIONS, shift=FUTURE_PREDICTIONS,
        train_df=train_df, test_df=test_df,
        label_columns=["height"])

    baseline_performance = _evaluate_baseline(window, visualize)
    print("EVALUATED BASELINE:", baseline_performance)

    dense_performance = _evaluate_dense(window, visualize)
    print("EVALUATED DENSE:", dense_performance)

    cnn_performance = _evaluate_cnn(window, visualize)
    print("EVALUATED CNN:", cnn_performance)

    lstm_performance = _evaluate_lstm(window, visualize)
    print("EVALUATED LSTM:", lstm_performance)

    performances = {
        "baseline": baseline_performance,
        "dense": dense_performance,
        "cnn": cnn_performance,
        "lstm": lstm_performance,
    }

    print("\n\nPERFORMANCES: [Mean Squared Error, Mean Absolute Error]")
    pprint(performances)

    if visualize or True:
        _plot_performance(performances)


if __name__ == "__main__":
    train_and_predict("datasets/time_series/chattahoochee.csv")
