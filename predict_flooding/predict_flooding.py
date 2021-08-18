import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model

from predict_flooding.models import *
from predict_flooding.window_generator import SlidingWindowGenerator

FUTURE_PREDICTIONS = 48
EPOCHS = 50
PATIENCE = max(5, EPOCHS // 5)


def _evaluate_baseline(window: SlidingWindowGenerator):
    model: Model = baseline.Baseline(FUTURE_PREDICTIONS)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    performance = model.evaluate(window.test_dataset)

    window.plot(model)

    return performance


def _evaluate_dense(window: SlidingWindowGenerator):
    model: Model = dense.Dense(FUTURE_PREDICTIONS)
    _, performance = _run_model(model, window)
    return performance


def _evaluate_cnn(window: SlidingWindowGenerator):
    model: Model = cnn.CNN(FUTURE_PREDICTIONS)
    _, performance = _run_model(model, window)
    return performance


def _evaluate_lstm(window: SlidingWindowGenerator):
    model: Model = lstm.LSTM(FUTURE_PREDICTIONS)
    _, performance = _run_model(model, window)
    return performance


def _run_model(model: Model, window: SlidingWindowGenerator,
               num_epochs=EPOCHS, patience=PATIENCE):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=patience, mode="min")

    model.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()],
                  optimizer=tf.optimizers.Adam())
    history = model.fit(window.train_dataset, epochs=num_epochs, callbacks=[early_stopping])
    performance = model.evaluate(window.test_dataset)

    window.plot(model)

    return history, performance


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

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    print(train_df.head())
    print(test_df.head())

    window_generator: SlidingWindowGenerator = SlidingWindowGenerator(
        input_width=100, label_width=FUTURE_PREDICTIONS, shift=FUTURE_PREDICTIONS,
        train_df=train_df, test_df=test_df,
        label_columns=["height"])

    baseline_performance = _evaluate_baseline(window_generator)
    print("EVALUATED BASELINE:", baseline_performance)

    dense_performance = _evaluate_dense(window_generator)
    print("EVALUATED DENSE:", dense_performance)

    cnn_performance = _evaluate_cnn(window_generator)
    print("EVALUATED CNN:", cnn_performance)

    lstm_performance = _evaluate_lstm(window_generator)
    print("EVALUATED LSTM:", lstm_performance)

    print("\n".join([
        "Baseline performance: {}".format(baseline_performance),
        "Dense performance: {}".format(dense_performance),
        "CNN performance: {}".format(cnn_performance),
        "LSTM performance: {}".format(lstm_performance)
    ]))


if __name__ == "__main__":
    train_and_predict("datasets/time_series/chattahoochee.csv")
