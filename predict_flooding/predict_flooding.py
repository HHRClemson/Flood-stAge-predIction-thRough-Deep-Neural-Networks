import numpy as np
import pandas as pd
from pprint import pprint

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model

from predict_flooding.models import *
from predict_flooding.window_generator import SlidingWindowGenerator

INPUT_WIDTH = 100
FUTURE_PREDICTIONS = 36
EPOCHS = 50
PATIENCE = max(5, EPOCHS // 5)


def _r_square(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2


def _run_model(model: Model, window: SlidingWindowGenerator, visualize, path,
               num_epochs=EPOCHS, patience=PATIENCE, fit_model=True):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=patience, mode="min")

    model.compile(loss=tf.losses.MeanSquaredError(name="MSE"),
                  metrics=[tf.metrics.MeanAbsoluteError(name="MAE"),
                           _r_square,
                           tf.metrics.RootMeanSquaredError(name="RMSE"),
                           tf.metrics.MeanAbsolutePercentageError(name="MAPE")],
                  optimizer=tf.optimizers.Adam())

    history = model.fit(window.train_dataset, epochs=num_epochs,
                        callbacks=[early_stopping]) if fit_model else None
    performance = model.evaluate(window.test_dataset)

    if visualize:
        window.plot(model, path)

    return history, performance


def _plot_performance(performances, path):
    x = np.arange(len(performances))

    # sort the models by metric performance descending
    results = sorted(performances.items(), key=lambda x: x[1][1], reverse=True)
    model_names = [m[0].upper() for m in results]

    mae = [m[1][1] for m in results]
    r2 = [m[1][2] for m in results]
    rmse = [m[1][3] for m in results]
    mape = [m[1][4] for m in results]

    metrics = [("Mean Absolute Error", mae),
               ("R Squared Error", r2),
               ("Root Mean Square Error", rmse),
               ("Mean Absolute Percentage Error", mape)]

    for metric in metrics:
        plt.bar(x, metric[1], label=metric[0])
        plt.xticks(ticks=x, labels=model_names, rotation=45)
        plt.ylabel(metric[0])
        plt.legend()

        plt.savefig(path + "{}-metric.png".format(metric[0].replace(" ", "-")))
        plt.show()
        plt.close()


def train_and_predict(path, df=None,
                      input_width=INPUT_WIDTH, future_predictions=FUTURE_PREDICTIONS,
                      visualize=True, prefix=""):
    if df is None:
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
    #train_df = (train_df - train_mean) / train_std
    #test_df = (test_df - train_mean) / train_std

    window: SlidingWindowGenerator = SlidingWindowGenerator(
        input_width=input_width, label_width=future_predictions, shift=future_predictions,
        train_df=train_df, test_df=test_df,
        label_columns=["height"])

    models = [baseline.Baseline(future_predictions),
              dense.Dense(future_predictions),
              cnn.CNN(future_predictions),
              lstm.LSTM(future_predictions)]

    plot_paths = "flooding_results/" + prefix

    performances = {}

    print("\n\nSTART TRAINING WITH input_width={0}, label_width={1}".format(
        input_width, future_predictions))

    for model in models:
        name = model.name
        print("\nSTART {} TRAINING:".format(name.upper()))
        _, performance = _run_model(model, window, visualize, plot_paths,
                                    fit_model=False if name == "Baseline" else True)

        print("EVALUATED {}:".format(name.upper()), performance)
        performances[name] = performance

    print("\n\nPERFORMANCES: [MSE, MAE, R2, RMSE, MAPE]")
    pprint(performances)

    if visualize:
        _plot_performance(performances, plot_paths)


if __name__ == "__main__":
    path = "datasets/time_series/chattahoochee.csv"

    df = pd.read_csv(path)
    df = df.drop("time", axis=1)
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["perception"] = pd.to_numeric(df["perception"], errors="coerce")

    windows = [(100, 3 * 4), (100, 6 * 4), (100, 9 * 4), (100, 12 * 4)]
    for i, window in enumerate(windows):
        train_and_predict("", df=df, input_width=window[0], future_predictions=window[1],
                          visualize=True, prefix=str(i) + "/")
