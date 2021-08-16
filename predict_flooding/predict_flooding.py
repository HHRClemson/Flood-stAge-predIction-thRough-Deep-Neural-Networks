import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt

from predict_flooding.window_generator import SlidingWindowGenerator
from predict_flooding.models import *


def train_and_predict(path):
    df = pd.read_csv(path)
    train_test_split = int(len(df) * 0.8)
    train_df = df[0:train_test_split]
    test_df = df[train_test_split:]

    """Normalize the data by using the training mean and standard deviation."""
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    window_generator: SlidingWindowGenerator = SlidingWindowGenerator(
        input_width=100, label_width=1, shift=0, train_df=train_df, test_df=test_df)

    train_ds = window_generator.train_dataset
    test_ds = window_generator.test_dataset



if __name__ == "__main__":
    train_and_predict("../datasets/time_series/chattahoochee.csv")
