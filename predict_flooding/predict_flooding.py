import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models, Model, layers

import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    train_and_predict("../datasets/time_series/chattahoochee.csv")
