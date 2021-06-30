import numpy as np
import json
import cv2
import os

import tensorflow as tf
from tensorflow.keras import layers, Model, metrics

import matplotlib.pyplot as plt

from generate_data.gan import DCGAN
import generate_data.segmentation as segmentation

IMG_WIDTH = 512
IMG_HEIGHT = 512


def _load_dataset(path):
    pass


def _create_segmentations(path):
    images = []
    for img_name in os.listdir(path):
        if img_name == "labels.csv":
            continue

        img = cv2.imread(path + img_name)
        img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255
        images.append(img)

    return segmentation.predict_on_learned_model(images)


def _train_gan():
    segmentations = _create_segmentations("../datasets/new_data/Shamrock/")
    gan: Model = DCGAN()
    history = gan.fit(
        segmentations,
        epochs=50
    )
    gan.save("saved_models/gan")


if __name__ == "__main__":
    pass
