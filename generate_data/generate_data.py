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


def _create_segmentations(path):
    images = []
    for img_name in os.listdir(path):
        if img_name == "labels.csv" or os.path.isdir(path + img_name):
            continue

        img = cv2.imread(path + img_name)
        img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255
        images.append(img)

    images = np.array(images)
    print("Num images:", len(images))

    predictions = segmentation.predict_on_learned_model(images)

    num_digits = len(str(len(images)))
    for i, img in enumerate(predictions):
        img_name = path + "segmentations/" + str(i).zfill(num_digits) + ".png"
        # denormalize image again since we normalized all values to the interval [0, 1]
        img *= 255
        cv2.imwrite(img_name, img)


def _load_segmentation_data(path):
    images = []
    for img_name in os.listdir(path):
        img = cv2.imread(path + img_name)
        img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255
        img = tf.image.rgb_to_grayscale(img)
        images.append(img)
    return np.array(images)


def _train_gan():
    segmentations = _load_segmentation_data("datasets/new_data/Shamrock/segmentations/")
    gan: Model = DCGAN()
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy"
    )

    history = gan.train(
        segmentations,
        epochs=50
    )

    #gan.save("generate_data/saved_models/gan")


if __name__ == "__main__":
    _train_gan()
    #_create_segmentations("datasets/new_data/Shamrock/")
