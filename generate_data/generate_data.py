import numpy as np
import pandas as pd
import cv2
import os

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt

import generate_data.segmentation as segmentation
from generate_data.gan import ReGAN
from generate_data.cvae import CVAE

IMG_WIDTH = 256
IMG_HEIGHT = 256
LATENT_DIM = 512
ADAM_LEARNING_RATE = 1e-4


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
        if ".png" not in img_name:
            continue
        img = cv2.imread(path + img_name)
        img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255
        # img = tf.image.rgb_to_grayscale(img)
        images.append(img)
    return np.asarray(images)


def _load_all_images_no_labels(path):
    images = []
    dirs = os.listdir(path)

    for dir in dirs:
        curr_path = path + dir + "/"

        for img_name in sorted(os.listdir(curr_path)):
            # sort out label files (csv, json, ...)
            if ".png" not in img_name:
                continue

            img_path = curr_path + img_name
            img = cv2.imread(img_path)
            img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255
            images.append(img)

    return np.asarray(images)


def _train_autoencoder(path) -> Model:
    images = _load_all_images_no_labels(path)
    print("Start training the Autoencoder with {0} images:".format(len(images)))

    autoencoder: Model = CVAE()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(ADAM_LEARNING_RATE),
                        loss="binary_crossentropy")

    history = autoencoder.fit(images, epochs=1)
    #autoencoder.save("saved_models/autoencoder")

    return autoencoder


def _get_csv_key_from_filename(filename):
    # remove .png
    filename = filename[:-4]
    date, time = filename.split("_")
    hour, minutes = time.split(":")

    next_quarter = 15 * round(int(minutes) / 15)
    next_quarter = "0" + str(next_quarter) if next_quarter < 10 else str(next_quarter)
    if next_quarter == "60":
        hour = str(int(hour) + 1)
        next_quarter = "00"

    return "{d}_{h}:{m}".format(d=date, h=hour, m=next_quarter)


def _load_webcam_images_with_labels(path):
    images, labels = [], []
    files = os.listdir(path)
    heights = dict(pd.read_csv(path + "labels.csv").to_numpy())

    for f in files:
        if f == "labels.csv":
            continue

        img_path = path + f
        img = cv2.imread(img_path)
        img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255
        images.append(img)

        csv_key = _get_csv_key_from_filename(f)
        labels.append(heights[csv_key])

    return np.asarray(images), np.asarray(labels)


def _train_gan(encoder: Model, path):
    images, labels = _load_webcam_images_with_labels(path)
    print("Start training the GAN with {0} images:".format(len(images)))

    gan: Model = ReGAN(
        encoder,
        channels=3,
        noise_dim=LATENT_DIM,
        batch_size=32
    )
    gan.compile(optimizer=tf.keras.optimizers.Adam(ADAM_LEARNING_RATE),
                loss="binary_crossentropy")

    generated_imgs = gan.train(
        images, labels,
        epochs=1,
        batch_size=16,
    )
    gan.save("saved_models/GAN")

    generated_imgs.append(gan(tf.random.normal([16, LATENT_DIM])))

    for i, generated in enumerate(generated_imgs):
        fig = plt.figure(figsize=(4, 4))

        for j in range(len(generated)):
            fig.add_subplot(4, 4, j + 1)
            plt.imshow(generated[j])
            plt.axis("off")

        file_name = "gan-results/generated_images{}.svg".format(i)
        plt.savefig(file_name, format="svg", dpi=1200)
        plt.close()


if __name__ == "__main__":
    print("\n\n##START AUTOENCODER TRAINING##\n\n")
    autoencoder: Model = _train_autoencoder("datasets/webcam_images/")

    print("\n\n##START GAN TRAINING##\n\n")
    _train_gan(autoencoder, "datasets/webcam_images/Shamrock/")
