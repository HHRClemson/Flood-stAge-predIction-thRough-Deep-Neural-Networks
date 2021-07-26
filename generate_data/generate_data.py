import numpy as np
import cv2
import os

import tensorflow as tf
from tensorflow.keras import Model

import matplotlib.pyplot as plt

from generate_data.gan import ReGAN
import generate_data.segmentation as segmentation

IMG_WIDTH = 256
IMG_HEIGHT = 256
NOISE_DIM = 2048


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
        #img = tf.image.rgb_to_grayscale(img)
        images.append(img)
    return np.array(images)


def _train_gan():
    segmentations = _load_segmentation_data("datasets/new_data/")
    print(len(segmentations))
    gan: Model = ReGAN(channels=3,
                       noise_dim=NOISE_DIM,
                       batch_size=32)
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy"
    )

    generated_imgs = gan.train(
        segmentations,
        epochs=10,
        batch_size=16,
    )

    generated_imgs.append(gan(tf.random.normal([16, NOISE_DIM])))

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
    _train_gan()
    # _create_segmentations("datasets/new_data/Shamrock/")
