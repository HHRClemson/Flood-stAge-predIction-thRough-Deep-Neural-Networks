import numpy as np
import json
import cv2
import os

import tensorflow as tf
from tensorflow.keras import layers, Model, metrics

import matplotlib.pyplot as plt


def _get_mask(labels, shape):
    mask = np.zeros((*shape, 1))
    for label in labels:
        coords = np.array([[x, y] for x, y in zip(label[0], label[1])])
        cv2.fillPoly(mask, pts=[coords], color=255)
    return mask


def _load_dataset(path, img_width, img_height):
    segmentations = json.load(open(path + "segmentation.json"))
    keys = list(segmentations.keys())

    x_train, y_train = [], []
    x_val, y_val = [], []

    for i, img_name in enumerate(keys):
        # the img key in the json labels always has a random number at the end of the name
        img_path = (path + img_name).split(".png")[0] + ".png"
        img = cv2.imread(img_path)

        segment = segmentations[img_name]
        regions = segment["regions"]
        labels = []
        for region in regions:
            row = regions[region]["shape_attributes"]
            labels.append((row["all_points_x"], row["all_points_y"]))

        mask = _get_mask(labels, img.shape[:2])

        # the size of the validation set is 20% of the original dataset
        if i % 5 == 0:
            x_val.append(tf.cast(
                tf.image.resize_with_pad(img, img_width, img_height), np.uint8) / 255)
            y_val.append(tf.cast(
                tf.image.resize_with_pad(mask, img_width, img_height), np.uint8) / 255)
        else:
            x_train.append(tf.cast(
                tf.image.resize_with_pad(img, img_width, img_height), np.uint8) / 255)
            y_train.append(tf.cast(
                tf.image.resize_with_pad(mask, img_width, img_height), np.uint8) / 255)

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val)


def _load_kaggle_dataset(path, img_width, img_height):
    """
    Train on the Kaggle Water Segmentation dataset:
    https://www.kaggle.com/gvclsu/water-segmentation-dataset
    """
    training_path = path + "images/"
    truth_path = path + "truth/"
    dirs = os.listdir(path + "truth")

    x_train = []
    y_train = []

    for dir in dirs:
        training_dir = training_path + dir + "/"
        truth_dir = truth_path + dir + "/"

        for img_name in os.listdir(training_dir):
            img_path = training_dir + img_name
            img = cv2.imread(img_path)
            x_train.append(tf.cast(
                tf.image.resize_with_pad(img, img_width, img_height), np.uint8) / 255)

        for img_name in os.listdir(truth_dir):
            img_path = truth_dir + img_name
            img = cv2.imread(img_path)
            y_train.append(tf.cast(
                tf.image.resize_with_pad(img, img_width, img_height), np.uint8) / 255)

    return np.asarray(x_train), np.asarray(y_train)


def create_model(img_width, img_height) -> Model:
    inputs = layers.Input(shape=(img_width, img_height, 3), name="input_image")

    encoder = tf.keras.applications.MobileNetV2(input_tensor=inputs,
                                                include_top=False,
                                                weights="imagenet")
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    f = [16, 32, 48, 64, 128]
    x = encoder_output
    for i in range(1, len(skip_connection_names) + 1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, x_skip])

        x = layers.Conv2D(f[-i], (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(f[-i], (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    x = layers.Conv2D(1, (1, 1), padding="same")(x)
    x = layers.Activation("sigmoid")(x)

    unet = Model(inputs, x, name="U-Net")
    unet.compile(
        optimizer=tf.keras.optimizers.Nadam(1e-4),
        loss=dice_loss,
        metrics=[dice_coef, metrics.Recall(), metrics.Precision()]
    )

    return unet


def dice_coef(y_true, y_pred, smooth=1e-15):
    y_true = layers.Flatten()(y_true)
    y_pred = layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def display(images, name=None):
    fig = plt.figure()
    columns = len(images[0])
    rows = len(images)

    curr_img = 1
    for data in images:
        for img in data:
            fig.add_subplot(rows, columns, curr_img)
            plt.imshow(img)
            curr_img += 1
            plt.axis("off")
    if name is not None:
        plt.savefig(name, format="svg", dpi=1200)
    plt.show()


if __name__ == "__main__":
    img_width = 512
    img_height = 512
    x_train, y_train, x_val, y_val = _load_dataset("../datasets/segmentation/",
                                                   img_width, img_height)
    x_train, y_train = _load_kaggle_dataset("../datasets/kaggle/", img_width, img_height)
    display([[x_train[0], y_train[0]]], name="test-pics.svg")

    model: Model = create_model(img_width, img_height)
    model.summary()

    history = model.fit(
        x_train, y_train,
        epochs=1,
        #validation_data=(x_val, y_val)
    )

    predictions = model.predict(x_val)

    results = []
    for i in range(len(x_val)):
        results.append([x_val[i], y_val[i], predictions[i]])
    #display(results)

