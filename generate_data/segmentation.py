import numpy as np
import json
import cv2
import os

import tensorflow as tf
from tensorflow.keras import layers, Model, metrics

import matplotlib.pyplot as plt

IMG_WIDTH = 512
IMG_HEIGHT = 512


def _get_mask(labels, shape):
    mask = np.zeros((*shape, 1))
    for label in labels:
        coords = np.array([[x, y] for x, y in zip(label[0], label[1])])
        cv2.fillPoly(mask, pts=[coords], color=255)
    return mask


def _load_dataset(path):
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

        img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255
        mask = tf.cast(tf.image.resize_with_pad(mask, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255

        # the size of the validation set is 20% of the original dataset
        if i % 5 == 0:
            x_val.append(img)
            y_val.append(mask)
        else:
            x_train.append(img)
            y_train.append(mask)
    return x_train, y_train, x_val, y_val


def _load_kaggle_dataset(path):
    """
    Train on the Kaggle Water Segmentation dataset:
    https://www.kaggle.com/gvclsu/water-segmentation-dataset
    """
    images_path = path + "images/"
    truth_path = path + "truth/"
    dirs = os.listdir(path + "truth")

    x_train, y_train = [], []
    x_val, y_val = [], []

    for dir in dirs:
        print("CURR DIR: ", dir)
        
        images_dir = images_path + dir + "/"
        truth_dir = truth_path + dir + "/"

        for img_name in sorted(os.listdir(images_dir)):
            img_path = images_dir + img_name
            img = cv2.imread(img_path)
            img = tf.cast(tf.image.resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255

            x_train.append(img)

        for img_name in sorted(os.listdir(truth_dir)):
            img_path = truth_dir + img_name
            mask = cv2.imread(img_path)
            mask = tf.cast(tf.image.resize_with_pad(mask, IMG_WIDTH, IMG_HEIGHT), np.uint8) / 255
            mask = tf.image.rgb_to_grayscale(mask)

            y_train.append(mask)
    
    return x_train, y_train 


def _create_model() -> Model:
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name="input_image")

    encoder = tf.keras.applications.MobileNetV2(input_tensor=inputs,
                                                include_top=False,
                                                weights="imagenet")
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    f = [16, 32, 48, 64, 128]
    x = encoder_output
    for i in range(1, len(skip_connection_names) + 1):
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
        loss=_dice_loss,
        metrics=[_dice_coef, metrics.Recall(), metrics.Precision()]
    )

    return unet


def _dice_coef(y_true, y_pred, smooth=1e-15):
    y_true = layers.Flatten()(y_true)
    y_pred = layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def _dice_loss(y_true, y_pred):
    return 1.0 - _dice_coef(y_true, y_pred)


def _display(images, name=None):
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
        plt.savefig("results/" + name, format="svg", dpi=1200)
    plt.close()


def train_and_predict():
    """Train the U-Net model for segmentation and save the model"""
    x_train, y_train, = _load_kaggle_dataset("../datasets/kaggle/")
    x_train_webcam, y_train_webcam, x_val, y_val = _load_dataset("../datasets/segmentation/")

    x_train.extend(x_train_webcam)
    y_train.extend(y_train_webcam)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    model: Model = _create_model()
    model.summary()
      
    history = model.fit(
        x_train, y_train,
        epochs=150,
        batch_size=20,
        #validation_data=(x_val, y_val)
    )
    model.save("saved_models/u-net")        

    predictions = model.predict(x_val)

    results = []
    for i in range(len(x_val)):
        res = [x_val[i], y_val[i], predictions[i]]
        results.append(res)
        _display([res], name=str(i) + ".svg")
    _display(results, name="segmentation-results.svg")


def predict_on_learned_model(images):
    model: Model = tf.keras.models.load_model("saved_models/u-net")
    predictions = model.predict(images)
    return predictions
