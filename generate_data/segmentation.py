import numpy as np
import json
import cv2

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt


class WaterDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.segmentations = json.load(open(path + "segmentation.json"))
        self.keys = list(self.segmentations.keys())

    def __len__(self):
        return len(self.segmentations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.keys[idx]

        # img key in json always add a random number to the end of the path
        img_path = (self.path + img_name).split(".png")[0] + ".png"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))

        segment = self.segmentations[img_name]
        regions = segment["regions"]
        labels = []
        for region in regions:
            row = regions[region]["shape_attributes"]
            labels.append((row["all_points_x"], row["all_points_y"]))

        mask = self._get_mask(labels, img.shape[:2]) / 255

        if img.shape[:2] != (720, 1280):
            #img = cv2.resize(img, (720, 1280))
            #mask = cv2.resize(mask, (720, 1280))
            return self[(idx + 1) % len(self)]

        return img, mask

    @staticmethod
    def _get_mask(labels, shape):
        mask = np.zeros((*shape, 1))
        for label in labels:
            coords = np.array([[x, y] for x, y in zip(label[0], label[1])])
            cv2.fillPoly(mask, pts=[coords], color=255)
        return mask


def _create_model():
    return smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        classes=1,  # water
        activation="sigmoid"
    )


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 9))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title(), color='white')
        plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    data_path = "../datasets/segmentation/"
    dataset = WaterDataset(data_path)

    model = _create_model()

    train_epoch = smp.utils.train.TrainEpoch(
        model=model,
        loss=smp.utils.losses.DiceLoss(),
        metrics=[smp.utils.metrics.IoU(threshold=0.5)],
        optimizer=torch.optim.Adam([
            dict(params=model.parameters(), lr=0.0005)
        ]),
        verbose=True
    )

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    train_epoch.run(train_loader)
