import os
import cv2
import json
import random

import albumentations as aug
import albumentations.pytorch as APT
import numpy as np
import torch

from torchvision.transforms import v2
from PIL import Image
from itertools import count
from torch.utils.data import Dataset, DataLoader

TRAIN_TRANSFORM = aug.Compose(
    [
        aug.RandomResizedCrop(
            512, 512, scale=(0.6, 1.0), p=1.0
        ),
        aug.HorizontalFlip(p=0.5),
        aug.OneOf(
            [
                aug.RandomBrightnessContrast(0.1, 0.1, p=1),
                aug.RandomGamma(p=1)
            ],
            p=0.5,
        ),
        aug.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3
        ),
        aug.CoarseDropout(
            p=0.5,
            min_holes=1,
            max_holes=6,
            min_height=512 // 8,
            max_height=512 // 4,
            min_width=512 // 8,
            max_width=512 // 4,
            fill_value=(255, 0, 0),
        ),
        aug.CoarseDropout(
            p=1.0,
            max_holes=1,
            min_height=512 // 4,
            max_height=512 // 2,
            min_width=512 // 4,
            max_width=512 // 2,
            fill_value=(255, 0, 0),
        ),
        aug.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        aug.ToFloat(),
        APT.transforms.ToTensorV2(),
    ]
)

VAL_TRANSFORM = aug.Compose(
    [
        aug.Resize(width=512, height=512),
        aug.CoarseDropout(
            p=1.0,
            max_holes=1,
            min_height=512 // 4,
            max_height=512 // 2,
            min_width=512 // 4,
            max_width=512 // 2,
            fill_value=(255, 0, 0),
        ),
        aug.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        aug.ToFloat(),
        APT.transforms.ToTensorV2(),
    ]
)

TEST_TRANSFORM = aug.Compose(
    [
        aug.Resize(width=512, height=512),
        aug.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        aug.ToFloat(),
        APT.transforms.ToTensorV2(),
    ]
)

class ImageDataset(Dataset):

    def __init__(self, img_paths, ids=None, labels=False, transform=None, **kwargs) -> None:
        super(ImageDataset, self).__init__(**kwargs)
        self.images = sorted([os.path.abspath(elem) for elem in img_paths])
        self.transform = transform

        if labels:
            assert ids is not None, "Hotel ids should be provided at training"
            self.hotel_ids = list(sorted(ids))
            self.id2label = {k:v for k, v in zip(self.hotel_ids, count())}

        self.get_func = self.train_get if labels else self.test_get

    def __len__(self):
        return len(self.images)


    def train_get(self, idx):
        img_path = self.images[idx]
        hotel_id = img_path.split(os.sep)[-2]
        label = self.id2label[hotel_id]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return {
            "img": self.transform(image=img)["image"],
            "label": label,
            "id": int(hotel_id)
        }

    def test_get(self, idx):

        img_path = self.images[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return {
            "img": self.transform(image=img)["image"]
        }

    def __getitem__(self, idx):
        return self.get_func(idx)

def build_dataloader(img_paths, ids, labels, transform, loader_cfg):

    dset = ImageDataset(img_paths=img_paths, ids=ids, labels=labels, transform=transform)
    return DataLoader(dset, **loader_cfg)

