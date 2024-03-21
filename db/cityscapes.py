import sys
import json
import os
import numpy as np
import pickle
import cv2
import imgaug.augmenters as iaa
import random
import albumentations as A
import torch

from tabulate import tabulate
from torchvision import transforms
from torchvision.transforms import ToTensor
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from glob import glob
from natsort import natsorted
from pprint import pprint
from PIL import Image
from typing import Callable

from db.detection import DETECTION
from config import system_configs
from db.utils.lane import LaneEval
from db.utils.metric import eval_json
from db.seg_utils import get_label_info, to_one_hot


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

Transform = Callable[[np.ndarray, np.ndarray], dict]

class MultiAug(Transform):
    """Augmentation by HengChi."""
    def __init__(self,
                 width: int,
                 height: int,
                 brightness: tuple[float, float],
                 contrast: tuple[float, float],
                 ) -> None:
        # scale = random.choice([0.75, 1, 1.25, 1.5, 1.75, 2])
        scale = random.choice([0.75, 1, 1.25, 1.5])
        s_width = int(1280 * scale)
        s_height = int(720 * scale)
        self.transform = A.Compose(
            [
                A.Resize(width=s_width, height=s_height),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit = 0, scale_limit = 0, rotate_limit = 30),
                A.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=0, hue=0, p=0.5
                ),
                A.RandomCrop(width=width, height=height),
            ]
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        return self.transform(image=image, mask=mask)

class CITYSCAPES(DETECTION):
    def __init__(self, db_config: dict, split: str, csv_path: str):
        super(CITYSCAPES, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir   = system_configs.cache_dir

        self.input_h, self.input_w = db_config['input_size']
        self.root = os.path.join(data_dir, 'Cityscapes')

        self._split = split
        self._image_dir = os.path.join(self.root, self._split)
        self._label_dir = os.path.join(self.root, self._split + '_labels')

        self.img_w, self.img_h = 2048, 1024  # cityscapes original image resolution
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        
        self._image_list = natsorted(glob(os.path.join(self._image_dir, '*.png')))
        self._label_list = natsorted(glob(os.path.join(self._label_dir, '*.png')))
        for img, label in zip(self._image_list, self._label_list):
            assert img.split('/')[-1].split('.')[0] == label.split('/')[-1].split('.')[0].split('_gtFine_color')[0]
        
        self._label_info = get_label_info(csv_path)
        self._data = "cityscapes"

        self._db_inds = np.arange(len(self._image_list))

    def get_list(self):
        return self._image_list, self._label_list

    def get_path(self, idx):
        return self._image_list[idx], self._label_list[idx]

    def _get_img_heigth(self, path):
        return 1024

    def _get_img_width(self, path):
        return 2048

    def __getitem__(self, idx, transform=True):

        image = Image.open(self._image_list[idx]).convert('RGB')
        label = Image.open(self._label_list[idx]).convert('RGB')

        if transform:
            if self._split == 'train':
                train_transform = MultiAug(width=self.input_w, 
                                           height=self.input_h, 
                                           brightness=(0.5, 1.5), 
                                           contrast=(0.9, 1.1))
                augmented_pair = train_transform(image=np.array(image), mask=np.array(label))
                image = Image.fromarray(augmented_pair['image'])
                label = Image.fromarray(augmented_pair['mask'])

            elif self._split == 'val':
                image = transforms.Resize((self.input_h, self.input_w), Image.BILINEAR)(image)
                label = transforms.Resize((self.input_h, self.input_w), Image.NEAREST)(label)

        image = np.array(image)
        label = np.array(label)

        image = Image.fromarray(image)
        image = self.to_tensor(image)

        label = to_one_hot(label, self._label_info).astype(np.uint8)
        label = np.transpose(label, [2, 0, 1]).astype(np.float32)
        label = torch.from_numpy(label)

        return (image, label, idx)
    
    def __len__(self):
        return len(self._image_list)
            













