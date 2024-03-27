import cv2
import numpy as np
import torch
import torchvision
import pandas as pd
import random
import albumentations as A

from torchvision import transforms
from PIL import Image
from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from typing import Callable

Transform = Callable[[np.ndarray, np.ndarray], dict]

def get_label_info(csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        use_flag = row['use_flag']
        label[label_name] = [int(r), int(g), int(b), use_flag]
    return label

def to_one_hot(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = []
    class_index = 0
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        use_flag = label_info[info][3]
        if use_flag == 1:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)

            semantic_map.append(class_map)
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)

    return semantic_map

class MultiAug(Transform):
    """Augmentation by HengChi."""
    def __init__(self,
                 width: int,
                 height: int,
                 brightness: tuple[float, float],
                 contrast: tuple[float, float],
                 ) -> None:
        # scale = random.choice([0.7, 0.8])
        scale = random.choice([1, 1.2, 1.4, 1.6, 1.8, 2.0])
        s_width = int(system_configs.input_size[1] * scale)
        s_height = int(system_configs.input_size[0] * scale)
        self.transform = A.Compose(
            [
                A.Resize(width=s_width, height=s_height),
                A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(shift_limit = 0, scale_limit = 0, rotate_limit = 10),
                A.ShiftScaleRotate(shift_limit = 0, scale_limit = 0, rotate_limit = 30),
                A.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=0, hue=0, p=0.5
                ),
                A.RandomCrop(width=width, height=height),
            ]
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        return self.transform(image=image, mask=mask)

def kp_detection(db, k_ind):
    data_rng     = system_configs.data_rng
    batch_size   = system_configs.batch_size
    input_size   = db.configs["input_size"]
    images   = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32) # b, 3, H, W
    labels   = np.zeros((batch_size, system_configs.seg_n_class, input_size[0], input_size[1]), dtype=np.float32) # b, 3, H, W
    # masks    = np.zeros((batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)  # b, 1, H, W
    # seg_annos = np.zeros((batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)  # b, 1, H, W

    label_info = db._label_info
    to_tensor = db.to_tensor

    db_size = db.db_inds.size # 2975 | 500

    for b_ind in range(batch_size):

        if k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading ground truth
        # item  = db.detections(db_ind) # all in the raw coordinate
        image_path, label_path = db.get_path(db_ind)
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        if db._split == 'train':
            train_transform = MultiAug(width=input_size[1],
                                        height=input_size[0],
                                        brightness=(0.5, 1.5), 
                                        contrast=(0.9, 1.1))
            augmented_pair = train_transform(image=np.array(image), mask=np.array(label))
            image = Image.fromarray(augmented_pair['image'])
            label = Image.fromarray(augmented_pair['mask'])
        elif db._split== 'val':
            image = transforms.Resize((input_size[0], input_size[1]), Image.BILINEAR)(image)
            label = transforms.Resize((input_size[0], input_size[1]), Image.NEAREST)(label)

        image = np.array(image)
        label = np.array(label)

        image = image.astype(np.float32)

        label = to_one_hot(label, label_info).astype(np.uint8)
        label = np.transpose(label, [2, 0, 1]).astype(np.float32)

        image = (image / 255.).astype(np.float32)
        normalize_(image, db.mean, db.std)

        images[b_ind] = image.transpose((2, 0, 1))
        labels[b_ind] = label

    images   = torch.from_numpy(images)
    labels    = torch.from_numpy(labels)

    return {
            "xs": [images, labels],
            "ys": [labels],
           }, k_ind


def sample_data(db, k_ind):
    return globals()[system_configs.sampling_function](db, k_ind)


