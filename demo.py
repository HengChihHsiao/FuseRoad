#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import torch.nn.functional as F
import pprint
import argparse
import importlib
import sys
import numpy as np
import cv2
import time
import tqdm

from glob import glob
from natsort import natsorted
from torch import nn
from torch.autograd import Variable
from config import system_configs
from nnet.FuseRoad_factory import NetworkFactory
from db.datasets import datasets
from db.utils.FuseRoad_evaluator import Evaluator
from engine.mmseg_metrics import MetricType, MMSegMetric
from engine.sliding_window import slide_inference, slide_inference_rescale
from utils import crop_image, normalize_
from db.seg_utils import get_label_info, to_one_hot, reverse_one_hot, colour_code_segmentation

torch.backends.cudnn.benchmark = False

lane_db_mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
lane_db_std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)

def save_seg_pred(pred: torch.Tensor, save_path: str):
    which_stack = 0
    label_info = get_label_info(system_configs.seg_csv_path)

    pred = reverse_one_hot(pred[which_stack]).cpu()
    pred = colour_code_segmentation(np.array(pred), label_info)
    cv2.imwrite(save_path, cv2.cvtColor(np.uint8(pred), cv2.COLOR_RGB2BGR))


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)

        return results

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--model_name", help="model name", default="LSTR", type=str)
    parser.add_argument("--image_root", dest="image_root", default=None, type=str)
    parser.add_argument("--save_root", dest="save_root", default=None, type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def demo_on_image(img_path, save_path, nnet, evaluator=None, ind=0):
    model_mode = 'lane_shape'
    
    input_size  = system_configs.input_size # [h w]
    postprocessors = {'bbox': PostProcess()}

    image_file    = img_path
    image         = cv2.imread(image_file)
    height, width = image.shape[0:2]

    images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
    masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
    orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
    pad_image     = image.copy()
    pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
    resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
    resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
    masks[0][0]   = resized_mask.squeeze()
    resized_image = resized_image / 255.
    normalize_(resized_image, lane_db_mean, lane_db_std)
    resized_image = resized_image.transpose(2, 0, 1)
    images[0]     = resized_image
    images        = torch.from_numpy(images).cuda(non_blocking=True)
    masks         = torch.from_numpy(masks).cuda(non_blocking=True)
    torch.cuda.synchronize(0)  # 0 is the GPU id
    t0            = time.time()
    lane_outputs, weights, seg_outputs = nnet.test(model_mode, system_configs.max_iter, [images, masks])
    torch.cuda.synchronize(0)  # 0 is the GPU id
    t             = time.time() - t0
    results = postprocessors['bbox'](lane_outputs, orig_target_sizes)
    if evaluator is not None:
        evaluator.add_prediction(ind, results.cpu().numpy(), t)

    pred = results[0].cpu().numpy()
    img  = pad_image
    img_h, img_w, _ = img.shape
    pred = pred[pred[:, 0].astype(int) == 1]
    overlay = img.copy()
    color = (0, 255, 0)
    for i, lane in enumerate(pred):
        lane = lane[1:]  # remove conf
        lower, upper = lane[0], lane[1]
        lane = lane[2:]  # remove upper, lower positions

        # generate points from the polynomial
        ys = np.linspace(lower, upper, num=100)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * img_h).astype(int)
        points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                            lane[5]) * img_w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

        # draw lane with a polyline on the overlay
        for current_point, next_point in zip(points[:-1], points[1:]):
            overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=15)

        # draw lane ID
        if len(points) > 0:
            cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=color,
                        thickness=3)
    # Add lanes overlay
    w = 0.6
    img = ((1. - w) * img + w * overlay).astype(np.uint8)

    cv2.imwrite(save_path + '_lane_pred.png', img)

    # seg_pred
    save_seg_path = save_path[:-4] + '_seg_pred.png'
    save_seg_pred(seg_outputs, save_seg_path)
    

if __name__ == "__main__":
    args = parse_args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.model_name
    system_configs.update_config(configs["system"])

    lane_train_split = system_configs.train_split
    lane_val_split   = system_configs.val_split
    lane_test_split  = system_configs.test_split
    testiter = args.testiter
    image_root = args.image_root

    save_dir = system_configs.result_dir if args.save_root is None else args.save_root
    make_dirs([save_dir])
    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))

    print("building neural network...")
    nnet = NetworkFactory()

    print("loading parameters...")
    nnet.load_params(test_iter)
    nnet.cuda()
    nnet.eval_mode()

    if image_root == None:
        raise ValueError('--image_root is not defined!')
    print("processing [images]...")
    test_file = "test.images_FuseRoad"
    image_testing = importlib.import_module(test_file).testing
    
    image_list = natsorted(glob(os.path.join(image_root, "*.*")))
    for ind, img_path in enumerate(image_list):
        print(f"Processing image {ind+1}/{len(image_list)}: {os.path.basename(img_path)}")
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        demo_on_image(img_path, save_path, nnet, evaluator=None, ind=ind)
    
    
    
    