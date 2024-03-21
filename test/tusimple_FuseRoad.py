import os
import torch
import cv2
import json
import time
import numpy as np
import torch.nn.functional as F
import pandas as pd
import random
import albumentations as A

from torch.autograd import Variable
from PIL import Image

from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config import system_configs

from utils import crop_image, normalize_

from sample.vis import *

from typing import Callable

from engine.mmseg_metrics import MetricType, MMSegMetric
from engine.sliding_window import slide_inference, slide_inference_rescale

Transform = Callable[[np.ndarray, np.ndarray], dict]
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PLUM = (255, 187, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)
CORAL = (86, 114, 255)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)


GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [CORAL, GREEN, DARK_GREEN, PLUM, CHOCOLATE, PEACHPUFF, STATEGRAY]

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

class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        out_logits = out_logits[0].unsqueeze(0)
        out_curves = out_curves[0].unsqueeze(0)
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)

        return results

def kp_detection(lane_db, seg_db, nnet, result_dir, debug=False, evaluator=None, repeat=1,
                 isEncAttn=False, isDecAttn=False):
    # lane_shape
    model_mode = 'lane_shape'
    if lane_db.split != "train":
        db_inds = lane_db.db_inds if debug else lane_db.db_inds
        # db_inds = lane_db.db_inds[:50]
    else:
        db_inds = lane_db.db_inds[:100] if debug else lane_db.db_inds
    num_images = db_inds.size

    multi_scales = lane_db.configs["test_scales"]

    input_size  = lane_db.configs["input_size"]  # [h w]

    postprocessors = {'curves': PostProcess()}


    for ind in tqdm(range(0, num_images), ncols=67, desc="locating kps"):
        db_ind        = db_inds[ind]
        # image_id      = db.image_ids(db_ind)
        image_file    = lane_db.image_file(db_ind)
        image         = cv2.imread(image_file)
        image        = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_img = image.copy()

        height, width = image.shape[0:2]
        # item  = db.detections(db_ind) # all in the raw coordinate

        for scale in multi_scales:
            images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
            masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
            orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
            pad_image     = image.copy()
            pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
            resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
            resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
            masks[0][0]   = resized_mask.squeeze()
            resized_image = resized_image / 255.
            normalize_(resized_image, lane_db.mean, lane_db.std)
            resized_image = resized_image.transpose(2, 0, 1)
            images[0]     = resized_image
            images        = torch.from_numpy(images).cuda(non_blocking=True)
            masks         = torch.from_numpy(masks).cuda(non_blocking=True)

            # seeking better FPS performance
            images = images.repeat(repeat, 1, 1, 1).cuda(non_blocking=True)
            masks  = masks.repeat(repeat, 1, 1, 1).cuda(non_blocking=True)

            # below codes are used for drawing attention maps
            conv_features, enc_attn_weights, dec_attn_weights = [], [], []
            if isDecAttn or isEncAttn:
                hooks = [
                    nnet.model.module.layer4[-1].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)),
                    nnet.model.module.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])),
                    nnet.model.module.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1]))
                ]

            torch.cuda.synchronize(0)  # 0 is the GPU id
            t0            = time.time()
            outputs, weights, _ = nnet.test(model_mode, system_configs.max_iter, [images, masks])
            torch.cuda.synchronize(0)  # 0 is the GPU id
            t             = time.time() - t0

            # below codes are used for drawing attention maps
            if isDecAttn or isEncAttn:
                for hook in hooks:
                    hook.remove()
                conv_features = conv_features[0]
                enc_attn_weights = enc_attn_weights[0]
                dec_attn_weights = dec_attn_weights[0]

            results = postprocessors['curves'](outputs, orig_target_sizes)

            if evaluator is not None:
                evaluator.add_prediction(ind, results.cpu().numpy(), t / repeat)

        if debug:
            img_lst = image_file.split('/')
            lane_debug_dir = os.path.join(result_dir, "lane_debug")
            if not os.path.exists(lane_debug_dir):
                os.makedirs(lane_debug_dir)

            # # Draw dec attn
            if isDecAttn:
                h, w = conv_features.shape[-2:]
                keep = results[0, :, 0].cpu() == 1.
                fig, axs = plt.subplots(ncols=keep.nonzero().shape[0] + 1, nrows=2, figsize=(44, 14))
                # print(keep.nonzero().shape[0], image_file)
                # colors = COLORS * 100
                for idx, ax_i in zip(keep.nonzero(), axs.T):
                    ax = ax_i[0]
                    ax.imshow(dec_attn_weights[0, idx].view(h, w).cpu())
                    ax.axis('off')
                    ax.set_title('query id: [{}]'.format(idx))
                    ax = ax_i[1]
                    preds = lane_db.draw_annotation(ind, pred=results[0][idx].cpu().numpy(), cls_pred=None, img=raw_img)
                    ax.imshow(preds)
                    ax.axis('off')
                fig.tight_layout()
                img_path = os.path.join(lane_debug_dir, 'decAttn_{}_{}_{}.jpg'.format(
                    img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
                plt.savefig(img_path)
                plt.close(fig)

            # # Draw enc attn
            if isEncAttn:
                img_dir = os.path.join(lane_debug_dir, '{}_{}_{}'.format(
                    img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                f_map = conv_features
                shape = f_map.shape[-2:]
                image_height, image_width, _ = raw_img.shape
                sattn = enc_attn_weights[0].reshape(shape + shape).cpu()
                _, label, _ = lane_db.__getitem__(ind)
                for i, lane in enumerate(label):
                    if lane[0] == 0:  # Skip invalid lanes
                        continue
                    lane = lane[3:]  # remove conf, upper and lower positions
                    xs = lane[:len(lane) // 2]
                    ys = lane[len(lane) // 2:]
                    ys = ys[xs >= 0]
                    xs = xs[xs >= 0]
                    # norm_idxs = zip(ys, xs)
                    idxs      = np.stack([ys * image_height, xs * image_width], axis=-1)
                    attn_idxs = np.stack([ys * shape[0], xs * shape[1]], axis=-1)

                    for idx_o, idx, num in zip(idxs, attn_idxs, range(xs.shape[0])):
                        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 14))
                        ax_i = axs.T
                        ax = ax_i[0]
                        ax.imshow(sattn[..., int(idx[0]), int(idx[1])], cmap='cividis', interpolation='nearest')
                        ax.axis('off')
                        ax.set_title('{}'.format(idx_o.astype(int)))
                        ax = ax_i[1]
                        ax.imshow(raw_img)
                        ax.add_patch(plt.Circle((int(idx_o[1]), int(idx_o[0])), color='r', radius=16))
                        ax.axis('off')
                        fig.tight_layout()

                        img_path = os.path.join(img_dir, 'encAttn_lane{}_{}_{}.jpg'.format(
                            i, num, idx_o.astype(int)))
                        plt.savefig(img_path)
                        plt.close(fig)

            if not isEncAttn and not isDecAttn:
                preds = lane_db.draw_annotation(ind, pred=results[0].cpu().numpy(), cls_pred=None, img=raw_img)
                cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_'
                                         + img_lst[-2] + '_'
                                         + os.path.basename(image_file[:-4]) + '.jpg'), preds)
    if not debug:
        exp_name = 'tusimple'
        evaluator.exp_name = exp_name
        eval_str, _ = evaluator.eval(label='{}'.format(os.path.basename(exp_name)))
        print(eval_str)
                
    # segmenation evaluation
    model_mode = 'seg'
    if seg_db.split != "train":
        seg_db_inds = seg_db.db_inds if debug else seg_db.db_inds
    else:
        seg_db_inds = seg_db.db_inds[:100] if debug else seg_db.db_inds

    input_size   = seg_db.configs["input_size"]
    if system_configs.use_slide_window:
        seg_input_size   = system_configs.test_seg_input_size
    else:
        seg_input_size   = seg_db.configs["input_size"]

    seg_num_images  = seg_db.db_inds.size
    seg_multi_scales = seg_db.configs["test_scales"]
    seg_label_info = get_label_info(system_configs.seg_csv_path)

    seg_class_names = list()
    for i in range(len(seg_label_info)):
        if seg_label_info[list(seg_label_info.keys())[i]][3] == 1:
            seg_class_names.append(list(seg_label_info.keys())[i])
    # print(f'seg_class_names: {seg_class_names}')


    metric_computer = MMSegMetric(
        num_classes=system_configs.seg_n_class,
        # ignore_index=system_configs.seg_ignore_classes,
        metrics=[MetricType.IOU, MetricType.FSCORE],
        nan_to_num=1,
    )
    # print(f'image_list: {seg_db.get_list()}')
    for seg_ind in tqdm(range(0, seg_num_images), ncols=67, desc="locating seg"):
        seg_images = np.zeros((1, 3, seg_input_size[0], seg_input_size[1]), dtype=np.float32)
        masks = np.ones((1, 1, seg_input_size[0], seg_input_size[1]), dtype=np.float32)
        seg_labels = np.zeros((1, system_configs.seg_n_class, seg_input_size[0], seg_input_size[1]), dtype=np.float32)

        seg_db_ind       = seg_db_inds[seg_ind]
        seg_image_path, seg_label_path = seg_db.get_path(seg_db_ind)

        seg_image = Image.open(seg_image_path).convert('RGB')
        seg_label = Image.open(seg_label_path).convert('RGB')

        width, height = seg_image.size
        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)

        seg_image = transforms.Resize((seg_input_size[0], seg_input_size[1]))(seg_image)
        seg_label = transforms.Resize((seg_input_size[0], seg_input_size[1]))(seg_label)
        resized_mask = cv2.resize(pad_mask, (seg_input_size[1], seg_input_size[0]))

        seg_image = np.array(seg_image)
        seg_label = np.array(seg_label)

        seg_image = seg_image.astype(np.float32)

        seg_label = to_one_hot(seg_label, seg_label_info).astype(np.uint8)
        seg_label = np.transpose(seg_label, [2, 0, 1]).astype(np.float32)

        seg_image = seg_image / 255.
        normalize_(seg_image, seg_db.mean, seg_db.std)

        seg_images[0] = seg_image.transpose((2, 0, 1))
        seg_labels[0] = seg_label
        masks[0][0]   = resized_mask.squeeze()
        
        seg_images = torch.from_numpy(seg_images).cuda(non_blocking=True)
        seg_labels = torch.from_numpy(seg_labels).cuda(non_blocking=True)
        masks = torch.from_numpy(masks).cuda(non_blocking=True)

        # seeking better FPS performance
        seg_images = seg_images.repeat(repeat, 1, 1, 1).cuda(non_blocking=True)
        seg_labels = seg_labels.repeat(repeat, 1, 1, 1).cuda(non_blocking=True)
        masks = masks.repeat(repeat, 1, 1, 1).cuda(non_blocking=True)

        torch.cuda.synchronize(0)  # 0 is the GPU id
        t0            = time.time()
        if system_configs.use_slide_window:
            pred = slide_inference(nnet=nnet,
                                model_mode=model_mode,
                                iter=system_configs.max_iter,
                                img=seg_images,
                                pos_mask=masks,
                                crop_size=(input_size[0], input_size[1]),
                                stride=(128, 128),
                                num_classes=system_configs.seg_n_class)
        else:
            pred = nnet.test(model_mode, system_configs.max_iter, [seg_images, masks])
            
        torch.cuda.synchronize(0)
        t             = time.time() - t0

        one_hot_pred = pred.softmax(1).argmax(1)
        one_hot_label = seg_labels.argmax(1)
        metric_computer.compute_and_accum(one_hot_pred, one_hot_label)
    
    metrics_result = metric_computer.get_and_clear()
    metric_computer.show_result(metrics_result, seg_class_names)
    mIoU = metrics_result['IoU'].mean()

    # segmenation evaluation finised
    
    return 0

def testing(lane_db, seg_db, nnet, result_dir, debug=False, evaluator=None, repeat=1,
            debugEnc=False, debugDec=False):
    return globals()[system_configs.sampling_function](lane_db, seg_db, nnet, result_dir, debug=debug, evaluator=evaluator,
                                                       repeat=repeat, isEncAttn=debugEnc, isDecAttn=debugDec)