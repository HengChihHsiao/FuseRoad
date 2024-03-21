import importlib
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from thop import clever_format, profile
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

from config import system_configs
from db.seg_utils import (DiceLoss, colour_code_segmentation, get_label_info,
                          reverse_one_hot, to_one_hot)
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)

def save_seg_pred(model_mode: str, image: torch.Tensor, pred: torch.Tensor, gt_mask, iteration: int, debug_path: str, viz_split: str):
    which_stack = 0
    label_info = get_label_info(system_configs.seg_csv_path)

    save_dir = os.path.join(debug_path, viz_split)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = 'iter_{}_{}_layer_{}'.format(iteration, model_mode, which_stack)
    save_image_path = os.path.join(save_dir, save_name + '_seg_image.png')
    save_pred_path = os.path.join(save_dir, save_name + '_seg_pred.png')
    save_gt_path = os.path.join(save_dir, save_name + '_seg_gt.png')

    pred = reverse_one_hot(pred[which_stack]).cpu()
    pred = colour_code_segmentation(np.array(pred), label_info)
    cv2.imwrite(save_pred_path, cv2.cvtColor(np.uint8(pred), cv2.COLOR_RGB2BGR))

    gt_mask = reverse_one_hot(gt_mask[which_stack]).cpu()
    gt_mask = colour_code_segmentation(np.array(gt_mask), label_info)
    cv2.imwrite(save_gt_path, cv2.cvtColor(np.uint8(gt_mask), cv2.COLOR_RGB2BGR))

    torchvision.utils.save_image(image[which_stack], save_image_path, normalize=True)

class Network(nn.Module):
    def __init__(self, model, lane_loss, seg_loss):
        super(Network, self).__init__()

        self.model = model
        self.lane_loss  = lane_loss
        self.seg_loss = seg_loss

    def forward(self, model_mode, iteration, save, viz_split, xs, ys, **kwargs):

        if model_mode == 'lane_shape':
            preds, weights, seg_preds = self.model(model_mode, iteration, *xs, **kwargs)
            if save and iteration % 100 == 0:
                save_seg_pred(model_mode, xs[0], seg_preds, ys[0], iteration, system_configs.result_dir, viz_split)
            loss  = self.lane_loss(iteration, save, viz_split, preds, ys, **kwargs)
        elif model_mode == 'seg':
            preds = self.model(model_mode, iteration, *xs, **kwargs)
            print('--------------------------seg--------------------------') if system_configs.debug else None
            print("preds.shape: {}".format(preds.shape)) if system_configs.debug else None
            print("ys.shape: {}".format(len(ys))) if system_configs.debug else None
            print("ys.shape: {}".format(ys[0].shape)) if system_configs.debug else None
            # save pred
            if save and iteration % 100 == 0:
                save_seg_pred(model_mode, xs[0], preds, ys[0], iteration, system_configs.result_dir, viz_split)
            
            loss  = self.seg_loss(preds, ys[0])
            # seg_dice_loss = self.seg_loss[1](preds, ys[0])
            # loss = 0.5 * seg_ce_loss + 0.5 * seg_dice_loss
            # loss = seg_ce_loss

        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, model_mode, iteration, *xs, **kwargs):
        # return self.module(model_mode, *xs, **kwargs)
        return self.module(model_mode, iteration, *xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, flag=False):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        print("module_file: {}".format(module_file)) # models.LSTR_CS_seg_TuSimple
        nnet_module = importlib.import_module(module_file)

        # print("-------------------dir(nnet_module):-------------------")
        # for i in dir(nnet_module):
        #     print(i)

        self.model   = DummyModule(nnet_module.model(flag=flag))
        self.lane_loss    = nnet_module.lane_loss()

        # seg_weights = torch.log(torch.FloatTensor([0.06084986, 0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
        #                                            0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
        #                                            0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
        #                                            0.00413907])).cuda()
        # seg_weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
        #                                            0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
        #                                            0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
        #                                            0.00413907])).cuda()
        # seg_weights = (torch.mean(seg_weights) - seg_weights) / torch.std(seg_weights) * 0.05 + 1.0

        # seg_weights = [1.0 for i in range(system_configs.seg_n_class)]
        # seg_weights = torch.FloatTensor(seg_weights).cuda()
        # print("seg_weights: {}".format(seg_weights))

        self.seg_ce_loss = torch.nn.CrossEntropyLoss()
        # seg_dice_loss = DiceLoss(n_classes=system_configs.seg_n_class, weight=seg_weights)
        # self.seg_loss = [seg_ce_loss, seg_dice_loss]
        self.network = Network(self.model, self.lane_loss, self.seg_ce_loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes)
        self.flag    = flag

        # print("-------------------self.model:-------------------")
        # for i in dir(self.model):
        #     print(i)

        # print("-------------------self.loss:-------------------")
        # for i in dir(self.loss):
        #     print(i)

        # print("-------------------self.network:-------------------")
        # for i in dir(self.network):
        #     print(i)
        # for name, param in self.network.named_parameters():
        # print(name, param.shape)

        # Count total parameters
        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("Total parameters: {}".format(total_params))

        # Count MACs when input is 360 x 640 x 3
        input_test = torch.randn(1, 3, 360, 640).cuda()
        input_mask = torch.randn(1, 1, 360, 640).cuda()

        # input_test = torch.randn(4, 3, 480, 800).cuda()
        # input_mask = torch.randn(4, 1, 480, 800).cuda()
        macs, params, = profile(self.model, inputs=('lane_shape', 0, input_test, input_mask), verbose=False)
        macs, _ = clever_format([macs, params], "%.3f")
        print('MACs: {}'.format(macs))

        if system_configs.opt_algo == "adam":
            self.lane_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.lane_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=system_configs.weight_decay_rate
            )
        elif system_configs.opt_algo == 'adamW':
            self.lane_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                weight_decay=system_configs.weight_decay_rate
            )
        else:
            raise ValueError("unknown optimizer")

        if system_configs.seg_opt_algo == "adam":
            self.seg_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.seg_opt_algo == "sgd":
            self.seg_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.seg_learning_rate, 
                momentum=0.9, weight_decay=system_configs.seg_weight_decay_rate
            )
        elif system_configs.seg_opt_algo == 'adamW':
            self.seg_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.seg_learning_rate,
                weight_decay=system_configs.seg_weight_decay_rate
            )

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self, model_mode: str, iteration: int, scaler: GradScaler, save: bool, viz_split: str, xs: list, ys: list, **kwargs: any):
        xs = [x.cuda(non_blocking=True) for x in xs]
        ys = [y.cuda(non_blocking=True) for y in ys]
        # print(f'model_mode: {model_mode}, iteration: {iteration}, save: {save}, viz_split: {viz_split}') if system_configs.debug else None

        print(f'model_mode: {model_mode}') if system_configs.debug and iteration == 1 else None
        # for name, param in self.network.named_parameters():
        #     # print(name, param.shape)
        #     if model_mode == 'lane_shape':
        #         if 'lane' in name:
        #             param.requires_grad = True
        #         elif 'seg' in name:
        #             param.requires_grad = False
        #     elif model_mode == 'seg':
        #         if 'lane' in name:
        #             param.requires_grad = False
        #         elif 'seg' in name:
        #             param.requires_grad = True
        #     print(f'name: {name}, param.requires_grad: {param.requires_grad}')  if system_configs.debug and iteration == 1 else None

        self.lane_optimizer.zero_grad()
        self.seg_optimizer.zero_grad()
        with autocast():
            loss_kp = self.network(model_mode, iteration, save, viz_split, xs, ys)

        if model_mode == 'lane_shape':
            loss      = loss_kp[0]
            loss_dict = loss_kp[1:]
            loss      = loss.mean()
            scaler.scale(loss).backward()
            scaler.step(self.lane_optimizer)
            scaler.update()

        elif model_mode == 'seg':
            loss      = loss_kp
            scaler.scale(loss).backward()
            scaler.step(self.seg_optimizer)
            scaler.update()

        if model_mode == 'lane_shape':
            return loss, loss_dict
        elif model_mode == 'seg':
            return loss

    def validate(self, model_mode: str, iteration: int, save: bool, viz_split: str, xs: list, ys: list, **kwargs: any):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]
            loss_kp = self.network(model_mode, iteration, save, viz_split, xs, ys)

            if model_mode == 'lane_shape':
                loss      = loss_kp[0]
                loss_dict = loss_kp[1:]
                loss      = loss.mean()
                return loss, loss_dict

            elif model_mode == 'seg':
                loss      = loss_kp
                return loss

    def test(self, model_mode: str, iteration: int, xs: list, **kwargs: any):
        with torch.no_grad():
            # xs = [x.cuda(non_blocking=True) for x in xs]
            # return self.model(*xs, **kwargs)
            return self.model(model_mode, iteration, *xs, **kwargs)

    def set_lane_lr(self, lr):
        # print("setting learning rate to: {}".format(lr))
        for param_group in self.lane_optimizer.param_groups:
            param_group["lr"] = lr

    def set_seg_lr(self, lr):
        # print("setting learning rate to: {}".format(lr))
        for param_group in self.seg_optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration, is_bbox_only=False):
        cache_file = system_configs.snapshot_file.format(iteration)

        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model.load_state_dict(model_dict)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)
