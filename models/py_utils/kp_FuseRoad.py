import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .detr_loss import SetCriterion
from .matcher import build_matcher

from .misc import *

from sample.vis import save_debug_images_boxes
from config import system_configs

# KEA_module
from .orn import ORConv2d, RotationInvariantPooling
from .SegFormer.SegFormer import get_mit, get_segformer_head

BN_MOMENTUM = 0.1

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        # modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        # self.regular_conv.weight = self.regular_conv.weight.half() if x.dtype == torch.float16 else \
        #     self.regular_conv.weight
        x = torchvision.ops.deform_conv2d(input=x.float(),
                                          offset=offset.float(),
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=(self.padding, self.padding),
                                          # mask=modulator,
                                          stride=self.stride,
                                          )
        # return x
        return self.relu(x)  

class STN(nn.Module):
    def __init__(self, in_channels=512):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 7 * 14, 32),  # in_features, out_features, bias = True
            nn.ReLU(True),
            nn.Linear(32, 3 * 2) # Affine
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, query):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 7 * 14)
        theta = self.fc_loc(xs)  
        theta = theta.view(-1, 2, 3)  # 
        grid = F.affine_grid(theta=theta, size=x.size(), align_corners=True)
        x = F.grid_sample(query, grid, align_corners=True)
        return x
    
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class kp(nn.Module):
    def __init__(self,
                 flag=False,
                 backbone_output_dims=None,
                 attn_dim=None,
                 num_queries=None,
                 aux_loss=None,
                 pos_type=None,
                 drop_out=0.1,
                 num_heads=None,
                 dim_feedforward=None,
                 enc_layers=None,
                 dec_layers=None,
                 pre_norm=None,
                 return_intermediate=None,
                 lsp_dim=None,
                 mlp_layers=None,
                 num_cls=None,
                 norm_layer=FrozenBatchNorm2d
                 ):
        super(kp, self).__init__()
        self.flag = flag

        # SegFormer
        self.encoder = get_mit(system_configs.mit_name)
        if system_configs.mit_pretrained_path:
            print(f'Loading pretrained model from {system_configs.mit_pretrained_path}')
            self.encoder.load_state_dict(torch.load(system_configs.mit_pretrained_path), strict=False)
        self.seg_decoder = get_segformer_head(system_configs.mit_name, system_configs.seg_n_class)

        hidden_dim = attn_dim
        self.lane_aux_loss = aux_loss
        self.lane_position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.lane_query_embed = nn.Embedding(num_queries, hidden_dim)
        self.lane_input_proj = nn.Conv2d(backbone_output_dims, hidden_dim, kernel_size=1)  # the same as channel of self.layer4

        self.lane_transformer = build_transformer(hidden_dim=hidden_dim,
                                                dropout=drop_out,
                                                nheads=num_heads,
                                                dim_feedforward=dim_feedforward,
                                                enc_layers=enc_layers,
                                                dec_layers=dec_layers,
                                                pre_norm=pre_norm,
                                                return_intermediate_dec=return_intermediate)

        self.lane_class_embed    = nn.Linear(hidden_dim, num_cls)
        self.lane_specific_embed = MLP(hidden_dim, hidden_dim, lsp_dim - 4, mlp_layers)
        self.lane_shared_embed   = MLP(hidden_dim, hidden_dim, 4, mlp_layers)

        # debug
        self.image_count = 0

        # SRKE 
        if system_configs.use_SRKE == True:
            print(f'---------------------- using SRKE ----------------------')
            self.lane_map_embedding = nn.Conv2d(1, backbone_output_dims, kernel_size=10, stride=10)
            self.lane_or_conv = ORConv2d(in_channels=backbone_output_dims, out_channels=backbone_output_dims, kernel_size=3, padding=1, arf_config=(1, 8))
            self.lane_or_pool = RotationInvariantPooling(256, 8)
            normal_init(self.lane_or_conv, std=0.01)
            self.lane_stn = STN(in_channels=backbone_output_dims)
            self.lane_prior_query_proj = nn.Sequential(
                nn.Conv2d(system_configs.SRKE_proj_dims, backbone_output_dims, kernel_size=1,
                        stride=1, padding=0, bias=False),
                nn.BatchNorm2d(backbone_output_dims, momentum=BN_MOMENTUM),
            )


    def _make_layer(self, block, planes, blocks, stride=1):
        if system_configs.block == 'BottleNeck':
            planes = int(planes / block.expansion)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _train(self, model_mode='lane_shape', iteration=0, *xs, **kwargs):
        self.image_count += 1
        images = xs[0] 
        masks  = xs[1] 

        ## Segmentation
        encoder_output = self.encoder(images)
        p = encoder_output[-1]
        seg_pred = self.seg_decoder(encoder_output)
        seg_pred = F.interpolate(seg_pred, size=images.size()[2:], mode="bilinear", align_corners=False)

        if model_mode == 'seg':
            return seg_pred
        seg_result = seg_pred
        
        if system_configs.use_SRKE == True and iteration > system_configs.warmup_steps:
            drivable_area_classes = system_configs.drivable_area_classes
            seg_pred = seg_pred.softmax(dim=1)
            seg_pred_tmp = seg_pred[:, drivable_area_classes, :, :]
            drivable_masks = seg_pred_tmp.sum(dim=1, keepdim=True)


        ## SRKE
        if system_configs.use_SRKE == True:
            if iteration > system_configs.warmup_steps:
                map_prompt = drivable_masks
            else:
                map_prompt = torch.logical_not(masks).float()

            query_embed = self.lane_map_embedding(map_prompt)
            or_feat = self.lane_or_conv(query_embed)
            or_pool_feat = self.lane_or_pool(or_feat)
            query = self.lane_stn(or_pool_feat, query_embed)
            query = F.interpolate(query, size=p.shape[-2:])
            query_feat = torch.cat([p, query], dim=1)
            query_feat = self.lane_prior_query_proj(query_feat)
        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos = self.lane_position_embedding(p, pmasks)

        if system_configs.use_SRKE == True:
            hs, _, weights  = self.lane_transformer(self.lane_input_proj(query_feat), pmasks, self.lane_query_embed.weight, pos)
        else:
            hs, _, weights  = self.lane_transformer(self.lane_input_proj(p), pmasks, self.lane_query_embed.weight, pos)

        output_class    = self.lane_class_embed(hs)
        output_specific = self.lane_specific_embed(hs)
        output_shared   = self.lane_shared_embed(hs)
        output_shared   = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared   = output_shared.repeat(1, 1, output_specific.shape[2], 1) 
        output_specific = torch.cat([output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]], dim=-1)
        
        out = {'pred_logits': output_class[-1], 'pred_curves': output_specific[-1]}
        if self.lane_aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_specific)
            
        return out, weights, seg_result

    def _test(self, model_mode, iteration, *xs, **kwargs):
        return self._train(model_mode, iteration, *xs, **kwargs)

    def forward(self, model_mode, iteration, *xs, **kwargs):
        if self.flag:
            return self._train(model_mode, iteration, *xs, **kwargs)
        return self._test(model_mode, iteration, *xs, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_curves': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class AELoss(nn.Module):
    def __init__(self,
                 debug_path=None,
                 aux_loss=None,
                 num_classes=None,
                 dec_layers=None
                 ):
        super(AELoss, self).__init__()
        self.debug_path  = debug_path
        weight_dict = {'loss_ce': 3, 'loss_curves': 5, 'loss_lowers': 2, 'loss_uppers': 2}
        # cardinality is not used to propagate loss
        matcher = build_matcher(set_cost_class=weight_dict['loss_ce'],
                                curves_weight=weight_dict['loss_curves'],
                                lower_weight=weight_dict['loss_lowers'],
                                upper_weight=weight_dict['loss_uppers'])
        losses  = ['labels', 'curves', 'cardinality']

        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(num_classes=num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=1.0,
                                      losses=losses)

    def forward(self,
                iteration,
                save,
                viz_split,
                outputs,
                targets):

        gt_cluxy = [tgt[0] for tgt in targets[1:]]
        loss_dict, indices = self.criterion(outputs, gt_cluxy)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Save detected images during training
        if save:
            which_stack = 0
            save_dir = os.path.join(self.debug_path, viz_split)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # save_name = 'iter_{}_layer_{}'.format(iteration % 5000, which_stack)
            save_name = 'iter_{}_layer_{}'.format(iteration, which_stack)
            save_path = os.path.join(save_dir, save_name)
            with torch.no_grad():
                gt_viz_inputs = targets[0]
                tgt_labels = [tgt[:, 0].long() for tgt in gt_cluxy]
                pred_labels = outputs['pred_logits'].detach()
                prob = F.softmax(pred_labels, -1)
                scores, pred_labels = prob.max(-1)  # 4 10

                pred_curves = outputs['pred_curves'].detach()
                pred_clua3a2a1a0 = torch.cat([scores.unsqueeze(-1), pred_curves], dim=-1)

                save_debug_images_boxes(gt_viz_inputs,
                                        tgt_curves=gt_cluxy,
                                        tgt_labels=tgt_labels,
                                        pred_curves=pred_clua3a2a1a0,
                                        pred_labels=pred_labels,
                                        prefix=save_path)

        return (losses, loss_dict_reduced, loss_dict_reduced_unscaled,
                loss_dict_reduced_scaled, loss_value)
