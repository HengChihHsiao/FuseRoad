import torch
import torch.nn as nn

# from .py_utils import kp, AELoss
from .py_utils.kp_FuseRoad import kp, AELoss
from config import system_configs

BN_MOMENTUM = 0.1


class model(kp):
    def __init__(self, flag=False):

        backbone_output_dims        = system_configs.backbone_output_dims
        attn_dim        = system_configs.attn_dim
        dim_feedforward = system_configs.dim_feedforward

        num_queries = system_configs.num_queries  # number of joints
        drop_out    = system_configs.drop_out
        num_heads   = system_configs.num_heads
        enc_layers  = system_configs.enc_layers
        dec_layers  = system_configs.dec_layers
        lsp_dim     = system_configs.lsp_dim
        mlp_layers  = system_configs.mlp_layers
        lane_cls     = system_configs.lane_categories

        aux_loss = system_configs.aux_loss
        pos_type = system_configs.pos_type
        pre_norm = system_configs.pre_norm
        return_intermediate = system_configs.return_intermediate

        super(model, self).__init__(
            flag=flag,
            backbone_output_dims=backbone_output_dims,
            attn_dim=attn_dim,
            num_queries=num_queries,
            aux_loss=aux_loss,
            pos_type=pos_type,
            drop_out=drop_out,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            return_intermediate=return_intermediate,
            num_cls=lane_cls,
            lsp_dim=lsp_dim,
            mlp_layers=mlp_layers
        )

class lane_loss(AELoss):
    def __init__(self):
        super(lane_loss, self).__init__(
            debug_path=system_configs.result_dir,
            aux_loss=system_configs.aux_loss,
            num_classes=system_configs.lane_categories,
            dec_layers=system_configs.dec_layers
        )

