import math 
import copy 
from collections import OrderedDict
from typing import Optional, Tuple

import torch 
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
import torchvision
import torchvision.transforms.v2


from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .rtdetr_decoder import RTDETRTransformer, TransformerDecoderLayer, TransformerDecoder

from src.core import register

__all__ = ['CustomedRTDETRTransformer4']

class FlashMultiheadAttention(nn.MultiheadAttention):

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False, # set to False to activate flash attention
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        tgt = super().forward(
            query, key, value, key_padding_mask, need_weights, attn_mask,
            average_attn_weights, is_causal
        )
        return tgt



@register
class CustomedRTDETRTransformer4(RTDETRTransformer):
    __share__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True):
        
        super().__init__(num_classes=num_classes,
                         hidden_dim=hidden_dim,
                         num_queries=num_queries,
                         position_embed_type=position_embed_type,
                         feat_channels=feat_channels,
                         feat_strides=feat_strides,
                         num_levels=num_levels,
                         num_decoder_points=num_decoder_points,
                         nhead=nhead,
                         num_decoder_layers=num_decoder_layers,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         activation=activation,
                         num_denoising=num_denoising,
                         label_noise_ratio=label_noise_ratio,
                         box_noise_scale=box_noise_scale,
                         learnt_init_query=learnt_init_query,
                         eval_spatial_size=eval_spatial_size,
                         eval_idx=eval_idx,
                         eps=eps,
                         aux_loss=aux_loss)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points
        )
        decoder_layer.self_attn = FlashMultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)