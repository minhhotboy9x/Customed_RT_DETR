import math 
import copy 
from collections import OrderedDict
from typing import Optional, Tuple

import torch 
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F 
try:
    import xformers.ops as xops
    HAS_XFORMERS = True
    print("Using xformers for memory-efficient attention.", HAS_XFORMERS)
except ImportError:
    HAS_XFORMERS = False
    xops = None
    print("xformers not found. Using standard attention implementation.", HAS_XFORMERS)

from .denoising import get_contrastive_denoising_training_group
from .rtdetr_decoder import RTDETRTransformer, TransformerDecoderLayer, TransformerDecoder


from src.core import register

__all__ = ['CustomedRTDETRTransformer4']

class FlashMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype
        )
        # Note: Flash attention is not supported in PyTorch < 2.0
        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise ImportError("Flash attention requires PyTorch 2.0 or later.")
    
    def _in_proj_qkv(self, q: Tensor, k:Tensor, v:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # input shape: (L, N, E) hoặc (N, L, E) nếu batch_first=True
        # in_proj_weight shape: (3*E, E)
        # in_proj_bias shape: (3*E,)
        E = self.embed_dim
        q_proj = F.linear(q, self.in_proj_weight[:E, :], self.in_proj_bias[:E])
        k_proj = F.linear(k, self.in_proj_weight[E:2*E, :], self.in_proj_bias[E:2*E])
        v_proj = F.linear(v, self.in_proj_weight[2*E:, :], self.in_proj_bias[2*E:])
        return q_proj, k_proj, v_proj
    
    def _reshape_to_heads(self, x: Tensor) -> Tensor:
        N, L, E = x.size()
        x = x.view(N, L, self.num_heads, self.head_dim)  # (N, L, heads, head_dim)
        return x # (N, L, heads, head_dim)
    
    def _reshape_from_heads(self, x: Tensor) -> Tensor:
        N, L, heads, head_dim = x.size()
        x = x.view(N, L, heads * head_dim)
        return x  # (N, L, E)

    def convert_attn_mask_to_bias(self, 
                                attn_mask,
                                key_padding_mask, 
                                is_causal,
                                len_q,
                                len_k,
                                num_heads=8):
        attn_bias = None

        # Convert key_padding_mask nếu có
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = key_padding_mask.float()  # Chuyển mask sang float
                key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.bool(), float('-inf'))
            # Shape (N, S) → (N, 1, 1, S)
            if key_padding_mask.dim() == 2:
                key_padding_mask = key_padding_mask[:, None, None, :]
            # Lúc này shape sẽ là (N, 1, 1, S)

        # Nếu attn_mask là None nhưng có key_padding_mask thì dùng luôn
        if attn_mask is None:
            attn_bias = key_padding_mask
        else:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.float()  # Chuyển mask sang float
                attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))  # Điền -inf vào vị trí True
            attn_bias = attn_mask

            # Chuẩn hóa shape
            if attn_bias.dim() == 2:      # (L, S)
                attn_bias = attn_bias[None, None, :, :]  # (1, 1, L, S)
            elif attn_bias.dim() == 3:    # (N*H, L, S)
                N_times_H, L, S = attn_bias.shape
                attn_bias = attn_bias.view(-1, num_heads, L, S)  # (N, H, L, S)
            elif attn_bias.dim() == 4:
                pass  # (N, H, L, S) hoặc (N, 1, L, S)
            else:
                raise ValueError(f"Unsupported attn_bias dim: {attn_bias.shape}")

            # Nếu có cả key_padding_mask thì cộng vào
            if key_padding_mask is not None:
                attn_bias = attn_bias + key_padding_mask  # broadcast qua L

        # Causal mask
        if is_causal:
            B = attn_bias.shape[0] if attn_bias is not None else 1
            L = len_q
            S = len_k
            causal_mask = xops.LowerTriangularMask()
            if attn_bias is None:
                attn_bias = causal_mask.materialize((B, num_heads, L, S))
            else:
                attn_bias = causal_mask.add_bias(attn_bias).materialize((B, num_heads, L, S))

        return attn_bias  # (N, H, L, S)

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
        if not HAS_XFORMERS:
            tgt = super().forward(
                query, key, value, key_padding_mask, need_weights, attn_mask,
                average_attn_weights, is_causal
            )
            return tgt
        else:
            # Use xformers for flash attention
            if not self.batch_first:
                assert "xformers only supports batch_first=True"
            
            query, key, value = self._in_proj_qkv(query, key, value)

            query = self._reshape_to_heads(query)  # (N, L, heads, head_dim)
            key = self._reshape_to_heads(key)      # (N, S, heads, head_dim)
            value = self._reshape_to_heads(value)  # (N, S, heads, head_dim)

            len_q = query.size(1)  # L
            len_k = key.size(1)    # S
            dtype = query.dtype
            device = query.device

            attn_bias = self.convert_attn_mask_to_bias(attn_mask, 
                                                       key_padding_mask,
                                                       is_causal,
                                                       len_q, 
                                                       len_k,
                                                       self.num_heads,) # (N, H, L, S)

            if attn_bias is not None:
                attn_bias = attn_bias.to(query.dtype)

            def pad_to_multiple_of_8(length):
                return ((length + 7) // 8) * 8
            
            if self.training: # training mode need to pad to multiple of 8
                padded_len_q = pad_to_multiple_of_8(len_q)  # e.g., 504 for 498
                padded_len_k = pad_to_multiple_of_8(len_k)  # e.g., 504 for 498

                if attn_bias is not None:
                    padded_attn_bias = torch.full(
                        (attn_bias.size(0), attn_bias.size(1), padded_len_q, padded_len_k),
                        float('-inf'),
                        dtype=dtype,
                        device=device
                    )
                    padded_attn_bias[:, :, :len_q, :len_k] = attn_bias
                else:
                    padded_attn_bias = None

                # print('before', query.shape, attn_bias.shape )
                if len_q != padded_len_q or len_k != padded_len_k:
                    query = F.pad(query, (0, 0, 0, 0, 0, padded_len_q-len_q), mode='constant', value=0)
                    key = F.pad(key, (0, 0, 0, 0, 0, padded_len_k-len_k), mode='constant', value=0)
                    value = F.pad(value, (0, 0, 0, 0, 0, padded_len_k-len_k), mode='constant', value=0)
            else:
                padded_attn_bias = attn_bias
            

            tgt = xops.memory_efficient_attention(
                query, key, value,
                attn_bias=padded_attn_bias,
                p=self.dropout if self.training else 0.0,
            )
            tgt = tgt[:, :len_q, :]  # (N, L, heads, head_dim)
            tgt = self._reshape_from_heads(tgt)  # (N, L, heads * head_dim)
            tgt = self.out_proj(tgt)  # (N, L, E)
            return tgt, None
            
            
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
    
    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           targets=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        query_mask = torch.zeros(
            (bs, self.nhead, self.num_queries), dtype=torch.bool, device=memory.device
        )
        # query_mask[..., 150:] = True  # set first 250 queries to True
        # if targets is not None:
        #     min_keep_queries = 150
        #     alpha = 15
        #     num_targets = [len(t['labels']) for t in targets]
        #     for b, n in enumerate(num_targets):
        #         keep_queries = min(int(min_keep_queries + alpha * math.sqrt(n)), 
        #                            self.num_queries)
        #         # Mask các query cuối: từ keep_queries đến hết
        #         if keep_queries < self.num_queries:
        #             query_mask[b, :, keep_queries:] = True
        
        if targets is not None:
            min_keep_queries = 150
            alpha = 15
            num_targets = torch.as_tensor([len(t['labels']) for t in targets], device=query_mask.device) #[bs]

            keep_queries = min_keep_queries + alpha * num_targets.sqrt() # [bs]
            keep_queries = keep_queries.clamp(max=self.num_queries).long()  # Giới hạn bởi self.num_queries

            # Tạo mặt nạ
            row_idx = torch.arange(self.num_queries, device=query_mask.device).unsqueeze(0).expand(bs, -1) # [bs, num_queries]
            keep_queries_expanded = keep_queries.unsqueeze(1) # [bs, 1]
            keep_mask = row_idx >= keep_queries_expanded  # [bs, num_queries]

            keep_mask = keep_mask.unsqueeze(1)  # [bs, 1, num_queries]
            query_mask = query_mask | keep_mask  # Kết hợp với mặt nạ hiện tại [bs, nhead, num_queries]

            col_keep = ~(query_mask[:, 0, :].all(dim=0))  # [num_queries], True là nên giữ
            last_idx = col_keep.nonzero().max().item() + 1  # cắt đến vị trí này
            query_mask = query_mask[:, :, :last_idx]  # [bs, nhead, new_num_queries]

        new_num_queries = query_mask.shape[-1]  # Số lượng query thực tế sau khi cắt
        query_attn_mask = query_mask.unsqueeze(-1) | query_mask.unsqueeze(-2) # [bs, nhead, num_queries, num_queries]
        query_attn_mask = query_attn_mask.reshape(-1, new_num_queries, new_num_queries)
            # [bs*nhead, new_num_queries, new_num_queries]

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, new_num_queries, dim=1)
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits, \
            query_mask, query_attn_mask


    def forward(self, feats, targets=None):

        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, \
            query_mask, query_attn_mask = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact, targets)

        if not self.training or not dn_meta:
            attn_mask = query_attn_mask
        else:
            bs, n_head, num_new_queries = query_mask.shape
            num_denoising = dn_meta['dn_num_split'][0]
            dn_meta['dn_num_split'][1] = num_new_queries
            attn_mask = attn_mask[:num_denoising+num_new_queries, :num_denoising+num_new_queries]  # [num_denoising+num_new_queries, num_denoising+num_new_queries]
            multihead_query_mask = query_mask.reshape(bs * n_head, num_new_queries)  # [bs * n_head, num_queries]
            attn_mask = attn_mask.unsqueeze(0).repeat(bs * n_head, 1, 1)  # [bs * n_head, tgt_size, tgt_size]
            # (1) Bất kỳ ai (denoising + query) cũng không được nhìn thấy query bị pad
            attn_mask[:, :, num_denoising:] |= multihead_query_mask.unsqueeze(1)  # [bs*n_head, tgt_size, num_queries]
            # (2) Query bị pad không được attend tới bất kỳ ai (kể cả denoising)
            attn_mask[:, num_denoising:, :] |= multihead_query_mask.unsqueeze(-1)  # [bs*n_head, num_queries, tgt_size]
            
        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
            
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
        
        out['query_mask'] = query_mask[:, 0, :]  # [bs, num_queries]

        return out