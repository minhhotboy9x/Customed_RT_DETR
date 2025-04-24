import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .rtdetr_decoder import RTDETRTransformer


from src.core import register

__all__ = ['CustomedRTDETRTransformer']

def batched_iou(anchor_boxes, gt_boxes, gt_masks):
    """
    anchor_boxes: [B, N, 4]
    gt_boxes: [B, M, 4]
    gt_masks: [B, M] - True nếu là box thật
    return: [B, N, M]
    """
    B, N, _ = anchor_boxes.shape
    M = gt_boxes.shape[1]

    # Expand anchor boxes và gt boxes để broadcast
    anchors = anchor_boxes.unsqueeze(2)  # [B, N, 1, 4]
    gts = gt_boxes.unsqueeze(1)         # [B, 1, M, 4]

    # Tính phần giao (intersection)
    inter_x1 = torch.max(anchors[..., 0], gts[..., 0])
    inter_y1 = torch.max(anchors[..., 1], gts[..., 1])
    inter_x2 = torch.min(anchors[..., 2], gts[..., 2])
    inter_y2 = torch.min(anchors[..., 3], gts[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Tính diện tích anchor và gt
    anchor_area = (anchors[..., 2] - anchors[..., 0]) * (anchors[..., 3] - anchors[..., 1])
    gt_area = (gts[..., 2] - gts[..., 0]) * (gts[..., 3] - gts[..., 1])

    # Tính IoU
    union = anchor_area + gt_area - inter_area
    iou = inter_area / union.clamp(min=1e-6)  # tránh chia cho 0

    iou = iou * gt_masks.unsqueeze(1)  # [B, N, M] - chỉ giữ lại IoU với các box thật
    
    return iou  # shape [B, N, M]

def get_box_coords(levels, x, y, strides):
    """
    Trả về (x1, y1, x2, y2) trên ảnh gốc ứng với mỗi điểm (x, y) ở mỗi level.
    """
    stride = strides[levels]  # [bs, 300]

    # tọa độ trên ảnh gốc
    x1 = x * stride
    y1 = y * stride
    x2 = (x + 1) * stride
    y2 = (y + 1) * stride

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [bs, 300, 4]
    return boxes

def get_level_pos(topk_ind, spatial_shapes, level_start_index):
    bs, num_queries = topk_ind.shape
    topk_exp = topk_ind.unsqueeze(-1)  # [bs, 300, 1]
    level_index_exp = level_start_index.view(1, 1, -1)  # [1, 1, num_levels]
    level_mask = topk_exp >= level_index_exp  # [bs, 300, num_levels]
    levels = level_mask.sum(-1) - 1  # [bs, 300]

    level_offsets = level_start_index[levels] # [bs, 300]
    index_in_level = topk_ind - level_offsets  # [bs, 300]

    H, W = spatial_shapes[:, 0], spatial_shapes[:, 1]  # [num_levels]
    H_level = H[levels]  # [bs, 300]
    W_level = W[levels]  # [bs, 300]

    y = index_in_level // W_level  # [bs, 300]
    x = index_in_level % W_level  # [bs, 300]
    
    return levels, x, y

def get_gt_boxes(targets):
    """
    targets: list of dicts, mỗi dict chứa key 'boxes' với shape [num_boxes, 4]

    Trả về:
        padded_boxes: Tensor [B, max_num_boxes, 4]
        box_mask: BoolTensor [B, max_num_boxes] - True nếu là box thật
    """
    max_boxes = max(len(t['boxes']) for t in targets)
    padded_boxes = []
    box_masks = []

    for t in targets:
        boxes = t['boxes']
        num_boxes = boxes.shape[0]
        pad_len = max_boxes - num_boxes

        if pad_len > 0:
            padding = torch.zeros((pad_len, 4), device=boxes.device)
            padded = torch.cat([boxes, padding], dim=0)
            mask = torch.cat([torch.ones(num_boxes, device=boxes.device), 
                              torch.zeros(pad_len, device=boxes.device)], dim=0)
        else:
            padded = boxes
            mask = torch.ones(num_boxes, device=boxes.device)

        padded_boxes.append(padded)
        box_masks.append(mask.bool())

    padded_boxes = torch.stack(padded_boxes)   # [B, max_num_boxes, 4]
    box_masks = torch.stack(box_masks)         # [B, max_num_boxes]

    return padded_boxes, box_masks

def count_overlaped_queries(ious, levels, num_levels=3):
    """
    ious: [B, 300, max_num_gt]
    levels: [B, 300]
    """
    bs, num_queries = ious.shape[:2]
    num_gt = ious.shape[-1]

    overlapped_queries = torch.zeros(bs, num_levels, device=ious.device)  # [B, num_levels]
    queries_per_level = torch.zeros(bs, num_levels, device=ious.device, dtype=torch.int64)  # [B, num_levels]

    for lvl in range(num_levels):
        level_mask = levels == lvl  # [B, 300]
        iou_mask = ious > 0  # [B, 300, max_num_gt]
        overlap_mask = level_mask.unsqueeze(-1) & iou_mask  # [B, 300, max_num_gt]
        overlapped_queries[:, lvl] = overlap_mask.any(dim=-1).sum(dim=-1)  # Count queries per level
        queries_per_level[:, lvl] = level_mask.sum(dim=-1)  # Count total queries per level

    overlapped_queries = overlapped_queries.sum(dim=0)  # [num_levels] total queries per level
    queries_per_level = queries_per_level.sum(dim=0)  # [num_levels] total queries per level
    return overlapped_queries, queries_per_level  # [num_levels]

def sum_max_iou_per_level(ious, levels, num_levels=3):
    """
    ious: [B, 300, max_num_gt]
    levels: [B, 300]
    
    Trả về: [num_levels] - trung bình IoU lớn nhất của các query có overlap tại mỗi level
    """
    bs, num_queries, _ = ious.shape
    max_ious = ious.max(dim=-1).values  # [B, 300] - IoU lớn nhất của mỗi query

    sum_iou_per_level = torch.zeros(num_levels, device=ious.device)
    for lvl in range(num_levels):
        level_mask = levels == lvl  # [B, 300]
        selected_ious = max_ious[level_mask]  # IoU của các query tại level này
        valid_ious = selected_ious[selected_ious > 0]  # Chỉ tính những query có overlap

        if valid_ious.numel() > 0:
            sum_iou_per_level[lvl] = valid_ious.sum()
        else:
            sum_iou_per_level[lvl] = 0.0  # hoặc có thể cho NaN nếu muốn

    return sum_iou_per_level  # [num_levels]

def overlap_statistic(topk_ind, spatial_shapes, level_start_index, targets):
    device = topk_ind.device
    level_start_index = torch.tensor(level_start_index, device=device)
    spatial_shapes = torch.tensor(spatial_shapes, device=device) 
    levels, x, y = get_level_pos(topk_ind, spatial_shapes, level_start_index)
    strides = torch.tensor([8, 16, 32], device=device)  # [num_levels]
    anchor_boxes = get_box_coords(levels, x, y, strides)  # [bs, 300, 4]
    padded_gt_boxes, gt_masks = get_gt_boxes(targets)  # [bs, max_num_gt, 4]
    ious = batched_iou(anchor_boxes, padded_gt_boxes, gt_masks)  # [bs, 300, max_num_gt]
    
    # number of overlapped queries per level
    overlapped_queries, queries_per_level = count_overlaped_queries(ious, levels)  # [num_levels]
    sum_iou_per_level = sum_max_iou_per_level(ious, levels)  # [num_levels]

    return overlapped_queries, queries_per_level, sum_iou_per_level

@register
class CustomedRTDETRTransformer(RTDETRTransformer):
    __share__ = ['num_classes']

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           leverage_class_anchors = [0.],
                           targets=None,):
        bs, _, _ = memory.shape

        # target to detect small objects
        small_mask = torch.zeros(bs, device=memory.device)
        if targets is not None:
            for i in range(len(targets)):
                area = targets[i]['area']
                if area.numel() > 0 and (area < 1024.0).any():
                    small_mask[i] = 1
    
        small_mask = small_mask.unsqueeze(1).unsqueeze(1)
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

        # enc_outputs_class_tmp = F.sigmoid(enc_outputs_class)
        enc_outputs_class_tmp = enc_outputs_class
        for i in range(min(len(leverage_class_anchors), len(spatial_shapes))):
            start_id_anchor = 0 if i == 0 else start_id_anchor + spatial_shapes[i-1][0] * spatial_shapes[i-1][1]
            end_id_anchor = start_id_anchor + spatial_shapes[i][0] * spatial_shapes[i][1]
            enc_outputs_class_tmp[:, start_id_anchor:end_id_anchor, :] += \
                enc_outputs_class_tmp[:, start_id_anchor:end_id_anchor, :] * leverage_class_anchors[i] * small_mask

        _, topk_ind = torch.topk(enc_outputs_class_tmp.max(-1).values, self.num_queries, dim=1)
        
        overlapped_queries = None
        queries_per_level = None
        avg_max_ious = None
        if not self.training:
            overlapped_queries, queries_per_level, sum_iou_per_level = overlap_statistic(topk_ind, spatial_shapes, level_start_index, targets)

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
            overlapped_queries, queries_per_level, sum_iou_per_level


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
            overlapped_queries, queries_per_level, sum_iou_per_level = \
            self._get_decoder_input(memory, spatial_shapes, level_start_index,
                                    denoising_class, 
                                    denoising_bbox_unact,
                                    targets=targets,)

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
        if not self.training:
            out['overlapped_queries'] = overlapped_queries
            out['queries_per_level'] = queries_per_level
            out['sum_iou_per_level'] = sum_iou_per_level

        return out

    