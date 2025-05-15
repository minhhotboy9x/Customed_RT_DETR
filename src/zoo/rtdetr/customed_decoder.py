import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
import torchvision
import torchvision.transforms.v2


from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .rtdetr_decoder import RTDETRTransformer


from src.core import register

__all__ = ['CustomedRTDETRTransformer', 'CustomedRTDETRTransformer2']

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

def get_gt_boxes(targets, recalculate_area=False, noramlized=False):
    """
    targets: list of dicts, mỗi dict chứa key 'boxes' với shape [num_boxes, 4]

    Trả về:
        padded_boxes: Tensor [B, max_num_boxes, 4]
        box_mask: BoolTensor [B, max_num_boxes] - True nếu là box thật
    """
    max_boxes = max(len(t['boxes']) for t in targets)
    padded_boxes = []
    box_masks = []
    box_sizes = []
    
    for t in targets:
        boxes = t['boxes']
        orig_size = t['orig_size']
        if recalculate_area:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if noramlized:
                area = area * (orig_size[0] * orig_size[1])
        else:
            area = t['area']
        num_boxes = boxes.shape[0]
        pad_len = max_boxes - num_boxes

        size = torch.full((num_boxes,), -1, dtype=torch.long, device=area.device)
        size[area < 32**2] = 0       # small
        size[(area >= 32**2) & (area < 96**2)] = 1  # medium
        size[area >= 96**2] = 2      # large

        if pad_len > 0:
            padding = torch.zeros((pad_len, 4), device=boxes.device)
            padded_box = torch.cat([boxes, padding], dim=0)
            padded_size = torch.full((pad_len,), -1, dtype=torch.long, device=boxes.device)
            padded_size = torch.cat([size, padded_size], dim=0)
            mask = torch.cat([torch.ones(num_boxes, device=boxes.device), 
                              torch.zeros(pad_len, device=boxes.device)], dim=0)
        else:
            padded_box = boxes
            padded_size = size
            mask = torch.ones(num_boxes, device=boxes.device)

        padded_boxes.append(padded_box)
        box_masks.append(mask.bool())
        box_sizes.append(padded_size)

    padded_boxes = torch.stack(padded_boxes)   # [B, max_num_boxes, 4]
    box_masks = torch.stack(box_masks)         # [B, max_num_boxes]
    box_sizes = torch.stack(box_sizes)         # [B, max_num_boxes]

    return padded_boxes, box_masks, box_sizes

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

def sum_max_ious_match_unmatch_per_level(ious, levels, num_levels=3, iou_threshold=0.5):
    """
    ious: Tensor[B, 300, max_num_gt]
    levels: Tensor[B, 300]
    """
    max_ious, _ = ious.max(dim=2)  # [B, 300]: IoU max của mỗi query
    matched_mask = max_ious >= iou_threshold  # [B, 300]
    
    matched_sum = torch.zeros(num_levels, device=ious.device)
    unmatched_sum = torch.zeros(num_levels, device=ious.device)
    matched_count = torch.zeros(num_levels, dtype=torch.int, device=ious.device)
    unmatched_count = torch.zeros(num_levels, dtype=torch.int, device=ious.device)

    for lvl in range(num_levels):
        lvl_mask = (levels == lvl)  # [B, 300]
        
        matched = matched_mask & lvl_mask
        unmatched = (~matched_mask) & lvl_mask

        matched_sum[lvl] = max_ious[matched].sum()
        unmatched_sum[lvl] = max_ious[unmatched].sum()

        matched_count[lvl] = matched.sum()
        unmatched_count[lvl] = unmatched.sum()

    return matched_sum, unmatched_sum, matched_count, unmatched_count

def overlap_statistic(topk_ind, spatial_shapes, level_start_index, targets):
    device = topk_ind.device
    level_start_index = torch.tensor(level_start_index, device=device)
    spatial_shapes = torch.tensor(spatial_shapes, device=device) 
    levels, x, y = get_level_pos(topk_ind, spatial_shapes, level_start_index)
    strides = torch.tensor([8, 16, 32], device=device)  # [num_levels]
    anchor_boxes = get_box_coords(levels, x, y, strides)  # [bs, 300, 4]
    padded_gt_boxes, gt_masks, _ = get_gt_boxes(targets)  # [bs, max_num_gt, 4]
    ious = batched_iou(anchor_boxes, padded_gt_boxes, gt_masks)  # [bs, 300, max_num_gt]
    
    # number of overlapped queries per level
    overlapped_queries, queries_per_level = count_overlaped_queries(ious, levels)  # [num_levels]
    sum_iou_per_level = sum_max_iou_per_level(ious, levels)  # [num_levels]
    matched_sum, unmatched_sum, matched_count, unmatched_count = sum_max_ious_match_unmatch_per_level(ious, levels)  # [num_levels]
    return overlapped_queries, queries_per_level, sum_iou_per_level, \
        (matched_sum, matched_count), (unmatched_sum, unmatched_count)


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
        sum_iou_per_level = None
        matched_sum = None
        matched_count = None
        unmatched_sum = None
        unmatched_count = None
        if not self.training:
            overlapped_queries, queries_per_level, sum_iou_per_level, \
            (matched_sum, matched_count), (unmatched_sum, unmatched_count) = overlap_statistic(topk_ind, spatial_shapes, level_start_index, targets)

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
                overlapped_queries, queries_per_level, sum_iou_per_level, \
                (matched_sum, matched_count), (unmatched_sum, unmatched_count)


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
                overlapped_queries, queries_per_level, sum_iou_per_level, \
                (matched_sum, matched_count), (unmatched_sum, unmatched_count) = \
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
            out['matched_sum_count'] = (matched_sum, matched_count)
            out['unmatched_sum_count'] = (unmatched_sum, unmatched_count)

        return out

@register
class CustomedRTDETRTransformer2(RTDETRTransformer):
    __share__ = ['num_classes']

    def _get_rel_center_gt_boxes(self, targets):
        tmp_targets = copy.deepcopy(targets)
        if not self.training:
            for i in range(len(tmp_targets)):
                boxes = tmp_targets[i]['boxes']
                canvas_size = boxes.canvas_size
                if not boxes.normalize:
                    boxes.data = boxes.data / torch.tensor(canvas_size[::-1], device=boxes.device).tile(2)[None]
                    boxes.normalize = True
                boxes = torchvision.transforms.v2.functional.convert_bounding_box_format(
                    boxes, new_format='cxcywh')
                tmp_targets[i]['boxes'] = boxes

        gt_boxes, gt_masks, gt_sizes = get_gt_boxes(tmp_targets, recalculate_area=True, noramlized=True)  
            # [bs, max_num_gt, 4], format xyxy
        return gt_boxes, gt_masks, gt_sizes

    def _get_center_anchor_gt_boxes(self, 
                                    spatial_shapes, 
                                    level_start_index,
                                    targets):
        gt_boxes, gt_masks, gt_sizes = self._get_rel_center_gt_boxes(targets)  # [bs, max_num_gt, 4], format xyxy
        b, max_num_gt, _ = gt_boxes.shape
        gt_anchors = -torch.ones((len(level_start_index), b, max_num_gt), device=gt_boxes.device) # [num_levels, bs, max_num_gt, 2]
        for i in range(len(level_start_index)):
            start_id_anchor = level_start_index[i]
            scale = torch.tensor([spatial_shapes[i][1], spatial_shapes[i][0]], device=gt_boxes.device) # [W, H]
            gt_centers = (gt_boxes[..., :2] * scale).long()
            gt_anchor_indexes = gt_centers[..., 1] * scale[0] + gt_centers[..., 0] # [bs, max_num_gt]
            gt_anchors[i] = start_id_anchor + gt_anchor_indexes # [bs, max_num_gt]

        return gt_anchors.long(), gt_boxes, gt_masks, gt_sizes

    def _get_diff_center_gt_and_topk(self, 
                                    spatial_shapes, 
                                    level_start_index,
                                    targets,
                                    enc_topk_logits, # enc_topk_logits.shape = [bs, 300, num_class]
                                    enc_outputs_class # enc_outputs_class.shape = [bs, sum(w*h), num_class]
                                    ):
        gt_anchors, gt_boxes, gt_masks, gt_sizes = self._get_center_anchor_gt_boxes(spatial_shapes, 
                                                                                    level_start_index, 
                                                                                    targets)
        # gt_anchors.shape = [num_levels, bs, max_num_gt]
        # gt_boxes.shape = [bs, max_num_gt, 4]
        # gt_masks.shape = [bs, max_num_gt]
        # gt_sizes.shape = [bs, max_num_gt]

        enc_score_class = F.sigmoid(enc_outputs_class.max(-1).values) # [bs, sum(w*h)]
        enc_topk_score = F.sigmoid(enc_topk_logits.max(-1).values) # [bs, 300]
        enc_smallest_topk_score = enc_topk_score.min(dim=1).values # [bs] smallest score of topk

        diff_each_level = []
        num_diff_each_level = []
        for i in range(len(level_start_index)): # each level feature map
            diff_each_obj_size = torch.zeros(3, device=gt_boxes.device) # 3 for small, medium, large
            num_box_each_size = torch.zeros(3, device=gt_boxes.device) # 3 for small, medium, large
            gt_anchors_i = gt_anchors[i] # [bs, max_num_gt]
            score_center_i = enc_score_class.gather(dim=1, index=gt_anchors_i) # [bs, max_num_gt]
            diff_score_i = score_center_i - enc_smallest_topk_score.unsqueeze(1) # [bs, max_num_gt]
            gt_sizes_i = gt_sizes  # [bs, max_num_gt]

            for size in range(3):  # 0=small, 1=medium, 2=large
                # Tạo mask cho GT box hợp lệ và có size tương ứng
                valid_mask = (gt_sizes_i == size) # [bs, max_num_gt]
                
                # Lấy các diff score tương ứng
                valid_diff = diff_score_i[valid_mask]  # 1D tensor
                num_box_each_size[size] = valid_mask.sum()  # Số lượng GT box hợp lệ có size tương ứng
                # Tính tổng nếu có ít nhất một phần tử hợp lệ
                if valid_diff.numel() > 0:
                    diff_each_obj_size[size] = valid_diff.sum()

            diff_each_level.append(diff_each_obj_size)
            num_diff_each_level.append(num_box_each_size)
        diff_each_level = torch.stack(diff_each_level, dim=0) # [num_feat, num_obj_size]
        num_diff_each_level = torch.stack(num_diff_each_level, dim=0) # [num_feat, num_obj_size]
        return diff_each_level, num_diff_each_level
    
    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           targets=None,):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory) # [b, sum(w*h), num_class=80]
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors # [b, sum(w*h), 4]

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1) # [b, 300]

        if self.training: # replace some of the topk with the gt boxes
            assert targets is not None, "targets must be provided during training"
            gt_anchors, _, gt_masks, gt_sizes = self._get_center_anchor_gt_boxes(
                                                            spatial_shapes, 
                                                            level_start_index, 
                                                            targets)
            num_lvl, _, max_num_gt = gt_anchors.shape

            gt_masks &= (gt_sizes == 0) # [bs, max_num_gt], chỉ lấy các box nhỏ


            gt_anchors_flat = gt_anchors.permute(1, 0, 2).reshape(bs, num_lvl*max_num_gt) # [bs, num_lvl*max_num_gt]

            gt_masks_flat = gt_masks.unsqueeze(1).expand(bs, num_lvl, max_num_gt).reshape(bs, num_lvl * max_num_gt).clone()  # [B, L*max_num_gt]

            # kiểm tra xem có anchor nào trong trùng với topk không
            # → [bs, N, 1] == [bs, 1, K] → [bs, N, K]
            conflict = (gt_anchors_flat.unsqueeze(-1) == topk_ind.unsqueeze(1))  # [bs, num_lvl*max_num_gt, 300]

            # Với mỗi GT, nếu bất kỳ anchor nào trùng, đánh dấu là True
            has_conflict = conflict.any(dim=-1)  # [bs, num_lvl*max_num_gt]

            # Cập nhật gt_masks_flat: chỉ giữ những phần tử không conflict
            gt_masks_flat &= ~has_conflict

            valid_gt = torch.where(gt_masks_flat, gt_anchors_flat, torch.full_like(gt_anchors_flat, -1))  # [B, num_lvl*max_gt]
            
            valid_gt, _ = valid_gt.sort(dim=1, descending=True)  # Sắp xếp theo thứ tự tăng dần

            valid_count = (valid_gt >= 0).sum(dim=1)  # [B], số lượng giá trị hợp lệ cho mỗi sample
            replace_count = torch.minimum(valid_count, torch.full_like(valid_count, 300)).long() # [B], số lượng giá trị hợp lệ cho mỗi sample
            
            # Tạo một mask để xác định vị trí trong topk cần thay thế
            topk_mask = torch.arange(topk_ind.size(1), device=topk_ind.device).unsqueeze(0) \
                >= (topk_ind.size(1) - replace_count.unsqueeze(1)) # [bs, 300]
            
            mask_valid_gt = torch.arange(valid_gt.size(1), device=topk_ind.device).unsqueeze(0) < replace_count.unsqueeze(1)  # [bs, max_valid_len]


            # Khởi tạo tensor mới để gán đúng phần valid_gt vào cuối
            topk_ind[topk_mask] = valid_gt[torch.arange(bs, device=topk_ind.device)
                                            .repeat_interleave(replace_count), 
                                      torch.arange(valid_gt.size(1), device=topk_ind.device)
                                            .repeat(bs, 1)[mask_valid_gt]]

            # for b in range(bs):
            #     # Get the number of replacements for the current sample
            #     num_replacements = replace_count[b].item()
                
            #     # Replace from the end of topk_ind
            #     topk_ind[b, -num_replacements:] = valid_gt[b, :num_replacements]

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

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits, topk_ind, \
                enc_outputs_class

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

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind, enc_outputs_class = \
            self._get_decoder_input(memory, spatial_shapes, level_start_index, 
                                    denoising_class, denoising_bbox_unact, targets)

        if not self.training and targets is not None:
            diff_each_level, num_diff_each_level = self._get_diff_center_gt_and_topk(
                                                spatial_shapes, 
                                                level_start_index, 
                                                targets,
                                                enc_topk_logits,
                                                enc_outputs_class)
            

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

        if not self.training and targets is not None:
            out['stat_diff_gt_center'] = {
                'diff_each_level': diff_each_level,
                'num_diff_each_level': num_diff_each_level,
            }

        out['topk_ind'] = topk_ind
        return out