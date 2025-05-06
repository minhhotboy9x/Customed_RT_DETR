"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from src.core import register


__all__ = ['CustomedPostProcessor', 'CustomedPostProcessor2', 'CustomedPostProcessor3']


# postprocess without repeated queries
@register
class CustomedPostProcessor(nn.Module):
    '''
    box output in format 
    '''
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category', 'conf']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False, conf=0.25) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 
        self.conf = conf

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        
        # fix wrong postprocess
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)
        
        scores, labels = scores.max(dim=-1)
        boxes = bbox_pred
        if scores.shape[1] > self.num_top_queries:
            scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
            labels = torch.gather(labels, dim=1, index=index)
            boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            mask = sco > self.conf
            lab = lab[mask]
            box = box[mask]
            sco = sco[mask]
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )

# postprocess with counting repeated queries
@register
class CustomedPostProcessor2(nn.Module):
    '''
    box output in format 
    '''
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category', 'selected_query']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False, selected_query=True) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 
        self.selected_query = selected_query

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        num_repeated_queries = 0

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
            if self.selected_query:
                one_hot = torch.nn.functional.one_hot(index)
                freq = one_hot.sum(dim=1)
                duplicate_mask = freq > 1
                num_repeated_queries += duplicate_mask.sum()

            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results, num_repeated_queries
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )


# postprocess with counting overlapped queries
@register
class CustomedPostProcessor3(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results, outputs['overlapped_queries'], outputs['queries_per_level'], outputs['sum_iou_per_level'], \
                outputs['matched_sum_count'], outputs['unmatched_sum_count']
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )
