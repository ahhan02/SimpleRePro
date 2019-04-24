import torch
import torch.nn as nn
import mmcv
import logging
import os
import os.path as osp
from collections import defaultdict


def decoder(pred, size_grid_cell=7, num_boxes=2, num_classes=20, conf_thresh=0.1, nms_thresh=0.45):
        '''
        @description: 
        @param :
            pred: (tensor) size(1, 7, 7, 30)
        @return: 
            (tensor) boxes[[x1, y1, x2, y2]] labels[...]
        '''
        # assert(target.size()[0] == 1)
        boxes = []
        probs = []
        labels = []
        cell_size = 1 / size_grid_cell

        # Tensor.cpu()
        pred = pred.cpu()
        
        pred = pred.squeeze(0)                                  # (1, 7, 7, 30) -> (7, 7, 30)
        
        # contain1 = pred[:, :, 4].unsqueeze(2)
        # contain2 = pred[:, :, 9].unsqueeze(2)
        # contain = torch.cat((contain1, contain2), 2)
        contain = pred[:, :, 4].unsqueeze(2)
        for i in range(1, num_boxes):
            contain = torch.cat((contain, pred[:, :, i*5+4].unsqueeze(2)), 2)     
        
        # mask1 = contain > 0.1
        # mask2 = (contain == contain.max())
        # mask = (mask1 + mask2).gt(0)

        # filter bboxes with scores lower than some threshold will be done in nms stage
        # maybe we can always select the better one between `num_boxes` bounding boxes
        # _, mask = torch.max(contain, dim=2, keepdim=True)

        mask = contain > 0

        for i in range(size_grid_cell):
            for j in range(size_grid_cell):
                for b in range(num_boxes):
                    if mask[i, j, b]:
                        box = pred[i, j, b*5:b*5+4]     
                        conf = torch.Tensor([pred[i, j, b*5+4]])   
                        xy = torch.Tensor([j, i]) * cell_size           # uppperleft of the cell
                        box[:2] = box[:2] * cell_size + xy              # return cxcy relative to image

                        box_tlbr = torch.Tensor(4)                      # [cx, cy, w, h] -> [x1, y1, x2, y2]
                        box_tlbr[:2] = box[:2] - 0.5 * box[2:]
                        box_tlbr[2:] = box[:2] + 0.5 * box[2:]
                        prob, label = torch.max(pred[i, j, 5*num_boxes:], 0)

                        if float((conf * prob)[0]) > conf_thresh:
                            boxes.append(box_tlbr.view(-1, 4))
                            probs.append(conf * prob)
                            labels.append(torch.Tensor([label]))
                    
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            probs = torch.zeros((0,))
            labels = torch.zeros((0,))
        else:
            boxes = torch.cat(boxes, 0)                           # [n, 4]
            probs = torch.cat(probs, 0)                           # [n,]
            labels = torch.cat(labels, 0)                         # [n,]
            boxes, probs, labels = nms(boxes, probs, labels, num_classes, nms_thresh)
            
        return boxes, probs, labels


def nms(bboxes, probs, labels, num_classes, nms_thresh=0.45):
    '''
    @description: class-aware nms
    @param : 
        bboxes: (tensor) size([N, 4])
        probs: size([N,])
    @return: 
        keep: mask
    '''
    boxes_keep = []
    probs_keep = []
    labels_keep = []

    for c in range(num_classes):
        cls_inds = labels == c
        if not cls_inds.any():
            continue

        x1 = bboxes[cls_inds, 0]
        y1 = bboxes[cls_inds, 1]
        x2 = bboxes[cls_inds, 2]
        y2 = bboxes[cls_inds, 3]
        area = (x2 - x1) * (y2 - y1)

        keep = []
        _bboxes = bboxes[cls_inds]
        _probs = probs[cls_inds]
        _labels = labels[cls_inds]
        _, order = _probs.sort(0, descending=True)

        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(torch.tensor(order.item()))
                break
                
            i = order[0]
            keep.append(i)                                      # keep the one which has largest prob
            
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            ovr = inter / (area[i] + area[order[1:]] - inter)
            ids = (ovr <= nms_thresh).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]                                    # contine

        # keep = torch.tensor(keep, dtype=torch.long)
        keep = torch.LongTensor(keep)
        boxes_keep.append(_bboxes[keep])
        probs_keep.append(_probs[keep])
        labels_keep.append(_labels[keep])

    boxes_keep = torch.cat(boxes_keep, 0)                             # [n, 4]
    probs_keep = torch.cat(probs_keep, 0)                             # [n,]
    labels_keep = torch.cat(labels_keep, 0)                           # [n,]   
    return boxes_keep, probs_keep, labels_keep


def show_result(img, bboxes, labels, class_names, conf_thresh=0.1, thickness=1, font_scale=0.5, 
    bbox_color='green', text_color='green'):
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=conf_thresh,
        thickness=thickness, 
        font_scale=font_scale,
        bbox_color=bbox_color,
        text_color=text_color)


def get_logger(trial_log, model_path):
    # logger 
    logger = logging.getLogger(trial_log) 
    logger.setLevel(logging.DEBUG) 

    # file handler
    fh = logging.FileHandler(osp.join(model_path, 'train.log'))
    fh.setLevel(logging.INFO) 

    # console handler
    ch = logging.StreamHandler() 
    ch.setLevel(logging.DEBUG) 

    # handler formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    fh.setFormatter(formatter) 
    ch.setFormatter(formatter) 

    # logger handler 
    logger.addHandler(fh) 
    logger.addHandler(ch)

    return logger
     

def get_learning_rate(iter_num, base_lr=0.001, burn_in=1000, power=4):
    if iter_num < burn_in:
        return base_lr * pow(iter_num / burn_in, power)