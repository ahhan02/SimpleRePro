#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: xmhan
@LastEditors: xmhan
@Date: 2019-04-12 23:32:48
@LastEditTime: 2019-04-18 17:21:57
'''
import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import os
import os.path as osp
import numpy as np
from yolodataset import YoloV1DatasetVOC
from models.backbone import resnet50_yolov1
import mmcv
from utils.util import decoder
import tqdm
from collections import defaultdict

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1 , 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p/11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size-1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])

    return ap


def voc_eval(logger, det_results, gt_results, class_names, iou_thr=0.5, use_07_metric=False):
    '''
    @description: 
    @param: 
        det_results (dict): detection bboxes of each image, a list of K*6 ndarray
            {'0': [np.array([[image_id, confidence, x1, y1, x2, y2], ...]), ...], ...}
        gt_results (dict): detection bboxes of each image, a list of K*5 array
            {('0', '00005'): {'box': [[x1, y1, x2, y2], ...], 'det': [False, ...]}, ...]}, ...}
    @return:
    '''
    aps = []  
    for i in range(len(class_names)):
        # the i-th class
        cls_dets = det_results[i]

        npos = 0
        for cls_, img_ in gt_results:
            if cls_ == i:
                npos += len(gt_results[(cls_, img_)]['box'])

        if len(cls_dets) == 0:
            ap = -1
            logger.info('---class {} ap {}---'.format(class_names[i], ap))
            aps += [ap]
            continue

        # extract infos
        image_ids = [cls_det[0] for cls_det in cls_dets]
        confidence = np.array([cls_det[1] for cls_det in cls_dets])
        BB = np.array([cls_det[2:] for cls_det in cls_dets])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        # sorted_scores = np.sort(-confidence)

        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            bb = BB[d]
            ovmax = -np.inf
            R = gt_results[(i, image_ids[d])]
            BBGT = np.array(R['box'])

            if len(BBGT) > 0:
                # compute overlaps
                ixmin = np.maximum(BBGT[:, 0], bb[0]) 
                iymin = np.maximum(BBGT[:, 1], bb[1]) 
                ixmax = np.minimum(BBGT[:, 2], bb[2]) 
                iymax = np.minimum(BBGT[:, 3], bb[3]) 
                iw = np.maximum(ixmax - ixmin + 1., 0.) 
                ih = np.maximum(iymax - iymin + 1., 0.) 
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + 
                             (BBGT[:, 2] - BBGT[:, 0] + 1.) * 
                             (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            
            if ovmax > iou_thr:
                if not R['det'][jmax]:
                    tp[d] = 1
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1
            else:
                fp[d] = 1

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / npos

        # avoid divide by zero in case the first detection matches a difficult
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = voc_ap(rec, prec, use_07_metric)
        logger.info('---class {} ap {}---'.format(class_names[i], ap))

        aps += [ap]

    _map = np.mean(aps)
    logger.info('---map {}---'.format(_map))
    return _map


def calc_map(logger, test_dataset, trial_log=''):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet50_yolov1()
    model = nn.DataParallel(model)
    model_path = osp.join(osp.dirname(__file__), 'checkpoints', trial_log)

    if device.type == 'cpu':
        model.load_state_dict(torch.load(osp.join(model_path, 'latest.pth'), map_location='cpu'))
    else:
        model.load_state_dict(torch.load(osp.join(model_path, 'latest.pth')))
    model.to(device)
    model.eval()

    class_names = YoloV1DatasetVOC.CLASSES

    cls_det_results = defaultdict(list)
    cls_img_gt_results = defaultdict(lambda : defaultdict(list))

    with torch.no_grad():
        for i, (img, _) in tqdm.tqdm(enumerate(test_dataset)):
            w, h = test_dataset.img_infos[i]['width'], test_dataset.img_infos[i]['height']
            imgid = test_dataset.img_infos[i]['id']
            
            ann = test_dataset.get_ann_info(i)
            bboxs, labels = ann['bboxes'], ann['labels']

            for j in range(len(bboxs)):
                # remember align label
                cls_img_gt_results[ (labels[j]-1, imgid) ]['box'].append(bboxs[j])
                cls_img_gt_results[ (labels[j]-1, imgid) ]['det'].append(False)

            img = img.to(device)
            pred = model(img[None, :, :, :])

            # def convert_input_tensor_dim(in_tensor):
            #     out_tensor = torch.FloatTensor(in_tensor.size())
            #     out_tensor[:,:,:,:] = 0.
            #     out_tensor[:, :, :, 4] = in_tensor[:, :, :, 0]
            #     out_tensor[:, :, :, 9] = in_tensor[:, :, :, 1]
            #     out_tensor[:, :, :, :4] = in_tensor[:, :, :, 2:6]
            #     out_tensor[:, :, :, 5:9] = in_tensor[:, :, :, 6:10]
            #     out_tensor[:, :, :, 10:] = in_tensor[:, :, :, 10:]
            #     return out_tensor
            # pred = convert_input_tensor_dim(pred)

            # dont do nms when computing map
            # boxes, probs, labels = decoder(pred, num_classes=len(class_names), score_thr=0.1, nms_thr=0.45)
            boxes, probs, labels = decoder(pred, num_classes=len(class_names), score_thr=0.001, nms_thr=1.0)
            boxes, probs, labels = boxes.numpy(), probs.numpy(), labels.numpy().astype(np.int32)
            boxes = boxes.clip(min=0, max=1)

            boxes *= np.array([w, h, w, h]).astype(np.int32)
            for j in range(len(boxes)):
                # save in the `label`-th list
                cls_det_results[ labels[j] ].append([imgid, probs[j], boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]])

        _map = voc_eval(logger, cls_det_results, cls_img_gt_results, class_names, iou_thr=0.5, use_07_metric=False)
        return _map

if __name__ == '__main__':
    # preds = {0:[['image01', 0.9, 20, 20, 40, 40],
    #             ['image01', 0.8, 20, 20, 50, 50],
    #             ['image02', 0.8, 30, 30, 50, 50]],
    #          1:[['image01', 0.78, 60, 60, 90, 90]]}
    # target = {(0, 'image01'): {'box':[[20, 20, 41, 41]], 'det': [False]},
    #           (1, 'image01'): {'box':[[60, 60, 91, 91]], 'det': [False]},
    #           (0, 'image02'): {'box':[[30, 30, 51, 51]], 'det': [False]}}
    
    trial_log = 'haoran'

    import logging
    logger = logging.getLogger(trial_log) 
    logger.setLevel(logging.DEBUG) 

    ch = logging.StreamHandler() 
    ch.setLevel(logging.DEBUG) 
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    ch.setFormatter(formatter) 
    logger.addHandler(ch)

    # voc_eval(logger, preds, target, ['cat','dog'])
    # exit(0)

    img_size = 448
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = YoloV1DatasetVOC(
        # data_root='/Users/xmhan/data/VOCdevkit',
        data_root='/data/data/VOCdevkit',
        img_prefix='VOC2007',
        ann_file='VOC2007/ImageSets/Main/test.txt',
        transform=data_transform,
        with_difficult=False,
        num_debug_imgs=None,
        test_mode=True)

    calc_map(logger, test_dataset, trial_log)