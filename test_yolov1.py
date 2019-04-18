#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: xmhan
@LastEditors: xmhan
@Date: 2019-04-11 14:49:12
@LastEditTime: 2019-04-18 17:11:08
'''
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import os.path as osp
import mmcv
from models.backbone import resnet50_yolov1
from yolodataset import YoloV1DatasetVOC
from utils.util import *
                
if __name__ == '__main__':
    model = resnet50_yolov1()
    # if torch.cuda.device_count() > 1:
        # print("Use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs

    trial_log = 'voc07'

    model = nn.DataParallel(model)
    model_path = osp.join(osp.dirname(__file__), 'checkpoints', trial_log)
    model.load_state_dict(torch.load(osp.join(model_path, 'latest.pth'), map_location='cpu'))
    model.eval()

    img_size = 448
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = YoloV1DatasetVOC(
        data_root='/Users/xmhan/data/VOCdevkit', 
        img_prefix='VOC2007', 
        ann_file='VOC2007/ImageSets/Main/val.txt',
        transform=data_transform,
        num_debug_imgs=None,
        test_mode=True)
    
    class_names = YoloV1DatasetVOC.CLASSES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, (img, target) in enumerate(test_dataset):
        # for i, img in enumerate(test_dataset):
            imgname = test_dataset.img_infos[i]['filename']
            print(i, imgname)

            w, h = test_dataset.img_infos[i]['width'], test_dataset.img_infos[i]['height']
            ann = test_dataset.get_ann_info(i)

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
                
            boxes, probs, labels = decoder(pred, num_classes=len(class_names), score_thr=0.15, nms_thr=0.45)
            boxes = boxes.numpy().clip(min=0, max=1)
            boxes *= np.array([w, h, w, h])
            probs = probs.numpy()

            boxes = np.concatenate((boxes, probs[:, np.newaxis]), 1)
            labels = labels.numpy().astype(np.int32)
            show_result(imgname, boxes, labels, class_names)

            boxes_gt, labels_gt = test_dataset.decoder(target)
            boxes_gt[:, :4] *= torch.Tensor([w, h, w, h])
            boxes_gt = boxes_gt.numpy().astype(np.int32)
            labels_gt = labels_gt.numpy().astype(np.int32)
            show_result(imgname, boxes_gt, labels_gt, class_names, bbox_color='red', text_color='red')
