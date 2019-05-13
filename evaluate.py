import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import os.path as osp
from models.backbone import resnet50_yolov1
from datasets.pascal_voc import PASCAL_VOC
from utils.util import show_result, decoder, get_logger
from utils.voc_eval import calc_map
import argparse
import yaml

parser = argparse.ArgumentParser(
    description='Pytorch YoloV1 Training')
parser.add_argument('--trial_log', default='voc07+12_moreaug_14x14')
parser.add_argument('--config', default='configs/config.yaml')
args = parser.parse_args()


def main():
    global args
    args = parser.parse_args()

    workpath = osp.abspath(osp.dirname(__file__))
    with open(osp.join(workpath, args.config)) as f:
        if yaml.__version__ == '5.1':
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    model_file = 'best.pth'
    model_path = osp.join(osp.dirname(__file__), 'checkpoints', args.trial_log)

    img_size = args.img_size
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PASCAL_VOC(
        data_root=args.data_root, 
        img_prefix='VOC2007', 
        ann_file='VOC2007/ImageSets/Main/test.txt',
        transform=data_transform,
        num_debug_imgs=args.num_debug_imgs,
        test_mode=True)

    logger = get_logger(args.trial_log, model_path)
    calc_map(logger, test_dataset, model_path, model_file,
             args.size_grid_cell, args.num_boxes, args.num_classes,
             args.conf_thresh, args.iou_thresh, args.nms_thresh)


if __name__ == '__main__':
    main()