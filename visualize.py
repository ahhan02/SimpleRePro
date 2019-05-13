import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import os.path as osp
from models.backbone import resnet50_yolov1
from datasets.pascal_voc import PASCAL_VOC
from utils.util import show_result, decoder
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

    model = resnet50_yolov1()
    model = nn.DataParallel(model)

    model_file = 'best.pth'
    model_path = osp.join(osp.dirname(__file__), 'checkpoints', args.trial_log)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        model.load_state_dict(torch.load(osp.join(model_path, model_file), map_location='cpu'))
    else:
        model.load_state_dict(torch.load(osp.join(model_path, model_file)))
    model.to(device)
    model.eval()

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PASCAL_VOC(
        data_root=args.data_root, 
        # img_prefix='VOC2007', 
        # ann_file='VOC2007/ImageSets/Main/test.txt',
        img_prefix='vis', 
        ann_file='vis/vis.txt',
        transform=data_transform,
        num_debug_imgs=-1,
        test_mode=True)
    
    class_names = PASCAL_VOC.CLASSES
    demo_dir = osp.join(workpath, 'demo')

    with torch.no_grad():
        for i, (img, target) in enumerate(test_dataset):
            imgname = test_dataset.img_infos[i]['filename']
            print(i, imgname)
            w, h = test_dataset.img_infos[i]['width'], test_dataset.img_infos[i]['height']

            img = img.to(device)
            pred = model(img[None, :, :, :])

            boxes, probs, labels = decoder(pred, size_grid_cell=args.size_grid_cell,
                num_classes=args.num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
            boxes = boxes.numpy().clip(min=0, max=1)
            boxes *= np.array([w, h, w, h])
            probs = probs.numpy()

            boxes = np.concatenate((boxes, probs[:, np.newaxis]), 1)
            labels = labels.numpy().astype(np.int32)

            out_filename = osp.splitext( osp.split(imgname)[-1] )[0] + '.png'
            show_result(imgname, boxes, labels, class_names, out_file=osp.join(demo_dir, out_filename))

            boxes_gt, labels_gt = test_dataset.decoder(target)
            boxes_gt[:, :4] *= torch.Tensor([w, h, w, h])
            boxes_gt = boxes_gt.numpy().astype(np.int32)
            labels_gt = labels_gt.numpy().astype(np.int32)
            show_result(imgname, boxes_gt, labels_gt, class_names, bbox_color='red', text_color='red')


if __name__ == '__main__':
    main()