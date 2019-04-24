import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import os.path as osp
import mmcv
from models.backbone import resnet50_yolov1
from datasets.pascal_voc import PASCAL_VOC
from utils.utils import show_result, decoder
                
if __name__ == '__main__':
    trial_log = 'voc07+12_aug'

    model = resnet50_yolov1()
    # if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

    model_path = osp.join(osp.dirname(__file__), 'checkpoints', trial_log)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        model.load_state_dict(torch.load(osp.join(model_path, 'best.pth'), map_location='cpu'))
    else:
        model.load_state_dict(torch.load(osp.join(model_path, 'best.pth')))

    model.eval()

    img_size = 448
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PASCAL_VOC(
        data_root='/Users/xmhan/data/VOCdevkit', 
        img_prefix='VOC2007', 
        ann_file='VOC2007/ImageSets/Main/test.txt',
        transform=data_transform,
        num_debug_imgs=None,
        test_mode=True)
    
    class_names = PASCAL_VOC.CLASSES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, (img, target) in enumerate(test_dataset):
            imgname = test_dataset.img_infos[i]['filename']
            print(i, imgname)

            w, h = test_dataset.img_infos[i]['width'], test_dataset.img_infos[i]['height']
            ann = test_dataset.get_ann_info(i)

            img = img.to(device)
            pred = model(img[None, :, :, :])

            boxes, probs, labels = decoder(pred, num_classes=len(class_names), conf_thresh=0.15, nms_thresh=0.45)
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
