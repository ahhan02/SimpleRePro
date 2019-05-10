import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import mmcv
# from utils.utils import show_result

def show_result(img, bboxes, labels, class_names, conf_thresh=0.1, thickness=1, font_scale=0.5, 
    bbox_color='green', text_color='green'):
    if isinstance(img, str):
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


class PASCAL_VOC(data.Dataset):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    def __init__(self, 
                 data_root, 
                 img_prefix, 
                 ann_file,
                 num_boxes=2,
                 size_grid_cell=7,
                 img_size=448,
                 do_augmentation=False,
                 transform=None,
                 with_difficult=True,
                 num_debug_imgs=-1,
                 test_mode=False):
        '''
        @description: load dataset for yolov1
        @param : 
            data_root: /Users/xmhan/data/VOCdevkit
            img_prefix: VOC2007 | VOC2012
            ann_file: VOC2007/ImageSets/Main/trainval.txt | VOC2012/ImageSets/Main/trainval.txt
        @return: 
        '''
        self.data_root = data_root
        self.img_prefix = img_prefix if isinstance(img_prefix, list) else [img_prefix]
        self.ann_file = ann_file if isinstance(ann_file, list) else [ann_file]
        self.num_boxes = num_boxes
        self.size_grid_cell = size_grid_cell
        self.img_size = img_size
        self.with_difficult = with_difficult
        self.transform = transform
        self.do_augmentation = do_augmentation
        self.num_debug_imgs = num_debug_imgs
        self.test_mode = test_mode
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

        assert len(self.img_prefix) == len(self.ann_file)

        num_ds = len(self.ann_file)
        self.img_infos = []
        
        for i in range(num_ds):
            self.img_infos.extend(self.load_annotations(self.img_prefix[i], self.ann_file[i]))
        
        # tiny dataset
        if self.num_debug_imgs != -1:
            self.img_infos = self.img_infos[:int(self.num_debug_imgs)]

    def load_annotations(self, img_prefix, ann_file):
        '''
        @description: 
        @param : 
            ann_file: VOC2007/train.txt | VOC2012/train.txt
        @return: 
        '''
        img_infos = []
        img_ids = mmcv.list_from_file( osp.join(self.data_root, ann_file) )
        img_prefix = osp.join(self.data_root, img_prefix)
        for img_id in img_ids:
            filename = f'{img_prefix}/JPEGImages/{img_id}.jpg'
            xml_path = f'{img_prefix}/Annotations/{img_id}.xml'
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, xml_path=xml_path, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        '''
        @description: 
        @param :
            idx: index of sample
        @return: 
        '''
        # img_id = self.img_infos[idx]['id']
        # xml_path = osp.join(self.img_prefix, 'Annotations',
        #                     '{}.xml'.format(img_id))
        xml_path = self.img_infos[idx]['xml_path']
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]

            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        
        if self.with_difficult:            
            ann = dict(
                bboxes=np.vstack((bboxes, bboxes_ignore)).astype(np.float32),
                labels=np.append(labels, labels_ignore).astype(np.int64))
        else:
            ann = dict(
                bboxes=bboxes.astype(np.float32),
                labels=labels.astype(np.int64))
        return ann

    def encoder(self, boxes, labels):
        '''
        @description: 
        @param : 
            boxes: (tensor) [[x1, y1, x2, y2], [...]] where x1, y1, ... have been normalized by w, h
            labels: (tensor) [...]
        @return: 7x7x30
        '''
        len_encode = self.num_boxes * 5 + len(self.CLASSES)
        target = torch.zeros((self.size_grid_cell, self.size_grid_cell, len_encode))
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        cell_size = 1 / self.size_grid_cell

        # create yolo-style annotations
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1       # cx_norm / cell_size = cx / (w / self.size_grid_cell)
            delta_xy = (cxcy_sample - ij * cell_size) / cell_size

            for j in range(self.num_boxes):
                target[int(ij[1]), int(ij[0]), j*5:j*5+2] = delta_xy
                target[int(ij[1]), int(ij[0]), j*5+2:j*5+4] = wh[i]
                target[int(ij[1]), int(ij[0]), j*5+4] = 1

            # ensuring one-hot encoding of class label
            target[int(ij[1]), int(ij[0]), self.num_boxes*5:] = 0
            target[int(ij[1]), int(ij[0]), self.num_boxes*5+int(labels[i])-1] = 1
        return target

    def decoder(self, target):
        '''
        @description: 
        @param : 
            target: (tensor) size(1, 7, 7, 30)
        @return: (tensor) boxes[[x1, y1, x2, y2]] labels[...]
        '''
        # assert(target.size()[0] == 1)
        boxes = []
        labels = []
        cell_size = 1 / self.size_grid_cell
        target = target.squeeze(0)                                  # (1, 7, 7, 30) -> (7, 7, 30)
        mask = target[:, :, 4] > 0
        for i in range(self.size_grid_cell):
            for j in range(self.size_grid_cell):
                if mask[i, j]:
                    box = target[i, j, :4]                          # just ignore the 2rd box
                    xy = torch.Tensor([j, i]) * cell_size           # uppperleft of the cell
                    box[:2] = box[:2] * cell_size + xy              # return cxcy relative to image

                    box_tlbr = torch.FloatTensor(5)                 # [cx, cy, w, h] -> [x1, y1, x2, y2, prob]
                    box_tlbr[:2] = box[:2] - 0.5 * box[2:]
                    box_tlbr[2:4] = box[:2] + 0.5 * box[2:]

                    _, label = torch.max(target[i, j, 5*self.num_boxes:], 0)
                    box_tlbr[-1] = 1.

                    boxes.append(box_tlbr.view(-1, 5))
                    labels.append(torch.Tensor([label]))
                    
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,))
        else:
            boxes = torch.cat(boxes, 0)                         # (n, 4)
            labels = torch.cat(labels, 0)                       # (n,)
        return boxes, labels

    def random_flip(self, img, boxes):
        # if np.random.rand() < 0.5:
        if np.random.randint(2):
            img = mmcv.imflip(img)

            w = img.shape[1]
            flipped = boxes.copy()
            flipped[..., 0] = w - boxes[..., 2] - 1
            flipped[..., 2] = w - boxes[..., 0] - 1
            boxes = flipped
        return img, boxes
    
    def random_photo_metric_distortion(self, img):
        # https://github.com/open-mmlab/mmdetection/blob/a054aef422d650a644459bdc232248eefa8ae8b7/mmdet/datasets/extra_aug.py#L8
        # brightness_delta=32
        # contrast_range=(0.5, 1.5),
        # saturation_range=(0.5, 1.5),
        # hue_delta=18)
        # brightness_delta = photo_metric_distortion['brightness_delta']
        # contrast_lower, contrast_upper = photo_metric_distortion['contrast_range']
        # saturation_lower, saturation_upper = photo_metric_distortion['saturation_range']
        # hue_delta = self.photo_metric_distortion['hue_delta']

        brightness_delta = 32
        contrast_lower, contrast_upper = (0.5, 1.5)
        saturation_lower, saturation_upper = (0.5, 1.5)
        hue_delta = 18

        # change datatype before do photo metric distortion
        img = img.astype(np.float32)
        
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-brightness_delta, brightness_delta)
            img += delta
            img = img.clip(min=0, max=255)
                        
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(contrast_lower, contrast_upper)
                img *= alpha
                img = img.clip(min=0, max=255)

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(saturation_lower, saturation_upper)
            img[..., 1] = img[..., 1].clip(min=0, max=1)

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-hue_delta, hue_delta)
            # img[..., 0][img[..., 0] > 360] -= 360
            # img[..., 0][img[..., 0] < 0] += 360  
            img[..., 0] = img[..., 0].clip(min=0, max=360) 

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)
        
        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(contrast_lower, contrast_upper)
                img *= alpha
                img = img.clip(min=0, max=255)

        # randomly swap channels
        # if np.random.randint(2):
        #     img = img[..., np.random.permutation(3)]
        
        return img.astype(np.uint8)

    def random_scale(self, img, boxes):
        if np.random.randint(2):
            scale_h = np.random.uniform(0.8, 1.2)
            scale_w = np.random.uniform(0.8, 1.2) 
            h, w, _ = img.shape
            img = mmcv.imresize(img, (int(w*scale_w), int(h*scale_h)), return_scale=False) 
            scale_boxes = boxes.copy()
            scale_boxes *= np.tile([scale_w, scale_h], 2)
            boxes = scale_boxes
        return img, boxes
    
    def random_shift(self, img, boxes, labels):
        if np.random.randint(2):
            h, w, c = img.shape
            mean = [123.675, 116.28, 103.53]    #
            ratio = 0.4
            expand_img = np.full((int(h * ratio + h), int(w * ratio + w), c),
                                mean).astype(img.dtype)
            expand_img[int(ratio/2 * h) : int(ratio/2 * h + h), int(ratio/2 * w) : int(ratio/2 * w + w)] = img
            expand_boxes = boxes + np.tile((int(ratio/2 * w), int(ratio/2 * h)), 2)
            left = int(np.random.uniform(0, ratio * w))
            top = int(np.random.uniform(0, ratio * h))
            patch = np.array((int(left), int(top), int(left + w),
                                    int(top + h)))

            # center of boxes should inside the crop img
            center = (expand_boxes[:, :2] + expand_boxes[:, 2:]) / 2
            mask = (center[:, 0] > patch[0]) * (
                center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                    center[:, 1] < patch[3])
            if not mask.any():
                return img, boxes, labels

            boxes = expand_boxes[mask]
            labels = labels[mask]

            # adjust boxes
            img = expand_img[patch[1]:patch[3], patch[0]:patch[2]]
            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
            boxes -= np.tile(patch[:2], 2)  
        return img, boxes, labels

    def random_crop(self, img, boxes, labels):
        h, w, _ = img.shape
        if np.random.randint(2):
            new_w = np.random.uniform(0.6 * w, w)
            new_h = np.random.uniform(0.6 * h, h)

            # h / w in [0.5, 2]
            if new_h / new_w < 0.5 or new_h / new_w > 2:
                return img, boxes, labels

            left = np.random.uniform(w - new_w)
            top = np.random.uniform(h - new_h)

            patch = np.array((int(left), int(top), int(left + new_w),
                                int(top + new_h)))
                                
            # overlaps = bbox_overlaps(
            #             patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)

            # if overlaps.min() < 0.5:
            #     return img, boxes, labels

            # center of boxes should inside the crop img
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = (center[:, 0] > patch[0]) * (
                center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                    center[:, 1] < patch[3])
                    
            if not mask.any():
                return img, boxes, labels

            boxes = boxes[mask]
            labels = labels[mask]

            # adjust boxes
            img = img[patch[1]:patch[3], patch[0]:patch[2]]
            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
            boxes -= np.tile(patch[:2], 2)

        return img, boxes, labels

    def prepare_train_img(self, idx):
        img = mmcv.imread(self.img_infos[idx]['filename'])      # (H, W, 3)     
        ann = self.get_ann_info(idx)
        boxes, labels = ann['bboxes'], ann['labels']

        # data augmentation
        if self.do_augmentation:
            # extra augmentation
            img, boxes = self.random_scale(img, boxes)
            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)
            img = self.random_photo_metric_distortion(img)

            # apply transforms
            img, boxes = self.random_flip(img, boxes)
        
        # XXX ValueError: some of the strides of a given numpy array are negative.
        # img = img.copy()

        h, w, _ = img.shape
        boxes = torch.Tensor(boxes)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        target = self.encoder(boxes, labels)
        
        img = self.transform(img)
        return img, target

    def prepare_test_img(self, idx):
        img = mmcv.imread(self.img_infos[idx]['filename'])
        ann = self.get_ann_info(idx)
        boxes, labels = ann['bboxes'], ann['labels']
        boxes = torch.Tensor(boxes)
        
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        target = self.encoder(boxes, labels)

        img = self.transform(img)
        return img, target
        
    def __len__(self):
        return len(self.img_infos)
        
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
        

if __name__ == '__main__':
    vis = True
    if vis:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((448, 448)),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    train_dataset = PASCAL_VOC(
        data_root='/Users/xmhan/data/VOCdevkit', 
        img_prefix='VOC2007', 
        ann_file='VOC2007/ImageSets/Main/train.txt',
        transform=data_transform,
        size_grid_cell=14,
        with_difficult=False,
        do_augmentation=True,
        num_debug_imgs=64)

    class_names = PASCAL_VOC.CLASSES
    for i, (img, target) in enumerate(train_dataset):
        imgname = train_dataset.img_infos[i]['filename']
        print(i, imgname)

        # Image to numpy
        img = np.array(img)
        h, w, _ = img.shape

        # show pred results
        boxes, labels = train_dataset.decoder(target)
        boxes[:, :4] *= torch.Tensor([w, h, w, h])
        
        boxes = boxes.numpy().astype(np.int16)
        labels = labels.numpy().astype(np.int16)
        print(boxes, labels, (w, h))
        
        show_result(img, boxes, labels, class_names)

        # show gt results
        ann = train_dataset.get_ann_info(i)
        boxes_gt, labels_gt = ann['bboxes'], ann['labels']
        boxes_gt_ = np.ones((len(boxes_gt), 5))
        boxes_gt_[:, :4] = boxes_gt
        labels_gt -= 1

        w_gt, h_gt = train_dataset.img_infos[i]['width'], train_dataset.img_infos[i]['height']
        print(boxes_gt_, labels_gt, (w_gt, h_gt))
        show_result(imgname, boxes_gt_, labels_gt, class_names, bbox_color='red', text_color='red')