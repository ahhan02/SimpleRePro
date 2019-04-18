#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: xmhan
@LastEditors: xmhan
@Date: 2019-04-08 18:22:45
@LastEditTime: 2019-04-18 18:56:51
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from yololoss import YoloV1Loss
from yolodataset import YoloV1DatasetVOC
from models.backbone import resnet50_yolov1
from utils.visualize import Visualizer
from voc_eval import calc_map

import os
import os.path as osp
import time
import random
import numpy as np
import logging

trial_log = 'voc07+12_aug'
workpath = osp.join(osp.dirname(__file__), 'checkpoints')
model_path = osp.join(osp.abspath(workpath), trial_log)
if not osp.exists(model_path):
    os.makedirs(model_path)

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

# set seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# hyper-parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
num_epochs = 135
batch_size = 16
num_boxes = 2
size_grid_cell = 7
img_size = 448
burn_in = 1000
num_debug_imgs = None
model = resnet50_yolov1(pretrained=True)

if torch.cuda.device_count() > 1:
    num_gpus = torch.cuda.device_count()
    logger.debug(f'Use {num_gpus} GPUs!')
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
    batch_size *= num_gpus

model.to(device)
summary(model, (3, 448, 448))

data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

photo_metric_distortion = dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18)    

# load training dataset
train_dataset = YoloV1DatasetVOC(
    # data_root='/Users/xmhan/data/VOCdevkit',
    data_root='/data/data/VOCdevkit',
    # img_prefix='VOC2007', 
    # ann_file='VOC2007/ImageSets/Main/train.txt',
    img_prefix=['VOC2007', 'VOC2012'],
    ann_file=['VOC2007/ImageSets/Main/trainval.txt', 'VOC2012/ImageSets/Main/trainval.txt'],
    transform=data_transform,
    with_difficult=False,
    flip_ratio=0.5,
    photo_metric_distortion=photo_metric_distortion,
    num_debug_imgs=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = YoloV1DatasetVOC(
    # data_root='/Users/xmhan/data/VOCdevkit',
    data_root='/data/data/VOCdevkit',
    img_prefix='VOC2007', 
    ann_file='VOC2007/ImageSets/Main/test.txt',
    transform=data_transform,
    with_difficult=False,
    test_mode=True,
    num_debug_imgs=None)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

logger.info('training dataset: {}'.format(len(train_dataset)))
logger.info('validation dataset: {}'.format(len(val_dataset)))
dataloaders = {'train': train_loader, 'val': val_loader}

# optimizer
criterion = YoloV1Loss()
# optimizer = optim.SGD(model.parameters(), lr=0.001/10, momentum=0.9)
# scheduler_multistep = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)  
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, 
#     after_scheduler=scheduler_multistep)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
def get_learning_rate(iter_num, base_lr=0.001, burn_in=1000, power=4):
    if iter_num < burn_in:
        return base_lr * pow(iter_num / burn_in, power)

# def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
def train_model(model, criterion, optimizer, learning_rate=0.001, burn_in=1000, num_epochs=50):
    since = time.time()
    best_loss = np.inf
    best_map = 0
    iter_num = 0

    vis = Visualizer(env=trial_log)
        
    for epoch in range(num_epochs):
        logger.info('Epoch {} / {}'.format(epoch+1, num_epochs))
        logger.info('-' * 64)

        # set learning rate manually
        if epoch == 75 or epoch == 105:
            learning_rate *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # running_loss = 0.0
            total_loss = 0.0

            # Iterate over data.
            for i, (inputs, targets) in enumerate(dataloaders[phase]):
                if phase == 'train':
                    if iter_num < burn_in:
                        wu_lr = get_learning_rate(iter_num, learning_rate, burn_in)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = wu_lr
                        iter_num += 1
                    else:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                    
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # loss = criterion(outputs, targets)
                    loss, obj_coord_loss, obj_conf_loss, noobj_conf_loss, obj_class_loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # running_loss += loss.item() * inputs.size()[0]
                # running_loss += 0.99 * running_loss+ 0.01 * loss.item()
                total_loss += loss.item()

                if phase == 'train':
                    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                    vis.plot('current_lr', current_lr)
                    logger.info('Epoch [{}/{}], iter [{}/{}], lr: {:g}, loss: {:.4f}, average_loss: {:.4f}'.format(
                        epoch+1, num_epochs, i+1, len(train_loader), current_lr, loss.item(), total_loss/(i+1)))
                    logger.debug('  obj_coord_loss: {:.4f}, obj_conf_loss: {:.4f}, noobj_conf_loss: {:.4f}, obj_class_loss: {:.4f}'.format(
                        obj_coord_loss, obj_conf_loss, noobj_conf_loss, obj_class_loss))
                    vis.plot('train_loss', total_loss/(i+1))

            # deep copy the model
            if phase == 'val':
                current_loss = total_loss / (i+1)
                if best_loss > current_loss:
                    best_loss = current_loss
                    # torch.save(model.state_dict(), osp.join(model_path, 'best.pth'))
                logger.info('current val loss: {:.4f}, best val Loss: {:.4f}'.format(current_loss, best_loss))
                vis.plot('val_loss', total_loss/(i+1))

                torch.save(model.state_dict(), osp.join(model_path, f'epoch_{epoch}.pth'))
                os.system('ln -sf {} {}'.format(osp.join(model_path, f'epoch_{epoch}.pth'), osp.join(model_path, 'latest.pth')))

                current_map = calc_map(logger, val_dataset, trial_log)
                if best_map < current_map:
                    best_map = current_map
                    # torch.save(model.state_dict(), osp.join(model_path, 'best.pth'))
                    os.system('ln -sf {} {}'.format(osp.join(model_path, f'epoch_{epoch}.pth'), osp.join(model_path, 'best.pth')))

                logger.info('current val map: {:.4f}, best val map: {:.4f}'.format(current_map, best_map))
                vis.plot('val_map', current_map)

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Optimization Done.')

# train_model(model, criterion, optimizer, scheduler_warmup, num_epochs=50)
# train_model(model, criterion, optimizer, scheduler_multistep, num_epochs=50)
train_model(model, criterion, optimizer, learning_rate, burn_in, num_epochs)