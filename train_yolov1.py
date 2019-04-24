import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from warmup_scheduler import GradualWarmupScheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from datasets.pascal_voc import PASCAL_VOC
from models.backbone import resnet50_yolov1
from criterions.yololoss import YoloV1Loss
from utils.visualize import Visualizer
from utils.utils import get_logger, get_learning_rate
from utils.voc_eval import calc_map

import os
import os.path as osp
import time
import random
import numpy as np
import argparse
import yaml

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--trial_log', default='voc07_moreaug_7x7')
parser.add_argument('--config', default='configs/config.yaml')
parser.add_argument('--resume', default=False, help='resume')
args = parser.parse_args()


def main():
    global args
    args = parser.parse_args()

    workpath = osp.abspath(osp.dirname(__file__))
    with open(osp.join(workpath, args.config)) as f:
        if yaml.__version__ == '5.1':
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # seed settings
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # logger and checkpoint settings
    model_path = osp.join(workpath, 'checkpoints', args.trial_log)
    if not osp.exists(model_path):
        os.makedirs(model_path)
    logger = get_logger(args.trial_log, model_path)
    logger.info(f'args: {args}')

    # model settings
    model = resnet50_yolov1(pretrained=True)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = torch.load(osp.join(model_path, 'latest.tar'))
        except:
            raise FileNotFoundError

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        logger.debug(f'Use {num_gpus} GPUs!')
        model = nn.DataParallel(model)
        args.batch_size *= num_gpus
    model.to(device)

    # model statistics
    summary(model, (3, args.img_size, args.img_size))

    # dataset settings
    data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    # load training dataset
    train_dataset = PASCAL_VOC(
        # data_root='/Users/xmhan/data/VOCdevkit',
        data_root='/data/datasets/VOCdevkit',
        # img_prefix=['VOC2007'],
        # ann_file=['VOC2007/ImageSets/Main/train.txt'],
        img_prefix=['VOC2007', 'VOC2012'],
        ann_file=['VOC2007/ImageSets/Main/trainval.txt', 'VOC2012/ImageSets/Main/trainval.txt'],
        transform=data_transform,
        size_grid_cell=args.size_grid_cell,
        with_difficult=args.with_difficult,
        do_augmentation=args.do_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # load validation/testing dataset
    val_dataset = PASCAL_VOC(
        # data_root='/Users/xmhan/data/VOCdevkit',
        data_root='/data/datasets/VOCdevkit',
        img_prefix='VOC2007', 
        ann_file='VOC2007/ImageSets/Main/test.txt',
        transform=data_transform,
        size_grid_cell=args.size_grid_cell,
        with_difficult=args.with_difficult,
        test_mode=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info('training dataset: {}'.format(len(train_dataset)))
    logger.info('validation dataset: {}'.format(len(val_dataset)))
    dataloaders = {'train': train_loader, 'val': val_loader}

    # loss function
    criterion = YoloV1Loss(device, args.size_grid_cell, 
        args.num_boxes, args.num_classes, args.lambda_coord, args.lambda_noobj)
    train_model(model, criterion, optimizer, dataloaders, val_dataset, model_path, start_epoch, logger, device)


def train_model(model, criterion, optimizer, dataloaders, val_dataset, model_path, start_epoch, logger, device):
    since = time.time()
    best_loss = np.inf
    best_map = 0
    iter_num = 0
    trial_log = args.trial_log
    num_epochs = args.num_epochs
    test_interval = args.test_interval
    burn_in = args.burn_in
    lr = args.learning_rate
    lr_steps = args.lr_steps
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    nms_thresh = args.nms_thresh
    vis = Visualizer(env=trial_log)
        
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch {} / {}'.format(epoch+1, num_epochs))
        logger.info('-' * 64)

        # set learning rate manually
        if epoch in lr_steps:
            lr *= 0.1

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            total_loss = 0.0
            # Iterate over data.
            for i, (inputs, targets) in enumerate(dataloaders[phase]):
                # warmming up of the learning rate
                if phase == 'train':
                    if iter_num < args.burn_in:
                        burn_lr = get_learning_rate(iter_num, lr, burn_in)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = burn_lr
                        iter_num += 1
                    else:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss, obj_coord_loss, obj_conf_loss, noobj_conf_loss, obj_class_loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                total_loss += loss.item()

                if phase == 'train':
                    cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
                    vis.plot('cur_lr', cur_lr)
                    logger.info('Epoch [{}/{}], iter [{}/{}], lr: {:g}, loss: {:.4f}, average_loss: {:.4f}'.format(
                        epoch+1, args.num_epochs, i+1, len(dataloaders[phase]), cur_lr, loss.item(), total_loss/(i+1)))
                    logger.debug('  obj_coord_loss: {:.4f}, obj_conf_loss: {:.4f}, noobj_conf_loss: {:.4f}, obj_class_loss: {:.4f}'.format(
                        obj_coord_loss, obj_conf_loss, noobj_conf_loss, obj_class_loss))
                    vis.plot('train_loss', total_loss/(i+1))

                    # save model for inferencing
                    torch.save(model.state_dict(), osp.join(model_path, 'latest.pth'))

                    # save model for resuming training process
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, osp.join(model_path, 'latest.tar'))

            # evaluate latest model
            if phase == 'val':
                current_loss = total_loss / (i+1)
                if best_loss > current_loss:
                    best_loss = current_loss
                logger.info('current val loss: {:.4f}, best val Loss: {:.4f}'.format(current_loss, best_loss))
                vis.plot('val_loss', total_loss/(i+1))

                if epoch and epoch % (test_interval-1) == 0:
                    current_map = calc_map(logger, val_dataset, model_path, conf_thresh, iou_thresh, nms_thresh)
                    # save the best model as so far
                    if best_map < current_map:
                        best_map = current_map
                        torch.save(model.state_dict(), osp.join(model_path, 'best.pth'))
                    logger.info('current val map: {:.4f}, best val map: {:.4f}'.format(current_map, best_map))
                    vis.plot('val_map', current_map)

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Optimization Done.')

if __name__ == '__main__':
    main()