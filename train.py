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
from utils.util import get_logger, get_learning_rate, adjust_learning_rate
from utils.voc_eval import calc_map

import os
import os.path as osp
import time
import numpy as np
import argparse
import yaml

parser = argparse.ArgumentParser(
    description='Pytorch YoloV1 Training')
parser.add_argument('--trial_log', default='voc07+12_weight')
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
        else:
            config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # seed settings
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # logger and checkpoint settings
    model_path = osp.join(workpath, 'checkpoints', args.trial_log)
    if not osp.exists(model_path):
        os.makedirs(model_path)
    logger = get_logger(args.trial_log, model_path)
    logger.info(f'args: {args}')

    # model settings
    model = resnet50_yolov1(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        logger.debug(f'Use {num_gpus} GPUs!')
        model = nn.DataParallel(model)
        
        # adjust `batch_size` and `burn_in` 
        args.batch_size *= num_gpus
        args.burn_in /= num_gpus
        # args.learning_rate *= num_gpus
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 0
    iter_num = 0
    if args.resume:
        try:
            checkpoint = torch.load(osp.join(model_path, 'latest.tar'))
        except:
            raise FileNotFoundError

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        iter_num = checkpoint['iter_num']

    # model statistics
    summary(model, input_size=(3, args.img_size, args.img_size), batch_size=args.batch_size)

    # dataset settings
    data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    # load training dataset
    img_prefixs = args.img_prefix if isinstance(args.img_prefix, list) else [args.img_prefix]
    train_dataset = PASCAL_VOC(
        data_root=args.data_root,
        img_prefix=img_prefixs,
        ann_file=[f'{img_prefix}/ImageSets/Main/trainval.txt' for img_prefix in img_prefixs],
        transform=data_transform,
        size_grid_cell=args.size_grid_cell,
        with_difficult=args.with_difficult,
        do_augmentation=args.do_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # load validation/testing dataset
    val_dataset = PASCAL_VOC(
        data_root=args.data_root,
        img_prefix='VOC2007', 
        ann_file='VOC2007/ImageSets/Main/test.txt',
        transform=data_transform,
        size_grid_cell=args.size_grid_cell,
        with_difficult=args.with_difficult,
        test_mode=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info('training dataset: {}'.format(len(train_dataset)))
    logger.info('validation dataset: {}'.format(len(val_dataset)))
    dataloaders = {'train': train_loader, 'val': val_loader}

    # loss function
    criterion = YoloV1Loss(device, args.size_grid_cell, 
        args.num_boxes, args.num_classes, args.lambda_coord, args.lambda_noobj)
    train_model(model, criterion, optimizer, dataloaders, model_path, start_epoch, iter_num, logger, device)


def train_model(model, criterion, optimizer, dataloaders, model_path, start_epoch, iter_num, logger, device):
    since = time.time()
    best_loss = np.inf
    best_map = 0
    trial_log = args.trial_log
    num_epochs = args.num_epochs
    test_interval = args.test_interval
    burn_in = args.burn_in
    lr = args.learning_rate
    lr_steps = args.lr_steps
    size_grid_cell = args.size_grid_cell
    num_boxes = args.num_boxes
    num_classes = args.num_classes
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    nms_thresh = args.nms_thresh
    port = args.port
    vis = Visualizer(env=trial_log, port=port)
        
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch {} / {}'.format(epoch+1, num_epochs))
        logger.info('-' * 64)

        # set learning rate manually
        if epoch in lr_steps:
            lr *= 0.1
        adjust_learning_rate(optimizer, lr)

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
                        adjust_learning_rate(optimizer, burn_lr)
                        iter_num += 1
                    else:
                        adjust_learning_rate(optimizer, lr)
                    
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

            # save model for inferencing and resuming training process
            if phase == 'train':
                torch.save(model.state_dict(), osp.join(model_path, 'latest.pth'))
                torch.save({
                    'iter_num: ': iter_num,
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

                if epoch < 10 or (epoch+1) % test_interval == 0:
                    current_map = calc_map(logger, dataloaders[phase].dataset, model_path, 
                        size_grid_cell, num_boxes, num_classes, conf_thresh, iou_thresh, nms_thresh)
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