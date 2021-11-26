from __future__ import print_function, division, absolute_import
import argparse
import os
import io
import shutil
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import *

import sys

from nn_models.backbone import ProductNet
from utils.metric import accuracy, AverageMeter
from utils.data import ProductDataset
from utils.loss import FocalLoss
from utils.utils import get_learning_rate
import pretrainedmodels
import pretrainedmodels.utils
from utils.autoaugment import ImageNetPolicy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Product Training')
parser.add_argument('--data', metavar='DIR', default="path_to_imagenet",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='se_resnet50', #nasnetamobile
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=496, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--do-not-preserve-aspect-ratio',
                    dest='preserve_aspect_ratio',
                    help='do not preserve the aspect ratio when resizing an image',
                    action='store_false')
parser.add_argument('--output', default=os.path.join('result', 'se_resnet50_part2'))
parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0

def clean(val_loader, model):
    topx = AverageMeter()
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        with open('input/part2_clean_part1_3.txt', 'w') as f:
            for i, (image_path, input, target) in enumerate(tqdm(val_loader)):
                target = target.cuda()
                input = input.cuda()
                # compute output
                output = model(input)

                k = 3
                _, pred = output.topk(k, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                correct_img = correct.sum(0)
                mask =  torch.le(correct_img, 0).cpu().numpy()
                for ind, value in enumerate(mask):
                    if value == 1:
                        f.write(image_path[ind]+'\n')
    #            correct_k = correct[:k].view(-1).float().sum(0)
    #            correct_k = correct_k.mul_(100.0 / batch_size)
    #            topx.update(correct_k.item(), batch_size)
#    print(topx.avg)

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    #output_dir = os.path.join('result', args.arch)
    output_dir = args.output
    os.makedirs(output_dir + '/checkpoint', exist_ok=True)

    model = ProductNet(args.arch, args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # print(model)

    cudnn.benchmark = True
    val_augment = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    val_dataset = ProductDataset(val_augment,'train_part1.txt', 'val')
    #val_dataset = ProductDataset(val_augment,'val.txt', 'val')
    val_loader = DataLoader(
                        val_dataset,
                        sampler     = RandomSampler(val_dataset),
                        batch_size  = args.batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)

    # define loss function (criterion) and optimizer

    clean(val_loader, model)
    return


if __name__ == '__main__':
    main()



