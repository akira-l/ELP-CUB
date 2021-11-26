from __future__ import print_function, division, absolute_import
import argparse
import os
import io
import shutil
import time
from tqdm import tqdm

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


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image_path, input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            rate = get_learning_rate(optimizer)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Rate:{rate:.3f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                epoch, i, len(train_loader), rate=rate, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3))
    return top1.val, top3.val, losses.val

def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (image_path, input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # prec1, prec3 = accuracy(output.data, target.data, topk=(1, 3))
            prec1, prec3 = accuracy(output.data, target.data, topk=(1, 25))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))

        print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))
        return top1.avg, top3.avg, losses.avg

#def adjust_learning_rate(optimizer, epoch):
#    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#    #lr = args.lr * (0.1 ** (epoch // 30))
#    lr = args.lr * (0.1 ** (epoch // 15))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch):
    lr = 1e-2 # warm-up
    if epoch > 8:
        lr = 1e-3
    if epoch > 15:
        lr = 1e-4
    if epoch > 20:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, output_dir):
    filename = os.path.join(output_dir, 'checkpoint', 'checkpoint_epoch_%03d_prec1_%0.3f_pth.tar'%(state['epoch'], state['prec1']))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir,'model_best.pth.tar'))



def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    #output_dir = os.path.join('result', args.arch)
    output_dir = args.output
    os.makedirs(output_dir + '/checkpoint', exist_ok=True)

    # create model
    # print("=> creating model '{}'".format(args.arch))
    # if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
    #     print("=> using pre-trained parameters '{}'".format(args.pretrained))
    #     model = pretrainedmodels.__dict__[args.arch](num_classes=1000,
    #                                                  pretrained=args.pretrained)
    # else:
    #     model = pretrainedmodels.__dict__[args.arch]()

    # model.last_linear = nn.Linear(model.last_linear.in_features, 2019)

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

    print(model)

    cudnn.benchmark = True
    train_augment = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    val_augment = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    train_dataset = ProductDataset(train_augment,'train_part2.txt', 'train')
    train_loader = DataLoader(
                        train_dataset,
                        sampler     = RandomSampler(train_dataset),
                        batch_size  = args.batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)
    val_dataset = ProductDataset(val_augment,'val.txt', 'val')
    val_loader = DataLoader(
                        val_dataset,
                        sampler     = RandomSampler(val_dataset),
                        batch_size  = args.batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_prec1, train_prec3, train_loss=train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec3, val_loss = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'prec1': prec1
        }, is_best, output_dir)

        with io.open(os.path.join(output_dir, 'train.log'), 'a') as f:
            f.write('\t'.join([str(epoch), str(train_loss), str(train_prec1), str(train_prec3),\
		str(val_loss), str(prec1), str(prec3)])+'\n')


if __name__ == '__main__':
    main()



