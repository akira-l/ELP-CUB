#coding=utf-8
import os
import datetime
import argparse
import logging
import pandas as pd

import torch
import torch.nn as nn
from  torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP

from transforms import transforms
from utils.train_model import train
from utils.utils import setup_seed
#from utils.warmup_scheduler import GradualWarmupScheduler
from models.LoadModel import MainModel
from models.avg_mem import AvgMem
from models.cosine_model import CosSoftmaxModule
#from models.contrast_discriminator import ConDisModel
from config import LoadConfig, load_data_transformers
from dataset.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR
from models.models import classifier32, classifier32ABN

print(torch.__version__)

import pdb

#os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,5,6'

#def setup(rank, world_size):
#    os.environ['MASTER_ADDR'] = 'localhost'
#    os.environ['MASTER_PORT'] = '1234'
#
#    dist.init_process_group("gloo", rank=rank, world_size=world_size)
#
#    torch.manual_seed(42)
#
#def cleanup():
#    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='inat18', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, #'./CUB_base_87.5/weights_264_187_0.8550_0.9465.pth',
                        type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--not_use_dcl', dest='not_use_dcl',
                        action='store_false')
    parser.add_argument('--epoch', dest='epoch',
                        default=100, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=512, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=256, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.2, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=8, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=8, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=256, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=224, type=int)
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    parser.add_argument('--weighted', dest='weighted_sample',
                        action='store_true')
    parser.add_argument('--buff2', dest='buff_drop2',
                        action='store_true')
    args = parser.parse_args()
    return args

def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)


if __name__ == '__main__':
    args = parse_args()
    #setup_seed(20)
    print('args: \n', args, '\n\n', '========'*6)
    Config = LoadConfig(args, 'train')
    print('Config: \n', vars(Config), '\n\n', '========'*6)


    Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    cudnn.benchmark = True

    print('Choose model and train set')
    #fpn not implemented
    model = MainModel(Config)
    model = classifier32(num_classes=20)
    print('-'*10, 'main model', '-'*10)
    print(model)
    avg_mem = AvgMem(Config, Config.mem_m, 'all')
    as_model = CosSoftmaxModule(Config, ver='normal')
    #as_model = CosSoftmaxModule(Config, ver='con')
    print('-'*10, 'as model', '-'*10)
    print(as_model)
    print('-'*25)

    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...')
    else:
        if not args.resume is None:
            resume = args.resume
            #print('load from pretrained checkpoint %s ...'% resume)
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...'%resume)
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir')
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s'%(args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    model.cuda()
    model = nn.DataParallel(model)
    as_model.cuda()
    as_model = nn.DataParallel(as_model)


    # set new layer's lr
    ignored_cls = list(map(id, model.module.classifier.parameters()))
    ignored_params = ignored_cls
#
    print('the num of new layers:', len(ignored_params))
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
    simple_cls_ratio = 10
    optimizer = optim.SGD([{'params': base_params},
                               {'params': model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},

                              ], lr = base_lr, momentum=0.9)
    filter_optim = optim.SGD([
                               {'params': as_model.module.parameters(), 'lr': 0.01},

                              ], lr = base_lr, momentum=0.9)

    #exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[120, 180, 240, 300], gamma=0.1)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    filter_exp_lr_scheduler = lr_scheduler.MultiStepLR(filter_optim, milestones=[120, 180, 240, 300], gamma=0.1)


    optim_dict = {}
    optim_dict['common'] = optimizer
    optim_dict['filter'] = filter_optim

    sch_dict = {}
    sch_dict['common'] = exp_lr_scheduler
    sch_dict['filter'] = filter_exp_lr_scheduler

    models_transfer_dict = {}
    models_transfer_dict['base'] = model
    models_transfer_dict['align'] = avg_mem
    models_transfer_dict['bank'] = as_model


    train(Config,
          models_transfer_dict,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer_recieve=optim_dict,
          scheduler_recieve=sch_dict,#warmup_scheduler,
          data_loader=dataloader,
          save_dir=save_dir,
          data_ver='allw',
          data_size=args.crop_resolution,
          savepoint=args.save_point,
          checkpoint=args.check_point)


