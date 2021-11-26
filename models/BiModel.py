import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
#import pretrainedmodels

from models.Asoftmax_linear import AngleLinear
from models.swap_layer import SwapLayer
from config import pretrained_model

import pdb

class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.use_fpn = config.use_fpn
        self.use_Asoftmax = config.use_Asoftmax
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            raise Exception('no pretrainedmodels package now')
            #self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)

        if self.backbone_arch == 'resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU() 

        self.swap_layer = SwapLayer(9, [7, 7])
        self.cat_layer = nn.Conv2d(2*2048, 2048, 1, stride=1, padding=0, bias=False)
         

    def forward(self, x, swap_map=None):
        x = self.model(x)
        #x = self.relu(x)
        if swap_map is not None:
            swap_feat, pred_dis, pred_ang, pred_n_dis, pred_n_ang = self.swap_layer(x, swap_map)
            #cls_feat = self.relu(swap_cat)
            cls_feat = x
        else:
            cls_feat = x
            pred_dis = None
            pred_ang = None 
            pred_n_dis = None
            pred_n_ang = None
        
        x = self.avgpool(cls_feat)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        if swap_map is None:
            swap_out = None
        else:
            sw = self.avgpool(swap_feat)
            sw = sw.view(sw.size(0), -1) 
            swap_out = self.classifier(sw)
        return out, swap_out, pred_dis, pred_ang, pred_n_dis, pred_n_ang




