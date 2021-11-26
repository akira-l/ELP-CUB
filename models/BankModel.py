import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F

import pdb

class BankModel(nn.Module):
    def __init__(self, config):
        super(BankModel, self).__init__()

        self.use_dcl = config.use_dcl
        self.use_fpn = config.use_fpn
        self.use_Asoftmax = config.use_Asoftmax
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        print(self.backbone_arch)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.relu = nn.ReLU(inplace=True)

        self.trans_layer = nn.Sequential(
                                         nn.Linear(50, 32), 
                                         nn.ReLU(),
                                         nn.Linear(32, 8), 
                                         nn.ReLU(),
                                         nn.Linear(8, 1),
                                         nn.ReLU()
                                       )
        self.trans_classifier = nn.Linear(2048, self.num_classes, bias=False)



    def forward(self, mem_feat):
        bs = mem_feat.size(0)
        mem_down_feat = self.trans_layer(mem_feat)
        mem_cls_feat = mem_down_feat.view(bs, -1)
        mem_cls = self.trans_classifier(mem_cls_feat)
        return mem_cls 


