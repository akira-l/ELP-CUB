import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F

import pdb

class ConDisModel(nn.Module):
    def __init__(self, config):
        super(ConDisModel, self).__init__()

        self.num_classes = config.numcls
        self.relu = nn.ReLU(inplace=True)
        self.trans_layer = nn.Sequential(
                                         nn.Linear(100, 64), 
                                         nn.ReLU(),
                                         nn.Linear(64, 32), 
                                         nn.ReLU(),
                                         nn.Linear(32, 4), 
                                         nn.ReLU()
                                       )
        self.trans_classifier = nn.Linear(2048*4, self.num_classes, bias=False)


    def forward(self, feat):
        bs = feat.size(0)
        down_feat = self.trans_layer(feat)
        cls_feat = down_feat.view(bs, -1)
        mem_cls = self.trans_classifier(cls_feat)
        return mem_cls 


