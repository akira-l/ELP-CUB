import os
import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
#import pretrainedmodels

from models.Asoftmax_linear import AngleLinear
from models.swap_layer import SwapLayer
from config import pretrained_model
from utils.rhm_map import rhm

import pdb

class AsModel(nn.Module):
    def __init__(self, config):
        super(AsModel, self).__init__()
         
        self.dim = config.bank_dim
        self.num_classes = config.numcls
        
        self.trans_layer = nn.Sequential(
                                         nn.Linear(50, 32), 
                                         nn.ReLU(),
                                         nn.Linear(32, 8), 
                                         nn.ReLU(),
                                         nn.Linear(8, 1),
                                         nn.ReLU()
                                       )
        self.as_layer = AngleLinear(2048, self.num_classes)


    def forward(self, feat):
        bs = feat.size(0)
        trans_feat = self.trans_layer(feat)
        trans_feat_view = trans_feat.view(bs, -1) 
        out = self.as_layer(trans_feat_view)
        return out






