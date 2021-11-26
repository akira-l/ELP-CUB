import os
import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
#import pretrainedmodels

import pdb

class MapModel(nn.Module):
    def __init__(self, config):
        super(MapModel, self).__init__()

        self.dim = config.bank_dim
        self.structure_bank_num_max = config.otmap_struct_max 
        self.map_linear_1 = nn.Sequential( 
                                           nn.Linear(self.dim, 512), 
                                           nn.Linear(512, 128), 
                                           nn.Linear(128, 32), 
                                        )
        self.map_linear_2 = nn.Linear(32*self.structure_bank_num_max*2, 2)


    def forward(self, cat_feat):
        bs = cat_feat.size(0)
        x = self.map_linear_1(cat_feat)
        x_view = x.view(bs, -1)
        out = self.map_linear_2(x_view)
        return out



