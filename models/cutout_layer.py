import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

class CutoutLayer(nn.Module):
    def __init__(self, ratio, size):
        super(CuotoutLayer, self).__init__()
        self.drop_num = int(ratio * size[0]*size[1])
        assert self.drop_num > 0
        self.map_size = size

        self.fix_layer = self.init_fix_layer()

    def init_fix_layer(self):
        ch_ = 2048
        fix_layer = nn.Sequential(
                                  nn.Conv2d(ch_, ch_, 3, 3, padding=1), 
                                  nn.ReLU(),
                                  nn.Conv2d(ch_, ch_, 3, 3, padding=1), 
                                  nn.ReLU(),
                                 ) 
        return fix_layer

    def drop_pos(self):
        map_len = size[0] * size[1]
        rand_ind = list(range(map_len))
        random.shuffle(rand_ind)
        get_ind = rand_ind[:self.drop_num]
        ind_tensor = torch.FloatTensor(get_ind)
        ind_x = ind_tensor // size[0]
        ind_y = ind_tensor % size[0]
        return ind_x, ind_y
        
    def forward(self, feat, drop=True):
        if drop:
            pos_x, pos_y = self.drop_pos()
            drop_feat = feat[:, :, pos_x, pos_y]
            feat[:, :, pos_x, pos_y] = 0.0
            feat = self.fix_layer(feat)
            fix_feat = feat[:, :, pos_x, pos_y]
            return feat, drop_feat, fix_feat
        else:
            return feat, None, None
 
         
