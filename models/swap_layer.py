import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import pdb


class SwapLayer(nn.Module):
    def __init__(self, nei_num, size):
        super(SwapLayer, self).__init__()
        self.nei_num = nei_num
        self.map_size = size
        self.cuda_flag = True
       
        self.shuffle_emb, self.shuffle_pos, self.shuffle_ang = self.init_layers() 
        
    def init_layers(self):
        ch_ = 2048
        fc_dim = 32*4
        shuffle_emb_layer = nn.Sequential(
                                           nn.Conv2d(ch_, 512, 1, 1), 
                                           nn.ReLU(),
                                           nn.Conv2d(512, 128, 1, 1), 
                                           nn.ReLU(), 
                                           nn.Conv2d(128, 32, 1, 1), 
                                           nn.ReLU(), 
                                          )
        shuffle_pos = nn.Linear(fc_dim, 1)
        shuffle_ang = nn.Linear(fc_dim, 1)
        return shuffle_emb_layer, shuffle_pos, shuffle_ang

    def forward(self, feat, swap_map):
        bs_, ch_, feat_wid, feat_hei = feat.size()
        assert feat.dim() == 4 and feat.size(2) % self.map_size[0] == 0 and feat.size(3) % self.map_size[1] == 0
        hor_step = feat.size(2) // self.map_size[0]
        ver_step = feat.size(3) // self.map_size[1]
        reshape_feat = feat.view(bs_, ch_, self.map_size[0], hor_step, self.map_size[1], ver_step).transpose(3, 4).contiguous()
        reshape_feat = reshape_feat.view(bs_, ch_, self.map_size[0]*self.map_size[1], -1)
        shuffle_gather = []
        for sub_bs in range(bs_):
            shuffle_gather.append(reshape_feat[sub_bs, :, swap_map[sub_bs].long(), :])
        shuffle_feat = torch.stack(shuffle_gather) 
        #shuffle_feat = reshape_feat[:, :, swap_map.long(), :]

        shuffle_out = self.shuffle_emb(shuffle_feat)
        re_shuffle_out = shuffle_out.transpose(1, 2).contiguous().view(bs_, self.map_size[0]*self.map_size[1], -1)
        pos_out = self.shuffle_pos(re_shuffle_out).squeeze(2)
        ang_out = self.shuffle_ang(re_shuffle_out).squeeze(2)

        normal_out = self.shuffle_emb(reshape_feat)
        re_normal_out = normal_out.transpose(1, 2).contiguous().view(bs_, self.map_size[0]*self.map_size[1], -1) 
        npos_out = self.shuffle_pos(re_normal_out).squeeze(2)
        nang_out = self.shuffle_ang(re_normal_out).squeeze(2)
         
        return shuffle_feat.view(bs_, ch_, feat_wid, feat_hei), pos_out, ang_out, npos_out, nang_out
        

        
