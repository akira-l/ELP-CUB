import os, sys
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms, datasets
import torch.nn.functional as F

from utils.procrustes_util import transfer_coord 

import pdb


class StructureMem(nn.Module):
    def __init__(self, config):
        super(StructureMem, self).__init__()

        self.num_classes = config.numcls
        self.otmap_thresh = config.otmap_thresh
        self.context_bank_num = 5
        self.structure_bank_num_max = config.otmap_struct_max 
        self.dim = config.st_bank_dim
        self.forget_para = 0.8
        self.st_map_size = config.st_map_size

        self.feat_bank = torch.zeros(self.num_classes, self.dim, self.structure_bank_num_max).cuda()
        self.bank_confidence_transport = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()
        self.bank_confidence = torch.zeros(self.num_classes).cuda()
        self.structure_bank = torch.zeros(self.num_classes, self.structure_bank_num_max, self.dim).cuda()
        self.bank_position = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()

        self.update_feat_bank = torch.zeros(self.num_classes, self.dim, self.structure_bank_num_max).cuda()
        self.update_bank_confidence_transport = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()
        self.update_bank_confidence = torch.zeros(self.num_classes).cuda()
        self.update_bank_position = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()

        self.debug_img_list = ['' for x in range(200)]

        self.cos_sim = nn.CosineSimilarity(dim=2)
        self.cos_sim_1 = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.debug = False
        self.debug_save_num = 0

        self.feat_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.feat_pooling = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.feature = None
        self.gradient = None
        self.handlers = []

        self.relu = nn.ReLU() 



    def update_bank(self):
        self.feat_bank = self.update_feat_bank
        self.bank_confidence_transport = self.update_bank_confidence_transport
        self.bank_confidence = self.update_bank_confidence 
        self.bank_position = self.update_bank_position



    def operate_single_procrustes(self, pred_feat, bank_feat, st_position):
        re_candi = pred_feat.permute(1, 0) 
        re_bank = bank_feat.permute(1, 0)
        gather_sim_pred = []
        bank_size = re_bank.size(0)
        for cnt in range(bank_size): 
            sub_sim = self.cos_sim_1(re_bank[cnt].unsqueeze(0), re_candi)
            _, pos = torch.topk(sub_sim, 1)
            gather_sim_pred.append(pos.item())
        pred_pos = np.array(gather_sim_pred) 
        pred_pos_x = pred_pos // self.st_map_size 
        pred_pos_x = np.expand_dims(pred_pos_x, 1)
        pred_pos_y = pred_pos % self.st_map_size
        pred_pos_y = np.expand_dims(pred_pos_y, 1)
        pred_pos_coord = np.concatenate([pred_pos_x, pred_pos_y], 1)

        st_pos_x = st_position // self.st_map_size
        st_pos_y = st_position % self.st_map_size 
        st_pos_coord = torch.stack([st_pos_x, st_pos_y]).cpu()
        st_pos_coord = st_pos_coord.permute(1, 0).numpy()
        if pred_pos_coord.sum() <= 0 or st_pos_coord.sum() <= 0:
            return None

        remap_pred_coord = transfer_coord(pred_pos_coord, st_pos_coord, self.structure_bank_num_max ) 
        aligned_pred_coord = remap_pred_coord[:, 0] * self.st_map_size + remap_pred_coord[:, 1]
        max_size = self.st_map_size * self.st_map_size - 1 
        if (aligned_pred_coord > max_size).any() or (aligned_pred_coord < 0).any() or np.isnan(aligned_pred_coord).any():
            return None
        aligned_pred_coord = np.clip(aligned_pred_coord, 0, max_size)
        return aligned_pred_coord
        
        


    def structure_forward(self, feat, label, return_map=True):
        dim, hei, wei = feat.size()
        feat_view = feat.view(dim, -1)

        aligned_coord = self.operate_single_procrustes(feat_view, self.feat_bank[label], self.bank_position[label]) 
        if aligned_coord is None:
            return aligned_coord
        aligned_coord = torch.from_numpy(aligned_coord)
        return feat_view[:, aligned_coord.long()]



        

    def proc(self, scores, labels, feat, img_names=None):
        scores = self.softmax(scores)
        pred_val, pred_pos = torch.max(scores, 1) 
        pred_feat = feat# * weight[pred_pos].unsqueeze(2).unsqueeze(2)
        bs, dim, hei, wei = feat.size()
        re_feat = feat.view(bs, dim, -1)
        feat_hm = F.normalize(self.relu(feat).sum(1))
        feat_hm_view = feat_hm.view(bs, -1)
        feat_norm = feat_hm.view(bs, -1).mean(1).unsqueeze(1)
        #hm_ind = torch.nonzero(feat_hm_view > feat_norm)
        
        correct_judge = (pred_pos == labels)
        error_judge = (pred_pos != labels)
        update_judge = (pred_val.cpu() - self.bank_confidence[labels].cpu()) > 0.1
        forward_judge = (self.bank_confidence[labels].cpu() - pred_val.cpu()) > 0.1
        bank_judge = (self.bank_confidence[labels].cpu() != 0).cuda()
        pred_bank_judge = (self.bank_confidence[pred_pos].cpu() != 0).cuda()

        update_judge = correct_judge * update_judge.cuda()
        update_ind = torch.nonzero(update_judge).squeeze(1)

        forward_judge = correct_judge * forward_judge.cuda() + error_judge
        forward_ind = torch.nonzero(forward_judge * bank_judge).squeeze(1)

        error_judge *= bank_judge
        error_judge *= pred_bank_judge
        forward_error_ind = torch.nonzero(error_judge).squeeze(1)

        counter = 0
        aligned_pred_feat_gather = []
        structure_bank_feat_gather = []
        error_feat_gather = []
        error_hm_gather = []
        err_bank_feat_gather = []
        err_bank_confidence_gather = []
        self.update_feat_bank = self.feat_bank  
        for counter in range(len(pred_pos)):
            pick_num = self.structure_bank_num_max
            pick_val, pick_pos = torch.topk(feat_hm_view[counter], pick_num) 
            if (pred_pos[counter] == labels[counter]) and counter in update_ind:
                self.update_feat_bank[labels[counter].long()] = re_feat[counter][:, pick_pos].detach()
                self.update_bank_confidence_transport[labels[counter].long()] = pick_val.detach()
                self.update_bank_confidence[labels[counter]] = pred_val[counter]
                self.update_bank_position[labels[counter]] = pick_pos
                self.debug_img_list[labels[counter]] = img_names[counter]
        
            elif counter in forward_ind: 
                aligned_pred_feat = self.structure_forward(feat[counter], labels[counter].item(), pick_pos)
                if aligned_pred_feat is not None:
                    aligned_pred_feat_gather.append(aligned_pred_feat)
                    structure_bank_feat_gather.append(self.feat_bank[labels[counter].item(), :, :])
        return aligned_pred_feat_gather, structure_bank_feat_gather 





