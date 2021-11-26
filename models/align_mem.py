import os
import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
#import pretrainedmodels

from utils.rhm_map import rhm, rhm_single

import pdb


class AlignMem(nn.Module):
    def __init__(self, config):
        super(AlignMem, self).__init__()

        self.num_classes = config.numcls
        self.pick_num = config.bank_pick_num
        self.otmap_thresh = config.otmap_thresh
        self.context_bank_num = 5
        self.structure_bank_num_max = config.otmap_struct_max 
        self.dim = config.bank_dim
        self.forget_para = 0.8

        self.feat_bank = torch.zeros(self.num_classes, self.dim, self.structure_bank_num_max).cuda()
        self.bank_confidence_transport = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()
        self.bank_confidence = torch.zeros(self.num_classes).cuda()
        self.structure_bank = torch.zeros(self.num_classes, self.structure_bank_num_max, self.dim).cuda()

        self.update_feat_bank = torch.zeros(self.num_classes, self.dim, self.structure_bank_num_max).cuda()
        self.update_bank_confidence_transport = torch.zeros(self.num_classes, self.structure_bank_num_max).cuda()
        self.update_bank_confidence = torch.zeros(self.num_classes).cuda()

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



    def operate_single_ot(self, candi, candi_val, bank_feat, bank_confidence):
        if bank_confidence.sum() == 0:
            return torch.zeros(self.structure_bank_num_max, self.structure_bank_num_max).cuda(), torch.zeros(self.structure_bank_num_max, self.structure_bank_num_max).cuda(), 0  
        re_candi = candi.permute(1, 0).cuda()
        re_bank = bank_feat.permute(1, 0).cuda()
        
        '''
        simmap_1 = torch.zeros(self.structure_bank_num_max, self.structure_bank_num_max).cuda()
        for cnt_cand in range(self.structure_bank_num_max):
            for cnt_bank in range(self.structure_bank_num_max):
                sub_candi = re_candi[cnt_cand].unsqueeze(0)
                sub_bank = re_bank[cnt_bank].unsqueeze(0)
                simmap_1[cnt_cand, cnt_bank] = self.cos_sim_1(sub_candi, sub_bank)
        '''

        rep_candi = re_candi.repeat(self.structure_bank_num_max, 1)
        rep_bank = re_bank.repeat(1, self.structure_bank_num_max).view(self.structure_bank_num_max * self.structure_bank_num_max, self.dim)
        simmap = self.cos_sim_1(rep_candi, rep_bank).view(self.structure_bank_num_max, self.structure_bank_num_max)

        #eye_sim = simmap * torch.eye(self.structure_bank_num_max).cuda()
        #cost_map = (simmap - eye_sim)
        #cost_map = 1 - (simmap - eye_sim)
        cost_map = 1 - simmap
        #cost_map = simmap

        #vote = rhm_single(1 - simmap, candi_val, bank_confidence.cuda())
        vote = rhm_single(cost_map, candi_val, bank_confidence.cuda())
        return vote, simmap, 1 


    def align_forward(self,  pick_val, pick_feat, label, return_map=True):

        otmap, simmap, mask = self.operate_single_ot(pick_feat, pick_val, self.feat_bank[label], self.bank_confidence_transport[label]) 
        return otmap, mask, simmap


    def err_sim_feat_pos(self, feat, bank_feat):
        feat_num = feat.size(1)
        pos_gather = []
        for sub_dim in range(bank_feat.size(1)):
            sim_val = self.cos_sim(feat.permute(1, 0), sub_dim.repeat(feat_num, 1)) 
            max_val, max_pos = torch.topk(sim_val, 1)
            pos_gather.append(max_pos)
        return torch.LongTensor(pos_gather).cuda()
        

        

    def proc_bank(self, scores, labels, feat, pick_val, img_names=None):
        scores = self.softmax(scores)
        pred_val, pred_pos = torch.max(scores, 1) 
        bs, dim, tmp_num = feat.size()


        correct_judge = (pred_pos == labels)
        error_judge = (pred_pos != labels)
        update_judge = (pred_val.cpu() - self.bank_confidence[labels].cpu()) > 0.1
        forward_judge = (self.bank_confidence[labels].cpu() - pred_val.cpu()) > 0.1
        bank_judge = (self.bank_confidence[labels].cpu() != 0).cuda()
        pred_bank_judge = (self.bank_confidence[pred_pos].cpu() != 0).cuda()

        update_judge = correct_judge * update_judge.cuda()
        update_ind = torch.nonzero(update_judge).squeeze(1)

        forward_judge = correct_judge * forward_judge.cuda() # + error_judge
        forward_correct_ind = torch.nonzero(forward_judge * bank_judge).squeeze(1)
        bank_judge_ind = torch.nonzero(bank_judge).squeeze(1)

        error_judge *= bank_judge
        error_judge *= pred_bank_judge
        forward_error_ind = torch.nonzero(error_judge).squeeze(1)

        counter = 0
        otmap_gather = []
        otmap_mask_gather = []
        simmap_gather = []
        self.update_feat_bank = self.feat_bank  

        for counter in range(len(pred_pos)):
            #cur_hm_ind = torch.nonzero(feat_hm_view[counter] > feat_norm[counter]).squeeze(1)
            #pick_num = len(cur_hm_ind) if len(cur_hm_ind) < self.structure_bank_num_max else self.structure_bank_num_max 
            pick_num = self.structure_bank_num_max
            if (pred_pos[counter] == labels[counter]):
                cur_feat_bank = self.feat_bank[labels[counter].long()]
                if counter in update_ind:
                    if counter not in bank_judge_ind:
                        self.update_feat_bank[labels[counter].long()] = feat[counter].detach()
                        self.update_bank_confidence_transport[labels[counter].long()] = pick_val[counter].detach()
                        self.update_bank_confidence[labels[counter]] = pred_val[counter]
                        #self.debug_img_list[labels[counter]] = img_names[counter]
 
                    else:
                        cur_feat = feat[counter].detach() 
                        #self.update_feat_bank[labels[counter].long()] = 0.1 * pred_match_bank + 0.9 * cur_feat     
                        self.update_feat_bank[labels[counter].long()] = cur_feat     
                        self.update_bank_confidence_transport[labels[counter].long()] = pick_val[counter].detach()
                        self.update_bank_confidence[labels[counter]] = pred_val[counter]
                        #self.debug_img_list[labels[counter]] = img_names[counter]

                elif counter in forward_correct_ind: 
                    #otmap, bank_match_pred, pred_match_bank = self.correct_forward(pick_pos, pick_val, feat[counter], labels[counter])
                    #cur_feat = re_feat[counter][:, pick_pos].detach() 
                    #self.update_feat_bank[labels[counter].long()] = 0.9 * pred_match_bank + 0.1 * cur_feat     
                     
                    cur_bank_conf = self.bank_confidence[labels[counter]] 
                    self.update_bank_confidence[labels[counter]] = cur_bank_conf - (0.1 / len(pred_pos))
                    #aligned_feat_bank = torch.matmul(re_feat[counter][:, pick_pos].detach(), otmap.detach().t())  
                    #self.update_feat_bank[labels[counter].long()] = 0.9 * cur_feat_bank + 0.1 * aligned_feat_bank     
                    #otmap_gather.append(otmap)

        if False:
            otmap, mask, simmap = self.align_forward( pick_val[counter], feat[counter], labels[counter])
            otmap_gather.append(otmap)
            otmap_mask_gather.append(mask)
            simmap_gather.append(simmap)

        #return torch.stack(otmap_gather), torch.FloatTensor(otmap_mask_gather), torch.stack(simmap_gather)

    def perform_align(self, feat, pick_val, labels):
        otmap_gather = []
        otmap_mask_gather = []
        simmap_gather = []
        for cnt in range(feat.size(0)):

            otmap, mask, simmap = self.align_forward(pick_val[cnt], feat[cnt], labels[cnt])
            otmap_gather.append(otmap)
            otmap_mask_gather.append(mask)
            simmap_gather.append(simmap)
        return torch.stack(otmap_gather),\
               torch.FloatTensor(otmap_mask_gather).cuda(),\
               torch.stack(simmap_gather)


    def contrast_ind(self, feat, pick_val, con_cls, labels):
        total_num = feat.size(0)
        top2_val, top2_pos = torch.topk(con_cls, 2)
        gather_pos_ind = []
        gather_neg_ind = []
        gather_ind = []
        for cnt in range(total_num):
            pos_ind = labels[cnt]
            if top2_pos[cnt, 0] == labels[cnt]: 
                neg_ind = top2_pos[cnt, 1] 
                #otmap, mask, simmap = self.align_forward(pick_val[cnt], feat[cnt], pos_ind) 
            else:
                neg_ind = top2_pos[cnt, 0]
            if self.bank_confidence[neg_ind] != 0 and self.bank_confidence[pos_ind] != 0:
                gather_ind.append(cnt)
                gather_pos_ind.append(pos_ind)
                gather_neg_ind.append(neg_ind)
        if len(gather_pos_ind) != 0:
            return torch.LongTensor(gather_ind).cuda(),\
                   torch.LongTensor(gather_pos_ind).cuda(),\
                   torch.LongTensor(gather_neg_ind).cuda()
        else:
            return None, None, None



    def heatmap_debug_plot(self, heatmap, img_names, prefix):
        import cv2
        data_root = '../dataset/CUB_200_2011/dataset/data/'
        save_folder = './vis_tmp_save'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        counter = 0
        for sub_hm, img_name in zip(heatmap, img_names):
            np_hm = sub_hm.detach().cpu().numpy()
            np_hm -= np.min(np_hm)
            np_hm /= np.max(np_hm)  
            np_hm = np.uint8(255 * np_hm)
            np_hm = cv2.applyColorMap(np_hm, cv2.COLORMAP_JET)
            re_hm = cv2.resize(np_hm, (300, 300))
            
            raw_img = cv2.imread(os.path.join(data_root, img_name))
            re_img = cv2.resize(raw_img, (300, 300))

            canvas = np.zeros((300, 610, 3))
            canvas[:, :300, :] = re_img
            canvas[:, 310:, :] = re_hm
            save_name = img_name.split('/')[-1][:-4]
            cv2.imwrite(os.path.join(save_folder, prefix + '_' + save_name + '_' + str(counter) + '_heatmap_cmp.png'), canvas)
            counter += 1







