
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.entropy_2d import Entropy 
from models.emd_loss import get_emd_distance

from utils.rhm_map import rhm_single 

import pdb

def align_loss(otmap_gather_list, 
               use_structure,
               structure_max ):
    get_entropy = Entropy()
    get_sim_loss = nn.CosineEmbeddingLoss()
    get_smooth_l1_loss = nn.SmoothL1Loss()

    entropy_val = 0
    map_loss = 0
    otmap_len = len(otmap_gather_list)
    if otmap_len > 0 and use_structure:
        otmap_gather_stack = torch.stack(otmap_gather_list)
        otmap_best_label = [torch.eye(structure_max) for x in range(otmap_len)]
        otmap_best_label = torch.stack(otmap_best_label).cuda()
        otmap_best_label = Variable(otmap_best_label)

        entropy_val = get_entropy(otmap_gather_stack)
        map_loss = get_smooth_l1_loss(otmap_gather_stack, otmap_best_label) 
        #map_loss = get_ce_loss(otmap_gather_stack.view(
            
    emd_loss = 0
    '''
    if len(err_pred_gather_list) > 0 and use_structure:
        pred_dim = err_pred_gather_list[0].size(1)
        bank_dim = err_bank_gather_list[0].size(1)
        for cur_pred, cur_pred_hm, cur_bank, cur_bank_conf in zip(err_pred_gather_list, err_hm_gather_list, err_bank_gather_list, err_conf_gather_list):
            sim_matrix = F.cosine_similarity(cur_pred.unsqueeze(1).repeat(1, bank_dim, 1), cur_bank.unsqueeze(2).repeat(1, 1, pred_dim), dim=0) 
            pad_sim = torch.zeros(pred_dim, pred_dim).cuda()
            pad_sim[:bank_dim, :] = sim_matrix
            pad_bank_conf = torch.zeros(pred_dim).cuda()
            pad_bank_conf[:bank_dim] = cur_bank_conf
            trans_vote = rhm_single(1 - pad_sim, cur_pred_hm, pad_bank_conf)
            emd_dis = trans_vote * (1 - pad_sim)
            emd_loss += emd_dis.sum()

        emd_loss /= bank_dim
    '''

    return [entropy_val, map_loss, emd_loss]


