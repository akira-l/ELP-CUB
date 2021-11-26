"""Implementation of optimal transport+geometric post-processing (Hough voting)"""

import math

import torch.nn.functional as F
import torch

import utils.geometry

import pdb


def perform_sinkhorn(C,epsilon,mu,nu,a=[],warm=False,niter=1,tol=10e-9):
    """Main Sinkhorn Algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],1)).cuda() / C.shape[0]

    K = torch.exp(-C/epsilon)

    Err = torch.zeros((niter,2))
    for i in range(niter):
        b = nu/torch.mm(K.t(), a)
        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.mm(K, b)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI,mu,nu,Err


def appearance_similarity(src_feats, trg_feats, exp1=3):
    r"""Semantic appearance similarity (exponentiated cosine)"""
    src_feat_norms = torch.norm(src_feats, p=2, dim=1).unsqueeze(1)
    trg_feat_norms = torch.norm(trg_feats, p=2, dim=1).unsqueeze(0)
    sim = torch.matmul(src_feats, trg_feats.t()) / \
          torch.matmul(src_feat_norms, trg_feat_norms)
    sim = torch.pow(torch.clamp(sim, min=0), exp1)

    return sim


def appearance_similarityOT(cost, pred_val, bank_val, exp1=1.0, exp2=1.0, eps=0.05):

    #n1 = cost.size(0)
    #mu = torch.ones((n1,))/n1
    zero_pos = (pred_val == 0).float()
    pred_val = pred_val + zero_pos * 1e-5
    mu_weight = pred_val.sum()
    mu = pred_val / mu_weight
    #n2 = cost.size(1) 
    #nu = torch.ones((n2,))/n2
    zero_pos = (bank_val == 0).float()
    bank_val = bank_val + zero_pos * 1e-5
    nu_weight = bank_val.sum()
    nu = bank_val / nu_weight

    max_loop = 20

    #with torch.no_grad():
    epsilon = eps
    cnt = 0
    while True:
        PI,a,b,err = perform_sinkhorn(cost, epsilon, mu, nu)
        if not torch.isnan(PI).any():
            if cnt>0:
                PI[torch.isnan(PI)] = 0
                PI[torch.isinf(PI)] = 0
                print('PI has nan : ', cnt)
            break
        else: # Nan encountered caused by overflow issue is sinkhorn
            epsilon *= 2.0
            #print(epsilon)
            cnt += 1
        if cnt > max_loop:
            print('ot loop > ', max_loop, ' : ', cnt)
            PI[torch.isnan(PI)] = 0
            raise Exception('over loop')

    n1 = cost.size(0)
    PI = n1*PI # re-scale PI 
    #exp2 = 1.0 for spair-71k, TSS
    #exp2 = 0.5 # for pf-pascal and pfwillow
    PI = torch.pow(torch.clamp(PI, min=0), exp2)

    return PI



def hspace_bin_ids(src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
    r"""Compute Hough space bin id for the subsequent voting procedure"""
    src_ptref = torch.tensor(src_imsize, dtype=torch.float).to(src_box.device)
    src_trans = geometry.center(src_box)
    trg_trans = geometry.center(trg_box)
    xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
                  repeat(1, 1, len(trg_box)) + \
              trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

    bin_ids = (xy_vote / hs_cellsize).long()

    return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x


def build_hspace(src_imsize, trg_imsize, ncells):
    r"""Build Hough space where voting is done"""
    hs_width = src_imsize[0] + trg_imsize[0]
    hs_height = src_imsize[1] + trg_imsize[1]
    hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
    nbins_x = int(hs_width / hs_cellsize) + 1
    nbins_y = int(hs_height / hs_cellsize) + 1

    return nbins_x, nbins_y, hs_cellsize


def rhm(cost_map, pred_val, bank_val,  exp1=1.0, exp2=1.0, eps=0.05, ncells=8192):
    r"""Regularized Hough matching"""
    # exp1 default = 1.0
    # exp2 default = 1.0
    # eps default = 0.05
    # sim = 'OTGeo'

    # Prepare for the voting procedure

    gather_votes = []
    non_zero_cost = torch.nonzero(cost_map.sum(1).sum(1)).squeeze(1)
    non_zero_ind = (cost_map.sum(1).sum(1) != 0) 
    #cost_map = 1 - cost_map + 2*torch.eye(cost_map.size(1)).cuda()
    cost_map = cost_map * non_zero_ind.unsqueeze(1).unsqueeze(1).float()
    gather_votes = torch.zeros_like(cost_map)
    for ind in non_zero_cost:
        sub_pred_val_wei = pred_val[ind].sum()
        sub_pred_val = pred_val[ind] * (1 / sub_pred_val_wei) 
        sub_bank_val_wei = bank_val[ind].sum()
        sub_bank_val = bank_val[ind] * (1 / sub_bank_val_wei)
        vote = appearance_similarityOT(cost_map[ind], sub_pred_val, sub_bank_val)
        vote = vote * (1 / (vote.sum() + 1e-5))
        gather_votes[ind] = vote
    return gather_votes


def rhm_single(cost_map, pred_val, bank_val,  exp1=1.0, exp2=1.0, eps=0.05, ncells=8192):
    
    vote = appearance_similarityOT(cost_map, pred_val, bank_val)
    return vote





