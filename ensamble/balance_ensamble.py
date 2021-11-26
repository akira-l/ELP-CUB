import os
import torch
import torch.nn.functional as F

from save4submit import Submit_result

import pdb

all_data = torch.load('./ensamble/focal_weighted_score.pt')
balance_data = torch.load('./ensamble/score_gather.pt')
if all_data['name'] != balance_data['name']:
    raise Exception('name not match')

img_name = all_data['name']
all_feat = all_data['score'].cpu()

anno_load = open('./../butterfly/anno/train_all.txt').readlines()
anno = [int(x[:-1].split(' ')[-1]) for x in anno_load]
anno_count = [anno.count(x) for x in range(5419)]
count_1 = torch.Tensor([1.0 if x>100 else 0.5 for x in anno_count])
count_2 = torch.Tensor([0 if x>100 else 0.5 for x in anno_count])

balance_feat = F.log_softmax(balance_data['score'], dim=1).cpu()

result_gather = {}
for sub_name, sub_feat1, sub_feat2 in zip(img_name, all_feat, balance_feat):
    use_feat = sub_feat1*count_1 + sub_feat2*count_2
    val_, pos_ = torch.topk(use_feat, 3) 
    result_gather[sub_name] = '%d %d %d'%(pos_[0], pos_[1], pos_[2])

save_result = Submit_result('butterfly')
save_result(result_gather, 'all_balanced_result_gather.pt') 

