import os
import json

import torch

from utils.test_tool import cls_base_acc

resume = 'result_gather_weights-10-3999-[0.8889].pt' 
result_gather = torch.load('result_gather_weights-10-3999-[0.8889].pt')

cls_top1, cls_top3, cls_count = cls_base_acc(result_gather)

acc_report_io = open('acc_report_%s_%s.json'%(None, resume.split('/')[-1]), 'w')
json.dump({#'val_acc1':val_acc1,
                   #'val_acc2':val_acc2,
                   #'val_acc3':val_acc3,
                   'cls_top1':cls_top1,
                   'cls_top3':cls_top3,
                   'cls_count':cls_count}, acc_report_io)
acc_report_io.close()

