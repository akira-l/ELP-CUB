#coding=utf8
from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import LossRecord
from models.focal_loss import FocalLoss

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval_turn(model_recieve, data_loader, val_version, epoch_num, config):

    model = model_recieve['base']
    avg_bank = model_recieve['align']
    as_model = model_recieve['bank']

    model.train(False)
    as_model.train(False)

    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0

    tmp_val_corrects1 = 0
    tmp_val_corrects2 = 0
    tmp_val_corrects3 = 0

    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    get_l1_loss = nn.L1Loss()
    get_focal_loss = FocalLoss()
    get_ce_loss = nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    relu_layer = nn.ReLU()

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...'%val_version)
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            # print data
            #inputs,  labels, labels_swap, law_swap, img_name = data_val
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            img_names = data_val[-1]
            # forward
            outputs, cls_feat, avg_feat, top_avg_feat = model(inputs)
            pred_val, pred_ind = outputs.max(1)

            tmp_out = as_model.module.sp_Acls(avg_feat)
            tmp_score = F.softmax(tmp_out.detach(), 1)

            loss = 0
            ce_loss = get_ce_loss(outputs, labels).item()
            loss += ce_loss

            val_loss_recorder.update(loss)
            val_celoss_recorder.update(ce_loss)

            print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss))

            top3_val, top3_pos = torch.topk(outputs, 3)
            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

            top3_val, top3_pos = torch.topk(tmp_out, 3)
            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            tmp_val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            tmp_val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            tmp_val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

        val_acc1 = val_corrects1 / item_count
        val_acc2 = val_corrects2 / item_count
        val_acc3 = val_corrects3 / item_count

        tmp_val_acc1 = tmp_val_corrects1 / item_count
        tmp_val_acc2 = tmp_val_corrects2 / item_count
        tmp_val_acc3 = tmp_val_corrects3 / item_count

        t1 = time.time()
        since = t1-t0
        print('\n')
        print('--'*30)
        print('noraml eval: % 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=False), val_version, val_acc1, val_version, val_acc2, val_version, val_acc3, since))
        print('--' * 30)
        print('tmp eval: % 3d %s %s %s-loss: %.4f ||tmp-%s-acc@1: %.4f tmp-%s-acc@2: %.4f tmp-%s-acc@3: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=False), val_version, tmp_val_acc1, val_version, tmp_val_acc2, val_version, tmp_val_acc3, since))
        print('--' * 30)
        print('\n')

    return val_acc1, val_acc2, val_acc3


