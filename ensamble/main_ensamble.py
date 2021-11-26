import os
import sys
sys.path.append("..")

import pickle
import torch
import pandas as pd

from ensemble_analyse import ensamble_score
from utils.save4submit import Submit_result

import pdb

def ensemble(dataset, data_path, save_suffix):
    save_csv = Submit_result(dataset)
    ensamble_dict = ensamble_score(dataset, data_path)
    id_list = None
    accu_prob = None
    accu_weight = 0
    for score_file in ensamble_dict.keys():
        score_path = os.path.join(data_path, score_file)
        data = pickle.load(open(score_path, 'rb'))
        if id_list is None:
            id_list = data['id'].tolist()
        else:
            assert(id_list == data['id'].tolist())
        prob = data['probs']*ensamble_dict[score_file]
        accu_weight += ensamble_dict[score_file]
        if accu_prob is None:
            accu_prob = prob
        else:
            accu_prob += prob

    accu_prob /= accu_weight
    prob_tensor = torch.from_numpy(accu_prob)
    val_, pos_ = prob_tensor.topk(3, 1, True, True)
    pos = pos_.squeeze().data.numpy()
    result_gather = {}
    for img_name, pred_cls in zip(id_list, pos):
        result_gather[img_name] = ' '.join([str(x) for x in pred_cls])
    save_csv(result_gather, save_suffix)







def raw():

    ensemble_file = [
                     [0.2631, 'result/ft_30_clean/ft_30_clean_epoch_8_prec3_93.74007951040116_tta.pkl'],
                     [0.2368, 'result/nasnetalarge/nasnetalarge_epoch_22_prec3_93.26720514002238_tta.pkl'],
                     [0.2631, 'result/ft_30_inceptionresnetv2/ft_30_inceptionresnetv2_epoch_11_prec3_94.00793618096246_tta.pkl'],
                     [0.2368, 'result/senet154/senet154_epoch_20_prec3_94.61977755828725_tta.pkl'],
                     ]
    num_file = len(ensemble_file)
    w, file = ensemble_file[0]
    with open(file, 'rb') as f:
        data = pickle.load(f)
    all_id = data['id'].tolist()
    #print(all_id)
    all_prob = w*data['probs']
    all_w = w
    for i in range(1, num_file):
        w, file = ensemble_file[i]
        with open(file, 'rb') as f:
            data = pickle.load(f)
        assert(all_id == data['id'].tolist())
        prob = w*data['probs']
        all_prob += prob
        all_w += w
    all_prob /= all_w


    prob_torch = torch.from_numpy(all_prob)
    _, pred = prob_torch.topk(3, 1, True, True)
    all_pred = pred.squeeze().data.numpy()
    all_pred_str = []
    for pred in all_pred:
        s = ' '.join([str(top3) for top3 in pred])
        all_pred_str.append(s)

    csv_file = 'result/ensemble/senet154_incep_nasnet_seresnet50.csv'
    df = pd.DataFrame({'id':all_id, 'predicted':all_pred_str})
    df.to_csv(csv_file, header=True, index=False)
    print('Save csv to {}'.format(csv_file))

if __name__ == '__main__':

    ensemble('butterfly', './dcl_0.7_ensamble', 'dcl7_ensamble_v1')

