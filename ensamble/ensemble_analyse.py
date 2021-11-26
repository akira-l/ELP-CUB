import os
import sys
sys.path.append("..")

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from itertools import product
import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.optimize import differential_evolution
import multiprocessing
from multiprocessing import Pool
import queue
import threading

from config import LoadConfig

import pdb

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights):
    # make predictions
    yhats = array(members)
    # weighted sum across ensemble members
    summed = tensordot(yhats, weights, axes=((0),(0)))
    # argmax across classes
    result = argmax(summed, axis=1)
    return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testy):
    # make prediction
    yhat = ensemble_predictions(members, weights)
    # calculate accuracy
    return accuracy_score(testy, yhat)

# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result

# loss function for optimization process, designed to be minimized
def loss_function(weights, members, testy):
    # normalize weights
    normalized = normalize(weights)
    # calculate error rate
    return 1.0 - evaluate_ensemble(members, normalized, testy)


class null_args(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.swap_num = None
        self.backbone = None
        self.buff_drop2 = None


def get_val_dict(dataset):
    args = null_args(dataset)
    cfg = LoadConfig(args, 'ensamble')
    anno_path = os.path.join(cfg.anno_root.replace('../', '../../'), 'val.txt')
    anno = open(anno_path).readlines()
    val_dict = {}
    val_val = []
    for anno_item in anno:
        img_name, label = anno_item.strip('\n').split(' ')
        val_dict[img_name] = int(label)
        val_val.append(int(label))
    return val_dict, np.array(val_val)

def process_bar(task_name, num):
    str_list = ['-', '\\', '|', '/']
    print("\r%s in processing "%task_name+str_list[num%4], end="", flush=True)


def ensamble_score(dataset, ensamble_path):
    val_dict, labels = get_val_dict(dataset)
    file_list = os.listdir(ensamble_path)
    ensamble_dict = {}
    members = []
    id_list = None
    for file_name in file_list:
        if file_name[:3] == 'val' and file_name[-3:] == 'pkl':
            snap_name = file_name[3:]
            test_name = 'test'+snap_name
            val_name = 'val'+snap_name
            if test_name in file_list:
                ensamble_dict[test_name] = val_name
            else:
                raise Exception('not find corresponding val/test pkl: %s '%test_name)
            data = pickle.load(open(os.path.join(ensamble_path, val_name), 'rb'))
            print('load %s ...'%val_name)
            if not id_list is None:
                assert(data['id'], id_list)
            else:
                id_list = data['id']
            members.append(data['probs'])

    bound_w = [(0.0, 1.0) for x in range(len(members))]
    search_arg = (members, labels)

    pool = Pool()
    process_queue = queue.Queue()

    args_dict = {'func':loss_function, 'bounds':bound_w, 'args':search_arg, 'strategy':'best1bin',
                  'maxiter':1000, 'popsize':15, 'tol':1e-7,'mutation':(0.5, 1), 'recombination':0.7, 'seed':None,
                  'callback':None, 'disp':False, 'polish':True,'init':'latinhypercube'}
    result = pool.apply_async(differential_evolution, args=(loss_function, bound_w, search_arg, 'best1bin', 1000, 15, 1e-7, (0.5, 1), 0.7, None, None, False, True, 'latinhypercube', ))
    print('queue add ...')
    count_num = 0
    pool.apply_async(process_bar, args=('differential_evolution', count_num ))
    while 1:
        count_num += 1
        try:
            result.get(True)
            break
        except multiprocessing.context.TimeoutError:
            pool.apply_async(process_bar, args=('differential_evolution', count_num ))
    pool.close()
    pool.join()


    get_result = result.get()
    weights = normalize(get_result['x'])
    print('Optimized Weights: %s' % weights)
    score = evaluate_ensemble(members, weights, labels)
    print('Optimized Weights Score: %.3f' % score)

    #weights = [0.51155477, 0.48844523]
    for w_, name_ in zip(weights, ensamble_dict.keys()):
        ensamble_dict[name_] = w_
    return ensamble_dict










'''


val_prob = [
            'val_ft_30_senet154_secondmodel_epoch_4_prec3_94.88095233251178_tta.pkl',
            'val_senet154_epoch_20_prec3_94.61977755828725_tta.pkl',
            'val_ft_30_inceptionresnetv2_epoch_11_prec3_94.00793618096246_tta.pkl',
            'val_ft_30_clean_epoch_8_prec3_93.74007951040116_tta.pkl',
            'val_nasnetalarge_epoch_22_prec3_93.26720514002238_tta.pkl'
           ]

members = []
with open(val_prob[0], 'rb') as f:
    data = pickle.load(f)
members.append(data['probs'])
all_id = data['id'].tolist()

for prob_file in val_prob[1:]:
    with open(prob_file, 'rb') as f:
        data = pickle.load(f)
    members.append(data['probs'])
    assert(all_id == data['id'].tolist())

val_dict = {}
with open('../input/val.txt') as f:
    for line in f.readlines():
        key, value = line.strip('\n').split(' ')
        val_dict[key] = int(value)
target = []
for image_path in all_id:
    target.append(val_dict[image_path])
val_true = np.array(target)


testy = val_true
n_members = len(members)
for i in range(n_members):
    test_acc = (argmax(members[i], axis=1) == val_true).mean()
    print('Model %d: %.3f' % (i+1, test_acc))
# evaluate averaging ensemble (equal weights)
weights = [1.0/n_members for _ in range(n_members)]
score = evaluate_ensemble(members, weights, testy)
print('Equal Weights Score: %.3f' % score)
# define bounds on each weight

bound_w = [(0.0, 1.0)  for _ in range(n_members)]
# arguments to the loss function
# search_arg = (members, testX, testy)
search_arg = (members, testy)
# global optimization of ensemble weights
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
# get the chosen weights
weights = normalize(result['x'])
print('Optimized Weights: %s' % weights)
# evaluate chosen weights
score = evaluate_ensemble(members, weights, testy)
print('Optimized Weights Score: %.3f' % score)


'''
