import os, sys

import torch
import torch.nn.functional as F
import cv2

import pdb

def get_emd_distance(similarity_map, weight_1, weight_2):
    num_query = similarity_map.shape[0]
    num_proto = similarity_map.shape[1]
    num_node=weight_1.shape[-1]

    _, flow = emd_inference_opencv(1 - similarity_map, weight_1, weight_2)
    emd_dis = (1 - similarity_map) * torch.from_numpy(flow).cuda()
    return emd_dis.sum(-1).sum(-1)


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    pdb.set_trace()
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5


    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow

def emd_inference_opencv_test(distance_matrix,weight1,weight2):
    distance_list = []
    flow_list = []

    for i in range (distance_matrix.shape[0]):
        cost,flow=emd_inference_opencv(distance_matrix[i],weight1[i],weight2[i])
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    emd_distance = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()

    return emd_distance,flow






if __name__ == '__main__':
    random_seed = True
    if random_seed:
        pass
    else:

        seed = 1
        import random
        import numpy as np

        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    batch_size = 50
    num_node = 25
    form = 'L2'  # in [ 'L2', 'QP' ]


    cosine_distance_matrix = torch.rand(batch_size, num_node, num_node).cuda()

    weight1 = torch.rand(batch_size, num_node).cuda()
    weight2 = torch.rand(batch_size, num_node).cuda()



    emd_distance_cv, cv_flow = emd_inference_opencv_test(cosine_distance_matrix, weight1, weight2)

    emd_score_cv=((1-cosine_distance_matrix)*cv_flow).sum(-1).sum(-1)
    emd_score_qpth = ((1 - cosine_distance_matrix) * qpth_flow).sum(-1).sum(-1)
    print('emd difference:', (emd_score_cv - emd_score_qpth).abs().max())
    pass

