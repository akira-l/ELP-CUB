import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb


class EraseLayer(nn.Module):
    def __init__(self, nei_num, size):
	super(EraseLayer, self).__init__()
	self.nei_num = nei_num
	self.map_size = size
	self.cuda_flag = True
 
