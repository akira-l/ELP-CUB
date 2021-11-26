import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import pdb

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        num, ms1, ms2 = x.size()
        ent_p2g = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        ent_g2p = F.softmax(x, dim=2) * F.log_softmax(x, dim=2)
        ent_sum = - 1.0 * ent_p2g.view(num, -1).sum() - ent_g2p.view(num, -1).sum()
        return ent_sum / (ms1 * ms2)

if __name__ == '__main__':
    ent = Entropy()
    tmp = torch.rand(5, 10, 10)
    tmp2 = torch.ones(5, 10, 10) * 0.5
    ent1 = ent(tmp)
    ent2 = ent(tmp2)
    pdb.set_trace()
