import torch
from torch import nn
from torchvision import models, transforms, datasets
import torch.nn.functional as F

from models.CosFace_linear import MarginCosineProduct
import pdb

class CosSoftmaxModule(nn.Module):
    def __init__(self, config, ver='con'):
        super(CosSoftmaxModule, self).__init__()
        self.num_classes = config.numcls
        self.version = ver
        self.avgpool_1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.mem_m = config.mem_m

        self.avgpool_2d = nn.AdaptiveAvgPool2d(output_size=1)

        #self.sp_Acls = MarginCosineProduct(2048, self.num_classes, m=0.6)
        self.sp_Acls = nn.Sequential(
                                 nn.Linear(2048, 512), 
                                 #nn.ReLU(), 
                                 nn.Linear(512, self.num_classes)
                                 ) 

        self.relu = nn.ReLU()


    def forward(self, mem_feat):

        mem_out = self.sp_Acls(mem_feat)

        return mem_out





