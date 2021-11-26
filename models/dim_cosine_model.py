import torch
from torch import nn
from torchvision import models, transforms, datasets
import torch.nn.functional as F

from models.CosFace_linear import MarginCosineProduct
import pdb

class DimCosSoftmaxModule(nn.Module):
    def __init__(self, config, ver='con'):
        super(CosSoftmaxModule, self).__init__()
        self.num_classes = config.numcls
        self.version = ver
        self.avgpool_1d = nn.AdaptiveAvgPool1d(output_size=1)

        self.avgpool_2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.ks = 1
        self.kpad = 1 if self.ks == 3 else 0

        self.mem_dim_kernel = nn.Sequential(
            nn.ConvTranspose1d(2048, 2048, 3, stride=2),
            nn.ReLU(),
            nn.Conv1d(2048, 2048, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.sp_down = nn.Linear(196, 1)

        self.dim_Acls = MarginCosineProduct(200, self.num_classes, m=0.5)
        #self.Acls = nn.Linear(config.bank_dim, self.num_classes, bias=False)
        self.relu = nn.ReLU()


    def forward(self, feat, label, mem_feat):
        bs = feat.size(0)
        dim = feat.size(1)
        feat_view = feat.view(bs, dim, -1)

        #avg_feat = self.avgpool_2d(feat).view(bs, -1)
        mem_dim_kernel = self.mem_dim_kernel(mem_feat.unsqueeze(2))
        conv_with_dim = F.conv1d(feat_view, mem_dim_kernel, groups=1, padding=1)
        mem_dim_cls_feat = self.sp_down(conv_with_dim).view(bs, -1)
        out = self.dim_Acls(mem_dim_cls_feat, label)

        return out



