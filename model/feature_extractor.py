import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import model.resnet as model


class FeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', maml=True):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        model.ResNet.maml = maml
        model.SimpleBlock.maml = maml
        model.ConvBlock.maml = maml
        if backbone == 'resnet34':
            self.pretrained = model.ResNet34(flatten=False, leakyrelu=False)
        elif backbone == 'resnet18':
            self.pretrained = model.ResNet18(flatten=False, leakyrelu=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        
    def forward(self, x, mode='mte'):
        out = self.pretrained.trunk(x)
        if mode=='mtr':
            out = out.flatten(start_dim=2)
            out = out.mean(dim=2)
        return out
