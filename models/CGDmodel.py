import torch
from torch import nn
from torch.nn import functional as F

from models.resnetv1 import resnet50, resnext50_32x4d





class GlobalDescriptor(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, x):
        assert x.dim() == 4, 'the input tensor of GlobalDescriptor must be the shape of [B, C, H, W]'
        if self.p == 1:
            return x.mean(dim=[-1, -2])
        elif self.p == float('inf'):
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            sum_value = x.pow(self.p).mean(dim=[-1, -2])
            return torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / self.p))

    def extra_repr(self):
        return 'p={}'.format(self.p)


class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim() == 2, 'the input tensor of L2Norm must be the shape of [B, C]'
        return F.normalize(x, p=2, dim=-1)


class CGDmodel(nn.Module):
    def __init__(self, config):
        super(CGDmodel, self).__init__()

        model_config = config['CGD-model']
        gd_config = model_config["gd_config"]
        backbone_type = model_config["backbone_type"]
        feature_dim = model_config["feature_dim"]
        num_classes = config['dataset']['n_classes']

        # Backbone Network
        backbone = resnet50(pretrained=True) if backbone_type == 'resnet50' else resnext50_32x4d(pretrained=True)
        self.features = []
        for name, module in backbone.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Linear):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)

        # Main Module
        n = len(gd_config)
        k = feature_dim // n
        assert feature_dim % n == 0, 'the feature dim should be divided by number of global descriptors'

        self.global_descriptors, self.main_modules = [], []
        for i in range(n):
            if gd_config[i] == 'S':
                p = 1
            elif gd_config[i] == 'M':
                p = float('inf')
            else:
                p = 3
            self.global_descriptors.append(GlobalDescriptor(p=p))
            self.main_modules.append(nn.Sequential(nn.Linear(2048, k, bias=False), L2Norm()))
        self.global_descriptors = nn.ModuleList(self.global_descriptors)
        self.main_modules = nn.ModuleList(self.main_modules)

        # Auxiliary Module
        self.auxiliary_module = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, num_classes, bias=True))

    def forward(self, x):
        shared = self.features(x)
        global_descriptors = []
        for i in range(len(self.global_descriptors)):
            global_descriptor = self.global_descriptors[i](shared)
            if i == 0:
                classes = self.auxiliary_module(global_descriptor)
            global_descriptor = self.main_modules[i](global_descriptor)
            global_descriptors.append(global_descriptor)
        global_descriptors = F.normalize(torch.cat(global_descriptors, dim=-1), dim=-1)
        return global_descriptors, classes
