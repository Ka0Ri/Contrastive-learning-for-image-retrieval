import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from efficientnet_pytorch import EfficientNet

class EfficientCLR(nn.Module):

    def __init__(self, config):
        super(EfficientCLR, self).__init__()
        model_config = config['Eff-model']
      
        self.ft = EfficientNet.from_pretrained(model_config['pretrained'])
        num_ftrs = model_config['num_ftrs']
        self.head = nn.AdaptiveAvgPool2d(output_size=1)
        self.projector = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, model_config['out_dim'])
        )

         
    def forward(self, x):
        h = self.ft.extract_features(x)
        h = self.head(h)
        h = torch.flatten(h, start_dim=1)

    
        x = self.projector(h)
        x = F.normalize(x, dim=1)
      
       
        return h, x
