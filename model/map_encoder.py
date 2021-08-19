import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .map_cnn import MapCNN


class MapEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_id = cfg.get('model_id', 'map_cnn')
        dropout = cfg.get('dropout', 0.0)
        self.normalize = cfg.get('normalize', True)
        self.dropout = nn.Dropout(dropout)
        if model_id == 'map_cnn':
            self.model = MapCNN(cfg)
            self.out_dim = self.model.out_dim
        elif 'resnet' in model_id:
            model_dict = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50
            }
            self.out_dim = out_dim = cfg.get('out_dim', 32)
            self.model = model_dict[model_id](pretrained=False, norm_layer=nn.InstanceNorm2d)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)
        else:
            raise ValueError('unknown map encoder!')

    def forward(self, x):
        if self.normalize:
            x = x * 2. - 1.
        x = self.model(x)
        x = self.dropout(x)
        return x
