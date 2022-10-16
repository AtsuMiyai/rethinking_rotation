from abc import *
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=4):
        super(BaseModel, self).__init__()
        self.rotate_cls_layer = nn.Linear(last_dim, num_classes)

    def forward(self, inputs):
        features = self.penultimate(inputs)
        output = self.rotate_cls_layer(features)
        return output


