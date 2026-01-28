import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Module

class Projector(Module):
    """对比学习投影器 (映射到隐空间 Z)"""
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.model = Sequential(Linear(input_dim, input_dim), ReLU(), Linear(input_dim, output_dim))
    def forward(self, x): return self.model(x)

class Decoder(Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = Sequential(Linear(input_dim, 1024), ReLU(), Linear(1024, output_dim))
    def forward(self, x): return self.model(x)

class SoftAlignment(nn.Module):

    def __init__(self, d_model=256, z_dim=128, num_classes=3, original_input_dim=(62 * 1325)):
        super().__init__()
        self.classifier = Linear(d_model, num_classes)
        self.projector = Projector(d_model, z_dim)
        self.decoder = Decoder(d_model, original_input_dim)

    def forward(self, flat_features):
        class_output = self.classifier(flat_features)
        pred = F.softmax(class_output, dim=1)
        z_features = self.projector(flat_features)
        recon_output = self.decoder(flat_features)
        return class_output, pred, z_features, recon_output