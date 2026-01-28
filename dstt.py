import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Module
from torch_geometric.utils import to_dense_batch


class DynamicHybridMask(Module):
    def __init__(self, num_features, num_channels, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.channel_attention = Sequential(
            Linear(num_features, 64),
            ReLU(),
            Linear(64, 1)
        )

    def forward(self, x):
        B, C, F_dim = x.shape
        device = x.device
        channel_scores = self.channel_attention(x)
        channel_probs = torch.sigmoid(channel_scores).squeeze(-1)

        mask_random = torch.rand(B, C, F_dim, device=device) > self.mask_ratio
        channel_indices = (torch.rand(B, C, 1, device=device) > self.mask_ratio).float()
        mask_channel = channel_indices.expand(-1, -1, F_dim)

        prob_matrix = channel_probs.unsqueeze(-1)
        random_threshold = torch.rand(B, C, 1, device=device)
        mask = torch.where(random_threshold < prob_matrix, mask_random, mask_channel.bool())
        return x * mask


class iTransformerEncoder(nn.Module):


    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class DSTT(nn.Module):

    def __init__(self, num_channels=62, seq_len=1325, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.masker = DynamicHybridMask(seq_len, num_channels)
        self.enc_embedding = nn.Linear(seq_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_channels, d_model) * 0.02)
        self.encoder = iTransformerEncoder(d_model, nhead, num_layers, dim_feedforward=512)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, batch, apply_mask=False):
        # 稀疏转稠密 (Batch, 62, 1325)
        x, _ = to_dense_batch(x, batch)
        if apply_mask: x = self.masker(x)

        # 映射与拓扑学习
        x_enc = self.enc_embedding(x) + self.pos_embedding
        x_enc = self.encoder(x_enc)

        # 聚合特征
        flat_features = self.norm(x_enc.mean(dim=1))
        return flat_features