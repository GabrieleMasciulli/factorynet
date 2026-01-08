import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SignalEncoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64):
        super().__init__()

        # 1D-CNN
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # reduces time dimension to 1
        )

        # projection head to the shared latent space
        self.projection = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, 1) -> needs (batch, 1, seq_len) for Conv1d
        x = x.transpose(1, 2)

        features = self.conv(x).squeeze(-1)
        latent = self.projection(features)

        return latent


class FactoryCLIP(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.command_encoder = SignalEncoder(latent_dim=latent_dim)
        self.current_encoder = SignalEncoder(latent_dim=latent_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, cmd, cur):
        z_cmd = self.command_encoder(cmd)
        z_cur = self.current_encoder(cur)
        return z_cmd, z_cur  # returns non-normalized embeddings


def contrastive_loss(z_cmd, z_cur, logit_scale):
    """
    Standard CLIP-style InfoNCE loss

    Note: input embeddings are assumed to be L2-normalized
    """
    scale = logit_scale.exp()  # 100 in CLIP
    logits = scale * torch.matmul(z_cmd, z_cur.t())

    # here we assume positive pairs are aligned by batch index i.e. (cmd[i], cur[i]) is a positive pair
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_cmd = F.cross_entropy(logits, labels)
    loss_cur = F.cross_entropy(logits.t(), labels)

    return (loss_cmd + loss_cur) / 2
