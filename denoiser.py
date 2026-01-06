from __future__ import annotations

import torch
import torch.nn as nn


class SphericalDDPMDenoiser(nn.Module):
    def __init__(self, embed_dim: int, max_steps: int, hidden_mult: int = 2) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.time_embed = nn.Embedding(self.max_steps + 1, embed_dim)  # 1..T used
        hidden_dim = hidden_mult * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z_t: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t_idx.clamp(0, self.max_steps))
        h = torch.cat([z_t, t_emb], dim=-1)
        return self.net(h)
