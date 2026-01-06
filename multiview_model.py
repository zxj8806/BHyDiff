from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseModel


class MultiViewBHyDiff(BaseModel):
    def __init__(self, num_views: int = 3, consis_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_views = num_views
        self.consistency_weight = consis_weight
        self.view_logits = nn.Parameter(torch.zeros(num_views))

    def encode_mv(self, x, adj_norm_list, T: int | None = None):
        if T is None:
            T = self.T

        mus_views: List[List[torch.Tensor]] = []
        logs_views: List[List[torch.Tensor]] = []
        zs_views: List[List[torch.Tensor]] = []
        for v in range(self.num_views):
            mus_v, logs_v, zs_v = self.encode(x, adj_norm_list[v], T=T)
            mus_views.append(mus_v)
            logs_views.append(logs_v)
            zs_views.append(zs_v)

        z_views: List[torch.Tensor] = []
        for v in range(self.num_views):
            z_v = self.aggregate_poe(mus_views[v], logs_views[v])
            z_views.append(z_v)

        w = F.softmax(self.view_logits[: self.num_views], dim=0)
        z_stack = torch.stack(z_views, dim=0)
        z = (w[:, None, None] * z_stack).sum(dim=0)
        return z, z_views, zs_views, w

    def consistency_loss(self, z: torch.Tensor, z_views: List[torch.Tensor]):
        loss = 0.0
        for zv in z_views:
            loss = loss + (zv - z).pow(2).mean()
        return loss / len(z_views)
