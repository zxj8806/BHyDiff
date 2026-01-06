from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from cluster_utils import GraphConvSparse, ClusterAssignment
from diffusion_utils import compute_diffusion_params


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.num_neurons = kwargs["num_neurons"]
        self.num_features = kwargs["num_features"]
        self.embedding_size = kwargs["embedding_size"]
        self.nClusters = kwargs["nClusters"]
        self.T = kwargs.get("T", 30)

        act = {
            "ReLU": F.relu,
            "Sigmoid": F.sigmoid,
            "Tanh": F.tanh,
        }.get(kwargs.get("activation", "ReLU"), F.relu)
        self.activation = act

        self.kl_hyp_weight = kwargs.get("kl_hyp_weight", 0.5)
        init_kappa = kwargs.get("init_kappa", 10.0)
        self.align_weight = kwargs.get("align_weight", 0.1)
        self.align_alpha = kwargs.get("align_alpha", 2)
        self.align_num_neg = kwargs.get("align_num_neg", 1)
        self.align_margin = kwargs.get("align_margin", 1.0)
        self.cluster_reg_weight = kwargs.get("cluster_reg_weight", 0.1)
        self.entropy_reg_weight = kwargs.get("entropy_reg_weight", 2e-3)

        self.log_kappa = nn.Parameter(torch.tensor(float(init_kappa)))
        self.z_weight = nn.Parameter(torch.zeros(self.T))

        self.base_gcn = GraphConvSparse(self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse(self.num_neurons, self.embedding_size, lambda x: x)
        self.gcn_logsigma2 = GraphConvSparse(self.num_neurons, self.embedding_size, lambda x: x)

        self.assignment = ClusterAssignment(self.nClusters, self.embedding_size, kwargs["alpha"])

    def kappa(self) -> torch.Tensor:
        return F.softplus(self.log_kappa)

    def encode(self, x, adj, T: int = 1):
        mus, logs2, zs = [], [], []
        for _ in range(T):
            h = self.base_gcn(x, adj)
            mu = self.gcn_mean(h, adj)
            ls2 = self.gcn_logsigma2(h, adj)
            z = torch.randn_like(mu) * torch.exp(ls2 / 2) + mu
            mus.append(mu)
            logs2.append(ls2)
            zs.append(z)
        return mus, logs2, zs

    def encode_once(self, x, adj):
        h = self.base_gcn(x, adj)
        mu = self.gcn_mean(h, adj)
        logs2 = self.gcn_logsigma2(h, adj)
        return mu, logs2

    def aggregate_poe(self, mus: List[torch.Tensor], logs2: List[torch.Tensor]) -> torch.Tensor:
        T = len(mus)
        w_t = F.softmax(self.z_weight[:T], 0).view(T, 1, 1)
        mu_stack = torch.stack(mus, 0)
        logvar_stack = torch.stack(logs2, 0)
        precision = torch.exp(-logvar_stack)
        prec_w = w_t * precision
        return (prec_w * mu_stack).sum(0) / prec_w.sum(0)

    def decode_diffusion(self, zs: List[torch.Tensor], start: int = 1):
        T = len(zs)
        sqrt_1m, cum = compute_diffusion_params(T, zs[0].device)
        norm = cum[start - 1]
        acc = None
        for tau, z_tau in enumerate(zs[start - 1:], start=start):
            z_tau = F.normalize(z_tau, 2, 1)
            sim = z_tau @ z_tau.t()
            w = sqrt_1m[tau - 1]
            acc = sim * w if acc is None else acc + sim * w
        return torch.clamp(acc / norm, 0, 1)

    def decode_diffusion_pairs_stream(
        self,
        mu: torch.Tensor,
        logs2: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        start: int = 1,
        micro_bs: int = 2000,
    ) -> torch.Tensor:
        T = int(self.T)
        device = mu.device
        dtype = mu.dtype

        sqrt_1m, cum = compute_diffusion_params(T, device)
        norm = cum[start - 1]

        B = row_idx.numel()
        out = torch.empty(B, device=device, dtype=dtype)
        d = mu.size(1)

        for s in range(0, B, micro_bs):
            e = min(s + micro_bs, B)
            rr = row_idx[s:e]
            cc = col_idx[s:e]

            mu_r = mu[rr]
            mu_c = mu[cc]
            ls2_r = logs2[rr]
            ls2_c = logs2[cc]

            acc = None
            bsz = rr.numel()

            for tau in range(start, T + 1):
                eps_r = torch.randn((bsz, d), device=device, dtype=dtype)
                eps_c = torch.randn((bsz, d), device=device, dtype=dtype)

                z_r = mu_r + eps_r * torch.exp(0.5 * ls2_r)
                z_c = mu_c + eps_c * torch.exp(0.5 * ls2_c)

                z_r = F.normalize(z_r, 2, 1)
                z_c = F.normalize(z_c, 2, 1)

                sim = (z_r * z_c).sum(dim=1)
                w = sqrt_1m[tau - 1]
                acc = sim * w if acc is None else acc + sim * w

            out[s:e] = torch.clamp(acc / norm, 0, 1)

        return out

    @staticmethod
    def align_loss_pairs(
        z: torch.Tensor,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
        row_neg: torch.Tensor,
        col_neg: torch.Tensor,
        alpha: int = 2,
        margin: float = 1.0,
    ) -> torch.Tensor:
        dp = z[row_pos] - z[col_pos]
        pos = dp.norm(2, 1).pow(alpha).mean()

        dn = z[row_neg] - z[col_neg]
        neg = F.relu(margin - dn.norm(2, 1)).pow(alpha).mean()
        return pos + neg
