from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

from diffusion_utils import compute_diffusion_params


def vmf_forward_chain(z0: torch.Tensor, steps: int, hypu_dist) -> List[torch.Tensor]:
    device = z0.device
    z_prev = F.normalize(z0, p=2, dim=1)
    zs: List[torch.Tensor] = []
    for t in range(1, steps + 1):
        theta_t = (math.pi / 2.0) * t / float(steps)
        v = hypu_dist.sample(z_prev.size(0)).to(device)
        z_new = math.cos(theta_t) * z_prev + math.sin(theta_t) * v
        z_new = F.normalize(z_new, p=2, dim=1)
        zs.append(z_new)
        z_prev = z_new
    return zs


def vmf_forward_last(z0: torch.Tensor, steps: int, hypu_dist) -> torch.Tensor:
    device = z0.device
    z_prev = F.normalize(z0, p=2, dim=1)
    for t in range(1, steps + 1):
        theta_t = (math.pi / 2.0) * t / float(steps)
        v = hypu_dist.sample(z_prev.size(0)).to(device)
        z_new = math.cos(theta_t) * z_prev + math.sin(theta_t) * v
        z_prev = F.normalize(z_new, p=2, dim=1)
    return z_prev


def vmf_decode_pairs_stream(
    z0: torch.Tensor,
    steps: int,
    hypu_dist,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    start: int = 1,
    micro_bs: int = 2000,
) -> torch.Tensor:
    device = z0.device
    dtype = z0.dtype
    T = int(steps)

    sqrt_1m, cum = compute_diffusion_params(T, device)
    norm = cum[start - 1]

    all_idx = torch.cat([row_idx, col_idx], dim=0)
    uniq_idx, inv = torch.unique(all_idx, sorted=True, return_inverse=True)
    B = row_idx.numel()
    inv_row = inv[:B]
    inv_col = inv[B:]

    z_prev = F.normalize(z0[uniq_idx], p=2, dim=1)  # [M, D]
    out = torch.zeros(B, device=device, dtype=dtype)

    for tau in range(1, T + 1):
        theta_t = (math.pi / 2.0) * tau / float(T)
        v = hypu_dist.sample(z_prev.size(0)).to(device)
        z_new = math.cos(theta_t) * z_prev + math.sin(theta_t) * v
        z_new = F.normalize(z_new, p=2, dim=1)

        if tau >= start:
            w = sqrt_1m[tau - 1]
            for s in range(0, B, micro_bs):
                e = min(s + micro_bs, B)
                rr_loc = inv_row[s:e]
                cc_loc = inv_col[s:e]
                sim = (z_new[rr_loc] * z_new[cc_loc]).sum(dim=1)
                out[s:e] += w * sim

        z_prev = z_new

    return torch.clamp(out / norm, 0, 1)
