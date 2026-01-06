from __future__ import annotations

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device=None,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


def compute_diffusion_params(timesteps: int, device) -> tuple[torch.Tensor, torch.Tensor]:

    betas = linear_beta_schedule(timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_1m = torch.sqrt(1.0 - alphas_cumprod)

    rev_cum = torch.flip(sqrt_1m, dims=[0]).cumsum(0)
    cum_sqrt_1m = torch.flip(rev_cum, dims=[0])
    return sqrt_1m, cum_sqrt_1m
