from __future__ import annotations

import math
from typing import Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from multiview_model import MultiViewBHyDiff
from denoiser import SphericalDDPMDenoiser
from vmf_utils import vmf_forward_last, vmf_decode_pairs_stream

from vmfmix.vmf import VMFMixture
from vmfmix.von_mises_fisher import VonMisesFisher, HypersphericalUniform
from cluster_utils import Clustering_Metrics


class BHyDiff(nn.Module):
    def __init__(
        self,
        vmf_strength: float = 0.5,
        vmf_steps: int | None = None,
        ddpm_weight: float = 0.1,
        ddpm_steps: int | None = None,
        **mv_kwargs: Any,
    ) -> None:
        super().__init__()

        self._HypersphericalUniform = HypersphericalUniform
        self._VonMisesFisher = VonMisesFisher
        self._VMFMixture = VMFMixture
        self._Clustering_Metrics = Clustering_Metrics

        self.base = MultiViewBHyDiff(**mv_kwargs)

        self.vmf_strength = float(vmf_strength)
        self.vmf_steps = int(vmf_steps) if vmf_steps is not None else int(getattr(self.base, "T", 30))

        self.ddpm_weight = float(ddpm_weight)
        self.ddpm_steps = int(ddpm_steps) if ddpm_steps is not None else int(getattr(self.base, "T", 30))
        self.denoiser = SphericalDDPMDenoiser(self.embedding_size, self.ddpm_steps)

        self._hypu = None

    @property
    def nClusters(self) -> int:
        return int(getattr(self.base, "nClusters"))

    @property
    def embedding_size(self) -> int:
        return int(getattr(self.base, "embedding_size"))

    def kappa(self) -> torch.Tensor:
        return self.base.kappa()

    def _get_hypu(self, device: torch.device):
        if (self._hypu is None) or (getattr(self._hypu, "device", None) != device):
            dim = self.embedding_size - 1
            self._hypu = self._HypersphericalUniform(dim, device=device)
        return self._hypu

    @torch.no_grad()
    def sample_spherical_ddpm(self, num_samples: int, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        hypu = self._get_hypu(device)
        steps = int(self.ddpm_steps)
        if steps <= 0:
            raise ValueError("ddpm_steps must be positive to sample Spherical-DDPM")

        theta_schedule = torch.linspace(0.0, math.pi / 2.0, steps + 1, device=device)[1:]
        z_t = hypu.sample(num_samples).to(device)

        for t_int in range(steps, 0, -1):
            theta_t = theta_schedule[t_int - 1]
            t_idx = torch.full((num_samples,), t_int, dtype=torch.long, device=device)
            z0_pred = self.denoiser(z_t, t_idx)
            z0_pred = F.normalize(z0_pred, p=2, dim=1)
            z_prev = torch.cos(theta_t) * z_t + torch.sin(theta_t) * z0_pred
            z_t = F.normalize(z_prev, p=2, dim=1)

        return z_t

    def train_mv_vmf(
        self,
        features: torch.Tensor,
        adj_norm_list: List[torch.Tensor],
        adj_label_list: List[torch.Tensor],
        y: np.ndarray,
        weight_tensors: List[torch.Tensor],
        norms: List[float],
        optimizer: str = "Adam",
        epochs: int = 300,
        lr: float = 5e-3,
        kappa_lr: float = 2e-3,
        save_path: str | None = None,
        dataset: str = "ogbn-arxiv-mv-vmf",
        verbose: bool = True,
        pos_per_step: int = 4_000,
        neg_ratio: float = 1.0,
        steps_per_epoch: int = 1,
        pair_micro_bs: int = 500,
        ddpm_node_bs: int = 8192,
    ) -> Tuple[np.ndarray, np.ndarray]:

        VonMisesFisher = self._VonMisesFisher
        HypersphericalUniform = self._HypersphericalUniform
        VMFMixture = self._VMFMixture
        Clustering_Metrics = self._Clustering_Metrics

        assert len(adj_norm_list) == self.base.num_views
        assert len(adj_label_list) == self.base.num_views

        base_params = [p for n, p in self.base.named_parameters() if n != "log_kappa"]
        denoiser_params = list(self.denoiser.parameters())
        optim_cls = torch.optim.Adam if optimizer == "Adam" else torch.optim.SGD
        opt = optim_cls(
            [
                {"params": base_params, "lr": lr},
                {"params": [self.base.log_kappa], "lr": kappa_lr},
                {"params": denoiser_params, "lr": lr},
            ],
            **({"momentum": 0.9} if optimizer == "SGD" else {}),
        )

        device = getattr(features, "device", torch.device("cpu"))

        vmf_cluster = VMFMixture(n_cluster=self.nClusters, max_iter=100)
        bar = tqdm(range(epochs), desc="BHyDiff", disable=not verbose)

        hypu = self._get_hypu(device)

        pos_row_full_views: List[torch.Tensor] = []
        pos_col_full_views: List[torch.Tensor] = []
        for al in adj_label_list:
            if not isinstance(al, torch.Tensor):
                raise TypeError("adj_label_list should contain torch sparse tensors (CPU) for ogbn-arxiv.")
            if al.is_sparse:
                al_c = al.coalesce()
                row_all, col_all = al_c.indices()
            else:
                row_all, col_all = al.nonzero(as_tuple=True)

            mask_offdiag = row_all != col_all
            row_all = row_all[mask_offdiag]
            col_all = col_all[mask_offdiag]
            undup_mask = row_all < col_all
            pos_row = row_all[undup_mask].cpu()
            pos_col = col_all[undup_mask].cpu()
            pos_row_full_views.append(pos_row)
            pos_col_full_views.append(pos_col)

        if len(pos_row_full_views) > 0:
            pos_row_fused = torch.cat(pos_row_full_views, dim=0)
            pos_col_fused = torch.cat(pos_col_full_views, dim=0)
            num_pos_fused = int(pos_row_fused.numel())
        else:
            pos_row_fused = torch.empty(0, dtype=torch.long)
            pos_col_fused = torch.empty(0, dtype=torch.long)
            num_pos_fused = 0

        N = int(adj_norm_list[0].size(0))

        bce = nn.BCELoss(reduction="mean")

        for ep in bar:
            opt.zero_grad()

            mus_views: List[torch.Tensor] = []
            logs_views: List[torch.Tensor] = []
            embed_views: List[torch.Tensor] = []
            for v in range(self.base.num_views):
                mu_v, logs_v = self.base.encode_once(features, adj_norm_list[v])
                mus_views.append(mu_v)
                logs_views.append(logs_v)
                embed_views.append(mu_v)

            w_views = F.softmax(self.base.view_logits[: self.base.num_views], dim=0)
            z_stack = torch.stack(embed_views, dim=0)
            z = (w_views[:, None, None] * z_stack).sum(dim=0)
            z_u = F.normalize(z, 2, 1)

            kappa_val = self.kappa()
            kappa_b = kappa_val.view(1, 1).expand(z_u.size(0), 1)
            qz = VonMisesFisher(z_u, kappa_b)
            pz = HypersphericalUniform(z_u.size(1) - 1, device=device)
            loss_kl = torch.distributions.kl.kl_divergence(qz, pz).mean()

            p = self.base.assignment(z_u)
            centers = F.normalize(self.base.assignment.cluster_centers, 2, 1)
            intra = ((p[:, :, None] * (z_u[:, None, :] - centers[None, :, :]).pow(2))).sum() / z_u.size(0)
            inter = torch.pdist(centers, 2).mean()
            loss_clu = intra / (inter + 1e-9)
            loss_ent = (p * torch.log(p + 1e-9)).sum() / p.size(0)

            embed_views_u = [F.normalize(zv, 2, 1) for zv in embed_views]
            loss_cons = self.base.consistency_weight * self.base.consistency_loss(z_u, embed_views_u)

            loss_rec_total = torch.tensor(0.0, device=device)
            loss_aln_total = torch.tensor(0.0, device=device)
            steps = max(1, int(steps_per_epoch))

            for _ in range(steps):
                for v in range(self.base.num_views):
                    pos_row_full = pos_row_full_views[v]
                    pos_col_full = pos_col_full_views[v]
                    num_pos_total = int(pos_row_full.numel())
                    if num_pos_total == 0:
                        continue

                    k_pos = min(int(pos_per_step), num_pos_total)
                    idx = torch.randint(0, num_pos_total, (k_pos,), device=pos_row_full.device)
                    pr = pos_row_full[idx].to(device, non_blocking=True)
                    pc = pos_col_full[idx].to(device, non_blocking=True)

                    num_neg = int(k_pos * float(neg_ratio))
                    nr = torch.randint(0, N, (num_neg,), device=device)
                    nc = torch.randint(0, N, (num_neg,), device=device)
                    mask = nr != nc
                    nr = nr[mask]
                    nc = nc[mask]
                    if nr.numel() > num_neg:
                        nr = nr[:num_neg]
                        nc = nc[:num_neg]

                    mu_v = mus_views[v]
                    logs_v = logs_views[v]

                    pos_base = self.base.decode_diffusion_pairs_stream(
                        mu_v, logs_v, pr, pc, start=1, micro_bs=pair_micro_bs
                    )
                    neg_base = self.base.decode_diffusion_pairs_stream(
                        mu_v, logs_v, nr, nc, start=1, micro_bs=pair_micro_bs
                    )

                    if self.vmf_strength > 0.0:
                        z_v = embed_views[v]
                        with torch.no_grad():
                            pos_vmf = vmf_decode_pairs_stream(
                                z_v,
                                steps=self.vmf_steps,
                                hypu_dist=hypu,
                                row_idx=pr,
                                col_idx=pc,
                                start=1,
                                micro_bs=pair_micro_bs,
                            )
                            neg_vmf = vmf_decode_pairs_stream(
                                z_v,
                                steps=self.vmf_steps,
                                hypu_dist=hypu,
                                row_idx=nr,
                                col_idx=nc,
                                start=1,
                                micro_bs=pair_micro_bs,
                            )
                        pos_score = (1.0 - self.vmf_strength) * pos_base + self.vmf_strength * pos_vmf
                        neg_score = (1.0 - self.vmf_strength) * neg_base + self.vmf_strength * neg_vmf
                    else:
                        pos_score = pos_base
                        neg_score = neg_base

                    loss_rec_v = bce(pos_score, torch.ones_like(pos_score)) + bce(
                        neg_score, torch.zeros_like(neg_score)
                    )
                    loss_rec_total = loss_rec_total + loss_rec_v

                if num_pos_fused > 0:
                    k_align = min(int(pos_per_step), num_pos_fused)
                    idx = torch.randint(0, num_pos_fused, (k_align,), device=pos_row_fused.device)
                    pr_f = pos_row_fused[idx].to(device, non_blocking=True)
                    pc_f = pos_col_fused[idx].to(device, non_blocking=True)

                    num_neg_align = int(k_align * float(neg_ratio))
                    nr_a = torch.randint(0, N, (num_neg_align,), device=device)
                    nc_a = torch.randint(0, N, (num_neg_align,), device=device)
                    mask_a = nr_a != nc_a
                    nr_a = nr_a[mask_a]
                    nc_a = nc_a[mask_a]
                    if nr_a.numel() > num_neg_align:
                        nr_a = nr_a[:num_neg_align]
                        nc_a = nc_a[:num_neg_align]

                    loss_aln_step = self.base.align_loss_pairs(
                        z_u,
                        pr_f,
                        pc_f,
                        nr_a,
                        nc_a,
                        alpha=self.base.align_alpha,
                        margin=self.base.align_margin,
                    )
                    loss_aln_total = loss_aln_total + loss_aln_step

            denom_rec = max(1, steps * self.base.num_views)
            denom_aln = max(1, steps)
            loss_rec_avg = loss_rec_total / float(denom_rec)
            loss_aln_avg = loss_aln_total / float(denom_aln)

            ddpm_loss = torch.tensor(0.0, device=device)
            if self.ddpm_weight > 0.0 and self.ddpm_steps > 0:
                steps_ddpm = int(self.ddpm_steps)
                theta_schedule = torch.linspace(0.0, math.pi / 2.0, steps_ddpm + 1, device=device)[1:]

                if z_u.size(0) > 50000:
                    m = min(int(ddpm_node_bs), int(z_u.size(0)))
                    idx_nodes = torch.randint(0, z_u.size(0), (m,), device=device)
                    z0 = z_u.detach()[idx_nodes]
                else:
                    z0 = z_u.detach()

                t_int = int(torch.randint(1, steps_ddpm + 1, (1,), device=device).item())

                z_prev = z0.clone()
                for tau in range(1, t_int + 1):
                    theta_tau = theta_schedule[tau - 1]
                    v_tau = hypu.sample(z_prev.size(0)).to(device)
                    z_next = torch.cos(theta_tau) * z_prev + torch.sin(theta_tau) * v_tau
                    z_prev = F.normalize(z_next, p=2, dim=1)
                z_t = z_prev

                t_idx = torch.full((z_t.size(0),), t_int, dtype=torch.long, device=device)
                z0_pred = self.denoiser(z_t, t_idx)
                z0_pred = F.normalize(z0_pred, p=2, dim=1)

                cos_sim = (z0_pred * z0).sum(dim=1)
                ddpm_loss = (1.0 - cos_sim).mean()

            loss = (
                loss_rec_avg
                + self.base.kl_hyp_weight * loss_kl
                + self.base.align_weight * loss_aln_avg
                + self.base.cluster_reg_weight * loss_clu
                + self.base.entropy_reg_weight * loss_ent
                + loss_cons
                + self.ddpm_weight * ddpm_loss
            )

            loss.backward()
            opt.step()

            vmf_cluster.fit(z_u.detach().cpu().numpy())
            y_pred = vmf_cluster.labels_
            metrics_main = Clustering_Metrics(y, y_pred)
            _, nmi_main, ari_main, *_ = metrics_main.evaluationClusterModelFromLabel()

            with torch.no_grad():
                z_last = vmf_forward_last(z_u, steps=self.vmf_steps, hypu_dist=hypu)
                z_last_np = z_last.cpu().numpy()
            vmf_cluster2 = VMFMixture(n_cluster=self.nClusters, max_iter=100)
            vmf_cluster2.fit(z_last_np)
            y_pred_vmf = vmf_cluster2.labels_
            metrics_vmf = Clustering_Metrics(y, y_pred_vmf)
            acc_vmf_eval, *_ = metrics_vmf.evaluationClusterModelFromLabel()

            if verbose:
                bar.set_description(
                    f"BHyDiff loss={loss.item():.4f} "
                    f"rec={loss_rec_avg.item():.4f} kl={loss_kl.item():.4f} "
                    f"aln={loss_aln_avg.item():.4f} clu={loss_clu.item():.4f} "
                    f"ent={loss_ent.item():.4f} ddpm={ddpm_loss.item():.4f} "
                )

        return y_pred, y
