from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp
import torch

from preprocessing import load_ogbn_arxiv, sparse_to_tuple, preprocess_graph
from adj_views import build_multiview_adjs
from bhydiff import BHyDiff


def main():
    dataset = "ogbn-arxiv"
    print("Loading dataset:", dataset)

    adj, features, labels = load_ogbn_arxiv()
    nClusters = int(labels.max() + 1)

    alpha = 1.0
    gamma_1 = 1.0
    gamma_2 = 1.0
    gamma_3 = 1.0

    num_neurons = 128
    embedding_size = 128
    save_path = "./results/"

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    num_views = 3
    adj_views = build_multiview_adjs(
        adj,
        num_views=num_views,
        degree_knn_k=10,
        feature_knn_k=10,
        features=features,
        twohop_exact_threshold=20000,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    features_tuple = sparse_to_tuple(features.tocoo())
    num_features = int(features_tuple[2][1])
    features_t = torch.sparse_coo_tensor(
        torch.LongTensor(features_tuple[0].T),
        torch.FloatTensor(features_tuple[1]),
        torch.Size(features_tuple[2]),
    ).to(device)

    adj_norm_list_t: List[torch.Tensor] = []
    adj_label_list_t: List[torch.Tensor] = []
    weight_tensors: List[torch.Tensor] = []
    norms: List[float] = []

    for adj_v in adj_views:
        adj_v = adj_v - sp.dia_matrix((adj_v.diagonal()[np.newaxis, :], [0]), shape=adj_v.shape)
        adj_v.eliminate_zeros()

        adj_norm = preprocess_graph(adj_v)
        adj_norm_t = torch.sparse_coo_tensor(
            torch.LongTensor(adj_norm[0].T),
            torch.FloatTensor(adj_norm[1]),
            torch.Size(adj_norm[2]),
        ).to(device)
        adj_norm_list_t.append(adj_norm_t)

        adj_label = adj_v + sp.eye(adj_v.shape[0])
        adj_label_tuple = sparse_to_tuple(adj_label)
        adj_label_t = torch.sparse_coo_tensor(
            torch.LongTensor(adj_label_tuple[0].T),
            torch.FloatTensor(adj_label_tuple[1]),
            torch.Size(adj_label_tuple[2]),
        )
        adj_label_list_t.append(adj_label_t)

        norms.append(1.0)
        weight_tensors.append(torch.tensor([1.0]))

    print("==> BHyDiff")

    network = BHyDiff(
        vmf_strength=0.35,
        ddpm_weight=0.1,
        num_views=len(adj_views),
        consis_weight=0.1,
        num_neurons=num_neurons,
        num_features=num_features,
        embedding_size=embedding_size,
        nClusters=nClusters,
        activation="ReLU",
        alpha=alpha,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
        T=10,
    ).to(device)

    _y_pred, _y_true = network.train_mv_vmf(
        features_t,
        adj_norm_list_t,
        adj_label_list_t,
        labels,
        weight_tensors,
        norms,
        optimizer="Adam",
        epochs=120,
        lr=1e-4,
        kappa_lr=1e-3,
        dataset=dataset + "-mv-vmf",
        save_path=save_path,
        pos_per_step=4_000,
        neg_ratio=1.0,
        steps_per_epoch=1,
        pair_micro_bs=500,
        ddpm_node_bs=8192,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()
