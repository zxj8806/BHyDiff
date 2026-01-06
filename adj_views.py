from __future__ import annotations

from typing import Any, List

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def _symmetrize_binary(adj: sp.coo_matrix) -> sp.coo_matrix:
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    A = adj.copy().tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocoo()


def _build_twohop_truncated(A_csr: sp.csr_matrix, max_2hop: int) -> sp.coo_matrix:

    n = A_csr.shape[0]
    rows: List[int] = []
    cols: List[int] = []

    indptr = A_csr.indptr
    indices = A_csr.indices

    for i in range(n):
        start_i, end_i = indptr[i], indptr[i + 1]
        neigh_i = indices[start_i:end_i]
        if neigh_i.size == 0:
            continue

        neigh_i_set = set(int(x) for x in neigh_i.tolist())
        second: set[int] = set()

        for j in neigh_i:
            j = int(j)
            sj, ej = indptr[j], indptr[j + 1]
            nbr_j = indices[sj:ej]
            for t in nbr_j:
                t = int(t)
                if t == i:
                    continue
                if t in neigh_i_set:
                    continue
                if t not in second:
                    second.add(t)
                    if len(second) >= max_2hop:
                        break
            if len(second) >= max_2hop:
                break

        for t in second:
            rows.append(i)
            cols.append(t)
            rows.append(t)
            cols.append(i)

    if len(rows) == 0:
        return sp.coo_matrix(A_csr.shape, dtype=np.float32)

    data = np.ones(len(rows), dtype=np.float32)
    A2 = sp.coo_matrix((data, (rows, cols)), shape=A_csr.shape)
    A2 = _symmetrize_binary(A2)
    return A2


def build_multiview_adjs(
    adj: sp.csr_matrix,
    num_views: int = 3,
    degree_knn_k: int = 10,
    feature_knn_k: int = 10,
    features: Any = None,
    twohop_exact_threshold: int = 20000,
) -> List[sp.coo_matrix]:

    if not sp.isspmatrix(adj):
        raise TypeError("adj must be a scipy sparse matrix")

    A0 = _symmetrize_binary(adj.tocoo())
    n = A0.shape[0]
    views: List[sp.coo_matrix] = []

    views.append(A0.tocoo())

    if num_views >= 2:
        A_csr = A0.tocsr()
        if n <= twohop_exact_threshold:
            A2 = A_csr @ A_csr
            A2.setdiag(0)
            A2 = (A2 > 0).astype(np.float32)
            A2.eliminate_zeros()
            views.append(A2.tocoo())
        else:
            A2_trunc = _build_twohop_truncated(A_csr, max_2hop=degree_knn_k)
            views.append(A2_trunc.tocoo())

    if num_views >= 3:
        if features is not None:
            if sp.isspmatrix(features):
                X = features.toarray()
            else:
                X = np.asarray(features)

            knn = NearestNeighbors(n_neighbors=feature_knn_k + 1, metric="cosine").fit(X)
            _, idxs = knn.kneighbors(X)
            rows, cols = [], []
            for i in range(n):
                for j in idxs[i, 1:]:
                    rows.append(i)
                    cols.append(int(j))
                    rows.append(int(j))
                    cols.append(i)
            data = np.ones(len(rows), dtype=np.float32)
            A_knn = sp.coo_matrix((data, (rows, cols)), shape=A0.shape)
            A_knn = _symmetrize_binary(A_knn)
        else:
            deg = np.array(A0.sum(1)).flatten()
            order = np.argsort(deg)
            row, col = [], []
            for idx, node in enumerate(order):
                for offset in range(1, degree_knn_k + 1):
                    jdx = idx + offset
                    if jdx >= n:
                        break
                    nb = order[jdx]
                    row.append(int(node))
                    col.append(int(nb))
                    row.append(int(nb))
                    col.append(int(node))
            if row:
                data = np.ones(len(row), dtype=np.float32)
                A_knn = sp.coo_matrix((data, (row, col)), shape=A0.shape)
                A_knn = _symmetrize_binary(A_knn)
            else:
                A_knn = sp.coo_matrix(A0.shape, dtype=np.float32)

        views.append(A_knn.tocoo())

    return views[:num_views]
