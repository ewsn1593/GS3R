"""
GS3R merging routines.

``gs3r_merge*`` started from the TSV merging routine in FusionBench
(fusion_bench/method/task_singular_vector; Gargiulo et al., CVPR 2025, Apache-2.0).
``wudi_merging*`` minimize the objective of WUDI-Merging (Cheng et al., ICML 2025, MIT).
"""

import copy
import math
import random

from typing import List, Optional
from tqdm import tqdm
import torch

from fusion_bench.utils.type import StateDictType




def canon_svd(target, full_matrices=False, k=-1):
    u, s, v = torch.linalg.svd(target, full_matrices=full_matrices)

    idx = u.abs().argmax(dim=0)
    signs = torch.sign(u[idx, torch.arange(u.size(1))])
    signs[signs == 0] = 1.0

    u_ = u * signs[None, :]
    v_ = v * signs[:, None]


    if (k != -1) and (s.shape[0] > k): # only when current_rank > k
        u_ = u_[:, :k]
        s = s[:k]
        v_ = v_[:k, :]

    return u_, s, v_


def change_sign(u,v):

    idx = u.abs().argmax(dim=0)
    signs = torch.sign(u[idx, torch.arange(u.size(1))])
    signs[signs == 0] = 1.0

    u_ = u * signs[None, :]
    v_ = v * signs[:, None]
    return u_, v_


def numeric_cutoff(x, degree=1e06):
    x = x * degree
    x = x.round()
    x = x / degree
    return x

def choose_R(u, U, v, V):
    R_fromU = torch.linalg.multi_dot((u.T, U))
    R_fromV = torch.linalg.multi_dot((v, V.T))

    err_fromU = (V - (R_fromU.T @ v)).abs().sum()
    err_fromV = (U - (u @ R_fromV)).abs().sum()

    R = R_fromU if err_fromU < err_fromV else R_fromV

    return R



def save_symmetric(X):
    assert(X.shape[0] == X.shape[1])
    X_ = X.triu()
    return X_

def restore_symmetric(X):
    assert(X.shape[0] == X.shape[1])
    X_ = X + X.triu(1).T
    return X_



################################################################################################

def wudi_merging(
    sum_vec,
    task_vectors: List[torch.Tensor],
    iter_num: int = 300,
):


    tvs = torch.stack(
        [
            task_vector.detach()
            for task_vector in task_vectors
        ]
    )
    num_tvs = len(tvs)

    optimizer = torch.optim.Adam([sum_vec], lr=1e-5, weight_decay=0)
    l2_norms = torch.square(
        torch.norm(tvs.reshape(tvs.shape[0], -1), p=2, dim=-1)
    )
    for i in tqdm(
        range(iter_num),
    ):
        disturbing_vectors = sum_vec.unsqueeze(0) - tvs
        product = torch.matmul(disturbing_vectors, tvs.transpose(1, 2))
        loss = torch.sum(
            torch.square(product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum_vec.detach()






def wudi_merging_flan(
    sum_vec,
    task_vectors: List[torch.Tensor],
    iter_num: int = 300,
):


    tvs = torch.stack(
        [
            task_vector.detach()
            for task_vector in task_vectors
        ]
    )
    num_tvs = len(tvs)

    optimizer = torch.optim.Adam([sum_vec], lr=1e-4, weight_decay=0)
    l2_norms = torch.square(
        torch.norm(tvs.reshape(tvs.shape[0], -1), p=2, dim=-1)
    )
    for i in tqdm(
        range(iter_num),
    ):
        disturbing_vectors = sum_vec.unsqueeze(0) - tvs
        product = torch.matmul(disturbing_vectors, tvs.transpose(1, 2))
        loss = torch.sum(
            torch.square(product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum_vec.detach()










def _kmeans_gpu(Y, k, n_iter=30, tol=1e-6):
    """GPU-resident k-means with k-means++ init. Y: [N, K] row-normalized."""
    N, D = Y.shape
    device = Y.device
 
    g = torch.Generator(device=device).manual_seed(0)
    first = torch.randint(N, (1,), generator=g, device=device).item()
    centers = Y[first:first + 1].clone()
    for _ in range(k - 1):
        d2 = torch.cdist(Y, centers).pow(2).min(dim=1).values
        s = d2.sum()
        if s.item() <= 0:
            idx = torch.randint(N, (1,), generator=g, device=device).item()
        else:
            probs = d2 / s
            idx = torch.multinomial(probs, 1, generator=g).item()
        centers = torch.cat([centers, Y[idx:idx + 1]], dim=0)
 
    prev_labels = None
    for _ in range(n_iter):
        d2 = torch.cdist(Y, centers).pow(2)
        labels = d2.argmin(dim=1)
        if prev_labels is not None and torch.equal(labels, prev_labels):
            break
        prev_labels = labels
 
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device, dtype=Y.dtype)
        new_centers.index_add_(0, labels, Y)
        counts.index_add_(0, labels, torch.ones(N, device=device, dtype=Y.dtype))
        nonempty = counts > 0
        new_centers[nonempty] = new_centers[nonempty] / counts[nonempty].unsqueeze(1)
        new_centers[~nonempty] = centers[~nonempty]
 
        if (new_centers - centers).abs().max() < tol:
            centers = new_centers
            break
        centers = new_centers
    return labels
 
 
def separate_tasks(mer_u, mer_s, mer_v, ortho_tol=1e-4,
                   k=None, verbose=False):
    """
    Cluster SVD components into tasks. Within-task components are mutually
    orthogonal (same SVD factor) while cross-task components are not.
 
    Affinity is the orthogonality indicator
        W_ij = 1[|<u_i, u_j>| < ortho_tol].
    K is selected by the eigengap heuristic on the symmetric normalized
    Laplacian.
    """
    device = mer_u.device
 
    norms = mer_u.norm(dim=0)
    valid_idx = torch.nonzero(norms > 1e-8, as_tuple=False).flatten()
    U = mer_u.index_select(1, valid_idx)
    Un = U / U.norm(dim=0, keepdim=True).clamp_min(1e-12)
    N = U.shape[1]
 
    Cu = (Un.T @ Un).abs()
    W = (Cu < ortho_tol).float()
    W.fill_diagonal_(0.0)
 
    d = W.sum(dim=1).clamp_min(1e-12)
    D_inv_sqrt = d.pow(-0.5)
    L = torch.eye(N, device=device) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])
    L = 0.5 * (L + L.T)
 
    eigvals, eigvecs = torch.linalg.eigh(L)
 
    if k is None:
        gaps = eigvals[1:] - eigvals[:-1]
        k = int(gaps.argmax().item()) + 1
        if verbose:
            top = min(N, 25)
            print(f"[separate_tasks] N = {N}")
            print(f"[separate_tasks] eigvals[:{top}] = {eigvals[:top].cpu().numpy()}")
            print(f"[separate_tasks] top gap = {gaps.max().item():.4f} at K = {k}")
 
    Y = eigvecs[:, :k]
    Y = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-12)
    labels_t = _kmeans_gpu(Y, k)            # <-- GPU 경로
 
    valid_idx_t = valid_idx.to(device)
    sep_u, sep_s, sep_v = [], [], []
    for c in range(k):
        local = torch.nonzero(labels_t == c, as_tuple=False).flatten()
        if local.numel() == 0:
            continue
        idx = valid_idx_t.index_select(0, local)
        s_c = mer_s.index_select(0, idx)
        order = torch.argsort(s_c, descending=True)
        idx = idx.index_select(0, order)
        sep_u.append(mer_u.index_select(1, idx))
        sep_s.append(mer_s.index_select(0, idx))
        sep_v.append(mer_v.index_select(0, idx))
 
    return sep_u, sep_s, sep_v, len(sep_u)


#############################################################################################################

def gs3r_merge( 
    task_vectors: List[StateDictType],
    exclude_keys: Optional[List[str]] = None,
    accelerator: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    procrustes: bool = True,
    idx: int = 0,
    singular_values = {},
) -> StateDictType:

    print("Computing SVD...")
    print(f"Procrustes : {procrustes}")
    assert(len(task_vectors) == 2) # For only two (inner algorithm)

    if exclude_keys is None:
        exclude_keys = []

    new_vector = {}
    mer_vec = task_vectors[0] 
    tar_vec = task_vectors[1] 

    for key in mer_vec:
        original_device = mer_vec[key].device
        original_dtype = mer_vec[key].dtype

        # SVD
        if (
            len(mer_vec[key].shape) == 2
            and key not in exclude_keys
        ):
            
            if not (
                    original_dtype == torch.float32 or original_dtype == torch.float64
                ):
                    mer_vec[key] = mer_vec[key].to(dtype=torch.float32)
                    tar_vec[key] = tar_vec[key].to(dtype=torch.float32)
                    
            mer_u, mer_s, mer_v = canon_svd(mer_vec[key].to(accelerator), full_matrices=False)
            tar_u, tar_s, tar_v = canon_svd(tar_vec[key].to(accelerator), full_matrices=False)

            #Singular Vector restoring
            if key in singular_values.keys():
                storage = singular_values[key]  
                R_u = storage[0]
                R_v = storage[1]

                R_u = restore_symmetric(R_u)
                R_v = restore_symmetric(R_v)

                R_u = R_u.to('cuda') 
                mer_u = torch.linalg.multi_dot((mer_u, R_u))
                R_u = R_u.to('cpu') 

                R_v = R_v.to('cuda') 
                if mer_v.shape[0] < mer_v.shape[1]:
                    mer_v = torch.linalg.multi_dot((R_v, mer_v))
                else:
                    mer_v = torch.linalg.multi_dot((mer_v, R_v))
                R_v = R_v.to('cpu') 


            sep_u, sep_s, sep_v, task_num = separate_tasks(mer_u, mer_s, mer_v)
            k = max(1, mer_s.shape[0] // (task_num + 1))
            
            sum_u = torch.cat((*[u[:, :k] for u in sep_u], tar_u[:, :k]), dim=1)
            sum_s = torch.cat((*[s[:k]    for s in sep_s], tar_s[:k]),    dim=0)
            sum_v = torch.cat((*[v[:k, :] for v in sep_v], tar_v[:k, :]), dim=0)

            if sum_s.shape[0] != mer_s.shape[0]:
                margin = mer_s.shape[0] - sum_s.shape[0]
                
                sort_result = torch.topk(
                    torch.cat((*[s[k:] for s in sep_s], tar_s[k:]), dim=0), 
                    margin)
                
                cover_u = torch.cat((*[u[:, k:] for u in sep_u], tar_u[:, k:]), dim=1)[:, sort_result.indices] 
                cover_s = sort_result.values
                cover_v = torch.cat((*[v[k:, :] for v in sep_v], tar_v[k:, :]), dim=0)[sort_result.indices, :]

                sum_u = torch.cat((sum_u, cover_u), dim=1)
                sum_s = torch.cat((sum_s, cover_s), dim=0)
                sum_v = torch.cat((sum_v, cover_v), dim=0)

            last_sort = torch.topk(sum_s, sum_s.shape[0])
            sum_u = sum_u[:, last_sort.indices]
            sum_s = last_sort.values
            sum_v = sum_v[last_sort.indices, :]

            if procrustes:
                u_u, s_u, v_u = canon_svd(sum_u, full_matrices=False) 
                u_v, s_v, v_v = canon_svd(sum_v, full_matrices=False) 

                
                # Procrustes Solution and Rotation Vector Storing
                pro_u = torch.linalg.multi_dot((u_u, v_u))
                pro_v = torch.linalg.multi_dot((u_v, v_v))
                pro_u, pro_v = change_sign(pro_u, pro_v) 

                recon = torch.linalg.multi_dot((pro_u, torch.diag(sum_s), pro_v))
                
                R_u = choose_R(pro_u, u_u, torch.eye(v_u.shape[1], device=v_u.device), v_u)
                RsR_u = torch.linalg.multi_dot((R_u, torch.diag(s_u), R_u.T))

                RsR_u = RsR_u.to('cpu') 

                if pro_v.shape[0] < pro_v.shape[1]:
                    R_v = choose_R(torch.eye(u_v.shape[1], device=u_v.device), u_v, pro_v, v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T))    
                else: 
                    R_v = choose_R(pro_v, u_v, torch.eye(v_v.shape[1], device=v_v.device), v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T)) 

                RsR_v = RsR_v.to('cpu') 
                torch.cuda.empty_cache() 
                

                RsR_u = save_symmetric(RsR_u)
                RsR_v = save_symmetric(RsR_v) 

                
                # Save Data
                new_vector[key] = recon
                singular_values[key] = [RsR_u, RsR_v]

                del(sep_u)
                del(sep_s)
                del(sep_v)
                torch.cuda.empty_cache()

            else:
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v
                    )
                )
            
        else:
            new_vector[key] = (mer_vec[key] + tar_vec[key]) / 2

        new_vector[key] = new_vector[key].to(
            device=original_device, dtype=original_dtype, non_blocking=True
        )

    return new_vector, singular_values

def gs3r_merge_trunc( 
    task_vectors: List[StateDictType],
    exclude_keys: Optional[List[str]] = None,
    accelerator: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    procrustes: bool = True,
    idx: int = 0,
    singular_values = {},
    k_s = 128, # -1 for full storage use
) -> StateDictType:

    print("Computing SVD...")
    print(f"Procrustes : {procrustes}")
    assert(len(task_vectors) == 2) # For only two (inner algorithm)

    if exclude_keys is None:
        exclude_keys = []

    new_vector = {}
    mer_vec = task_vectors[0] 
    tar_vec = task_vectors[1] 

    for key in mer_vec:
        original_device = mer_vec[key].device
        original_dtype = mer_vec[key].dtype

        # SVD
        if (
            len(mer_vec[key].shape) == 2
            and key not in exclude_keys
        ):
            
            if not (
                    original_dtype == torch.float32 or original_dtype == torch.float64
                ):
                    mer_vec[key] = mer_vec[key].to(dtype=torch.float32)
                    tar_vec[key] = tar_vec[key].to(dtype=torch.float32)
                    
            mer_u, mer_s, mer_v = canon_svd(mer_vec[key].to(accelerator), full_matrices=False, k=k_s)
            tar_u, tar_s, tar_v = canon_svd(tar_vec[key].to(accelerator), full_matrices=False, k=k_s)

            #Singular Vector restoring
            if key in singular_values.keys():
                storage = singular_values[key]  
                R_u = storage[0]
                R_v = storage[1]

                R_u = restore_symmetric(R_u).to('cuda')
                R_v = restore_symmetric(R_v).to('cuda')

                if mer_u.shape[0] < mer_u.shape[1]:
                    mer_u = torch.linalg.multi_dot((R_u, mer_u)) 
                else:
                    mer_u = torch.linalg.multi_dot((mer_u, R_u))

                if mer_v.shape[0] < mer_v.shape[1]:
                    mer_v = torch.linalg.multi_dot((R_v, mer_v))
                else:
                    mer_v = torch.linalg.multi_dot((mer_v, R_v))


            sep_u, sep_s, sep_v, task_num = separate_tasks(mer_u, mer_s, mer_v)
            k_cur = mer_s.shape[0] if mer_s.shape[0] < k_s else k_s # pre-defined rank k or min shape for current weight
            k = max(1, k_cur // (task_num + 1)) # Each Task rank
                        
            sum_u = torch.cat((*[u[:, :k] for u in sep_u], tar_u[:, :k]), dim=1)
            sum_s = torch.cat((*[s[:k]    for s in sep_s], tar_s[:k]),    dim=0)
            sum_v = torch.cat((*[v[:k, :] for v in sep_v], tar_v[:k, :]), dim=0)

            if sum_s.shape[0] != k_cur:
                margin = k_cur - sum_s.shape[0]
                
                sort_result = torch.topk(
                    torch.cat((*[s[k:] for s in sep_s], tar_s[k:]), dim=0), 
                    margin)
                
                cover_u = torch.cat((*[u[:, k:] for u in sep_u], tar_u[:, k:]), dim=1)[:, sort_result.indices] 
                cover_s = sort_result.values
                cover_v = torch.cat((*[v[k:, :] for v in sep_v], tar_v[k:, :]), dim=0)[sort_result.indices, :]

                sum_u = torch.cat((sum_u, cover_u), dim=1)
                sum_s = torch.cat((sum_s, cover_s), dim=0)
                sum_v = torch.cat((sum_v, cover_v), dim=0)

            last_sort = torch.topk(sum_s, sum_s.shape[0])
            sum_u = sum_u[:, last_sort.indices]
            sum_s = last_sort.values
            sum_v = sum_v[last_sort.indices, :]

            if procrustes:
                u_u, s_u, v_u = canon_svd(sum_u, full_matrices=False) 
                u_v, s_v, v_v = canon_svd(sum_v, full_matrices=False) 

                
                # Procrustes Solution and Rotation Vector Storing
                pro_u = torch.linalg.multi_dot((u_u, v_u))
                pro_v = torch.linalg.multi_dot((u_v, v_v))
                pro_u, pro_v = change_sign(pro_u, pro_v) 

                recon = torch.linalg.multi_dot((pro_u, torch.diag(sum_s), pro_v))
                
                if pro_u.shape[0] < pro_u.shape[1]:
                    R_u = choose_R(torch.eye(u_u.shape[1], device=u_u.device), u_u, pro_u, v_u)
                    RsR_u = torch.linalg.multi_dot((R_u, torch.diag(s_u), R_u.T))  
                else:
                    R_u = choose_R(pro_u, u_u, torch.eye(v_u.shape[1], device=v_u.device), v_u)
                    RsR_u = torch.linalg.multi_dot((R_u, torch.diag(s_u), R_u.T))

                if pro_v.shape[0] < pro_v.shape[1]:
                    R_v = choose_R(torch.eye(u_v.shape[1], device=u_v.device), u_v, pro_v, v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T))    
                else: 
                    R_v = choose_R(pro_v, u_v, torch.eye(v_v.shape[1], device=v_v.device), v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T))  

                RsR_u = save_symmetric(RsR_u)
                RsR_v = save_symmetric(RsR_v) 

                
                # Save Data
                new_vector[key] = recon
                singular_values[key] = [RsR_u, RsR_v]

            else:
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v
                    )
                )
            
        else:
            new_vector[key] = (mer_vec[key] + tar_vec[key]) / 2

        new_vector[key] = new_vector[key].to(
            device=original_device, dtype=original_dtype, non_blocking=True
        )

    return new_vector, singular_values






def gs3r_merge_opt( 
    task_vectors: List[StateDictType],
    exclude_keys: Optional[List[str]] = None,
    accelerator: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    procrustes: bool = True,
    idx: int = 0,
    singular_values = {},
    ntask: int = 8,
    iter_num: int = 400,
    opt_only_last: bool = True,
    init_recon: bool = False,
) -> StateDictType:

    print("Computing SVD...")
    print(f"Procrustes : {procrustes}")
    assert(len(task_vectors) == 2) # For only two (inner algorithm)

    if exclude_keys is None:
        exclude_keys = []

    new_vector = {}
    mer_vec = task_vectors[0] 
    tar_vec = task_vectors[1] 
    indexer=True
    for key in mer_vec:
        original_device = mer_vec[key].device
        original_dtype = mer_vec[key].dtype

        # SVD
        if (
            len(mer_vec[key].shape) == 2
            and key not in exclude_keys
        ):
            
            if not (
                    original_dtype == torch.float32 or original_dtype == torch.float64
                ):
                    mer_vec[key] = mer_vec[key].to(dtype=torch.float32)
                    tar_vec[key] = tar_vec[key].to(dtype=torch.float32)
            
            if (key in singular_values.keys()):
                if len(singular_values[key]) == 3:
                    print(f'Restore Optimization Delta of {key}')
                    mer_vec[key] = singular_values[key][2]
                    
            mer_u, mer_s, mer_v = canon_svd(mer_vec[key].to(accelerator), full_matrices=False)
            tar_u, tar_s, tar_v = canon_svd(tar_vec[key].to(accelerator), full_matrices=False)

            #Singular Vector restoring
            if key in singular_values.keys():
                storage = singular_values[key]  
                R_u = storage[0]
                R_v = storage[1]

                R_u = restore_symmetric(R_u)
                R_v = restore_symmetric(R_v)

                R_u = R_u.to('cuda') 
                mer_u = torch.linalg.multi_dot((mer_u, R_u))
                R_u = R_u.to('cpu') 

                R_v = R_v.to('cuda') 
                if mer_v.shape[0] < mer_v.shape[1]:
                    mer_v = torch.linalg.multi_dot((R_v, mer_v))
                else:
                    mer_v = torch.linalg.multi_dot((mer_v, R_v))
                R_v = R_v.to('cpu') 

  

            sep_u, sep_s, sep_v, task_num = separate_tasks(mer_u, mer_s, mer_v)
            k = max(1, mer_s.shape[0] // (task_num + 1))
            tvs = [torch.linalg.multi_dot((u_, torch.diag(s_), v_)) for u_,s_,v_ in zip(sep_u, sep_s, sep_v)] + [tar_vec[key].to(accelerator)]
            

            
            sum_u = torch.cat((*[u[:, :k] for u in sep_u], tar_u[:, :k]), dim=1)
            sum_s = torch.cat((*[s[:k]    for s in sep_s], tar_s[:k]),    dim=0)
            sum_v = torch.cat((*[v[:k, :] for v in sep_v], tar_v[:k, :]), dim=0)

            if sum_s.shape[0] != mer_s.shape[0]:
                margin = mer_s.shape[0] - sum_s.shape[0]
                
                sort_result = torch.topk(
                    torch.cat((*[s[k:] for s in sep_s], tar_s[k:]), dim=0), 
                    margin)
                
                cover_u = torch.cat((*[u[:, k:] for u in sep_u], tar_u[:, k:]), dim=1)[:, sort_result.indices] 
                cover_s = sort_result.values
                cover_v = torch.cat((*[v[k:, :] for v in sep_v], tar_v[k:, :]), dim=0)[sort_result.indices, :]

                sum_u = torch.cat((sum_u, cover_u), dim=1)
                sum_s = torch.cat((sum_s, cover_s), dim=0)
                sum_v = torch.cat((sum_v, cover_v), dim=0)

            last_sort = torch.topk(sum_s, sum_s.shape[0])
            sum_u = sum_u[:, last_sort.indices]
            sum_s = last_sort.values
            sum_v = sum_v[last_sort.indices, :]

            if procrustes:
                u_u, s_u, v_u = canon_svd(sum_u, full_matrices=False) 
                u_v, s_v, v_v = canon_svd(sum_v, full_matrices=False) 

                
                # Procrustes Solution and Rotation Vector Storing
                pro_u = torch.linalg.multi_dot((u_u, v_u))
                pro_v = torch.linalg.multi_dot((u_v, v_v))
                pro_u, pro_v = change_sign(pro_u, pro_v) 

                recon = torch.linalg.multi_dot((pro_u, torch.diag(sum_s), pro_v))
                
                R_u = choose_R(pro_u, u_u, torch.eye(v_u.shape[1], device=v_u.device), v_u)
                RsR_u = torch.linalg.multi_dot((R_u, torch.diag(s_u), R_u.T))

                RsR_u = RsR_u.to('cpu') 

                if pro_v.shape[0] < pro_v.shape[1]:
                    R_v = choose_R(torch.eye(u_v.shape[1], device=u_v.device), u_v, pro_v, v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T))    
                else: 
                    R_v = choose_R(pro_v, u_v, torch.eye(v_v.shape[1], device=v_v.device), v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T)) 

                RsR_v = RsR_v.to('cpu') 
                torch.cuda.empty_cache() 
                

                RsR_u = save_symmetric(RsR_u)
                RsR_v = save_symmetric(RsR_v) 

                
                # Save Data
                new_vector[key] = recon
                singular_values[key] = [RsR_u, RsR_v]

                # Optmized version
                if (opt_only_last and (idx != (ntask-1))):
                    print('Pass Optimization')
                else:
                    recon_ = copy.deepcopy(recon)
                    print(f'{key} - optimization')
                    if init_recon:
                        print(f'Init with Recon - {key}')
                        new_vector[key] = torch.nn.Parameter(recon)
                    else:
                        new_vector[key] = torch.nn.Parameter(torch.linalg.multi_dot((sum_u, torch.diag(sum_s), sum_v)))
                    new_vector[key] = wudi_merging(new_vector[key], tvs, iter_num=iter_num)
                    del(tvs)
                    torch.cuda.empty_cache()

                    singular_values[key].append(recon_.to('cpu')) # Same as saving delta matrix
                    del(recon_)
                    


                del(sep_u)
                del(sep_s)
                del(sep_v)
                torch.cuda.empty_cache()

            else:
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v
                    )
                )
            
        else:
            new_vector[key] = (mer_vec[key] + tar_vec[key]) / 2

        new_vector[key] = new_vector[key].to(
            device=original_device, dtype=original_dtype, non_blocking=True
        )

    return new_vector, singular_values




def gs3r_merge_opt_flan( 
    task_vectors: List[StateDictType],
    exclude_keys: Optional[List[str]] = None,
    accelerator: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    procrustes: bool = True,
    idx: int = 0,
    singular_values = {},
    ntask: int = 8,
    iter_num: int = 400,
    opt_only_last: bool = True,
    init_recon: bool = False,
) -> StateDictType:

    print("Computing SVD...")
    print(f"Procrustes : {procrustes}")
    assert(len(task_vectors) == 2) # For only two (inner algorithm)

    if exclude_keys is None:
        exclude_keys = []

    new_vector = {}
    mer_vec = task_vectors[0] 
    tar_vec = task_vectors[1] 
    indexer=True
    for key in mer_vec:
        original_device = mer_vec[key].device
        original_dtype = mer_vec[key].dtype

        # SVD
        if (
            len(mer_vec[key].shape) == 2
            and key not in exclude_keys
        ):
            
            if not (
                    original_dtype == torch.float32 or original_dtype == torch.float64
                ):
                    mer_vec[key] = mer_vec[key].to(dtype=torch.float32)
                    tar_vec[key] = tar_vec[key].to(dtype=torch.float32)


            if mer_vec[key].abs().sum() == 0:
                if tar_vec[key].abs().sum() == 0:
                    print(f'{key} : Not fine-tuned at all. Skip')
                    new_vector[key] = (mer_vec[key] + tar_vec[key]) / 2
                    new_vector[key] = new_vector[key].to(
                        device=original_device, dtype=original_dtype, non_blocking=True
                    )

                    continue
            
            
            if (key in singular_values.keys()):
                if len(singular_values[key]) == 3:
                    print(f'Restore Optimization Delta of {key}')
                    mer_vec[key] = singular_values[key][2]
                    
            mer_u, mer_s, mer_v = canon_svd(mer_vec[key].to(accelerator), full_matrices=False)
            tar_u, tar_s, tar_v = canon_svd(tar_vec[key].to(accelerator), full_matrices=False)

            #Singular Vector restoring
            if key in singular_values.keys():
                storage = singular_values[key]  
                R_u = storage[0]
                R_v = storage[1]

                R_u = restore_symmetric(R_u)
                R_v = restore_symmetric(R_v)

                R_u = R_u.to('cuda') 
                mer_u = torch.linalg.multi_dot((mer_u, R_u))
                R_u = R_u.to('cpu')

                R_v = R_v.to('cuda') 
                if mer_v.shape[0] < mer_v.shape[1]:
                    mer_v = torch.linalg.multi_dot((R_v, mer_v))
                else:
                    mer_v = torch.linalg.multi_dot((mer_v, R_v))
                R_v = R_v.to('cpu') 


            sep_u, sep_s, sep_v, task_num = separate_tasks(mer_u, mer_s, mer_v)
            k = max(1, mer_s.shape[0] // (task_num + 1))
            tvs = [torch.linalg.multi_dot((u_, torch.diag(s_), v_)) for u_,s_,v_ in zip(sep_u, sep_s, sep_v)] + [tar_vec[key].to(accelerator)]
            

            
            sum_u = torch.cat((*[u[:, :k] for u in sep_u], tar_u[:, :k]), dim=1)
            sum_s = torch.cat((*[s[:k]    for s in sep_s], tar_s[:k]),    dim=0)
            sum_v = torch.cat((*[v[:k, :] for v in sep_v], tar_v[:k, :]), dim=0)

            if sum_s.shape[0] != mer_s.shape[0]:
                margin = mer_s.shape[0] - sum_s.shape[0]
                
                sort_result = torch.topk(
                    torch.cat((*[s[k:] for s in sep_s], tar_s[k:]), dim=0), 
                    margin)
                
                cover_u = torch.cat((*[u[:, k:] for u in sep_u], tar_u[:, k:]), dim=1)[:, sort_result.indices] 
                cover_s = sort_result.values
                cover_v = torch.cat((*[v[k:, :] for v in sep_v], tar_v[k:, :]), dim=0)[sort_result.indices, :]

                sum_u = torch.cat((sum_u, cover_u), dim=1)
                sum_s = torch.cat((sum_s, cover_s), dim=0)
                sum_v = torch.cat((sum_v, cover_v), dim=0)

            last_sort = torch.topk(sum_s, sum_s.shape[0])
            sum_u = sum_u[:, last_sort.indices]
            sum_s = last_sort.values
            sum_v = sum_v[last_sort.indices, :]

            if procrustes:
                u_u, s_u, v_u = canon_svd(sum_u, full_matrices=False) 
                u_v, s_v, v_v = canon_svd(sum_v, full_matrices=False) 

                
                # Procrustes Solution and Rotation Vector Storing
                pro_u = torch.linalg.multi_dot((u_u, v_u))
                pro_v = torch.linalg.multi_dot((u_v, v_v))
                pro_u, pro_v = change_sign(pro_u, pro_v) 

                recon = torch.linalg.multi_dot((pro_u, torch.diag(sum_s), pro_v))
                
                R_u = choose_R(pro_u, u_u, torch.eye(v_u.shape[1], device=v_u.device), v_u)
                RsR_u = torch.linalg.multi_dot((R_u, torch.diag(s_u), R_u.T))

                RsR_u = RsR_u.to('cpu') 

                if pro_v.shape[0] < pro_v.shape[1]:
                    R_v = choose_R(torch.eye(u_v.shape[1], device=u_v.device), u_v, pro_v, v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T))    
                else: 
                    R_v = choose_R(pro_v, u_v, torch.eye(v_v.shape[1], device=v_v.device), v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T)) 

                RsR_v = RsR_v.to('cpu') 
                torch.cuda.empty_cache() 
                

                RsR_u = save_symmetric(RsR_u)
                RsR_v = save_symmetric(RsR_v) 

                
                # Save Data
                new_vector[key] = recon
                singular_values[key] = [RsR_u, RsR_v]

                # Optmized version
                if (opt_only_last and (idx != (ntask-1))):
                    print('Pass Optimization')
                elif 'shared' in key or 'embed_tokens' in key or 'lm_head' in key:
                    print(f'{key} : Embedding layer or head, skip')
                else:
                    recon_ = copy.deepcopy(recon)
                    print(f'{key} - optimization')
                    if init_recon:
                        print(f'Init with Recon - {key}')
                        new_vector[key] = torch.nn.Parameter(recon)
                    else:
                        new_vector[key] = torch.nn.Parameter(torch.linalg.multi_dot((sum_u, torch.diag(sum_s), sum_v)))
                    new_vector[key] = wudi_merging(new_vector[key], tvs, iter_num=iter_num)
                    del(tvs)
                    torch.cuda.empty_cache()


                    singular_values[key].append(recon_.to('cpu')) # Same as saving delta matrix
                    del(recon_)
                    
                    


                del(sep_u)
                del(sep_s)
                del(sep_v)
                torch.cuda.empty_cache()

            else:
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v
                    )
                )
            
        else:
            new_vector[key] = (mer_vec[key] + tar_vec[key]) / 2

        new_vector[key] = new_vector[key].to(
            device=original_device, dtype=original_dtype, non_blocking=True
        )

    return new_vector, singular_values







def gs3r_merge_opt_flan_trunc( 
    task_vectors: List[StateDictType],
    exclude_keys: Optional[List[str]] = None,
    accelerator: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    procrustes: bool = True,
    idx: int = 0,
    singular_values = {},
    ntask: int = 8,
    iter_num: int = 400,
    opt_only_last: bool = True,
    init_recon: bool = False,
    k_s = 128, # -1 for full storage use
) -> StateDictType:

    print(f"Trunc + Optimization - rank:{k_s}")
    print("Computing SVD...")
    print(f"Procrustes : {procrustes}")
    assert(len(task_vectors) == 2) # For only two (inner algorithm)

    if exclude_keys is None:
        exclude_keys = []

    new_vector = {}
    mer_vec = task_vectors[0] 
    tar_vec = task_vectors[1] 
    indexer=True
    for key in mer_vec:
        original_device = mer_vec[key].device
        original_dtype = mer_vec[key].dtype

        # SVD
        if (
            len(mer_vec[key].shape) == 2
            and key not in exclude_keys
        ):
            
            if not (
                    original_dtype == torch.float32 or original_dtype == torch.float64
                ):
                    mer_vec[key] = mer_vec[key].to(dtype=torch.float32)
                    tar_vec[key] = tar_vec[key].to(dtype=torch.float32)


            if mer_vec[key].abs().sum() == 0:
                if tar_vec[key].abs().sum() == 0:
                    print(f'{key} : Not fine-tuned at all. Skip')
                    new_vector[key] = (mer_vec[key] + tar_vec[key]) / 2
                    new_vector[key] = new_vector[key].to(
                        device=original_device, dtype=original_dtype, non_blocking=True
                    )

                    continue
            
            
            if (key in singular_values.keys()):
                if len(singular_values[key]) == 3:
                    print(f'Restore Optimization Delta of {key}')
                    #opt_delta = singular_values[key][2]
                    #mer_vec[key] = mer_vec[key] - opt_delta
                    mer_vec[key] = singular_values[key][2]
                    
            mer_u, mer_s, mer_v = canon_svd(mer_vec[key].to(accelerator), full_matrices=False, k=k_s)
            tar_u, tar_s, tar_v = canon_svd(tar_vec[key].to(accelerator), full_matrices=False, k=k_s)

            #Singular Vector restoring
            if key in singular_values.keys():
                storage = singular_values[key]  
                R_u = storage[0]
                R_v = storage[1]

                R_u = restore_symmetric(R_u)
                R_v = restore_symmetric(R_v)

                if mer_u.shape[0] < mer_u.shape[1]:
                    mer_u = torch.linalg.multi_dot((R_u, mer_u)) 
                else:
                    mer_u = torch.linalg.multi_dot((mer_u, R_u))

                if mer_v.shape[0] < mer_v.shape[1]:
                    mer_v = torch.linalg.multi_dot((R_v, mer_v))
                else:
                    mer_v = torch.linalg.multi_dot((mer_v, R_v))



            sep_u, sep_s, sep_v, task_num = separate_tasks(mer_u, mer_s, mer_v)
            k_cur = mer_s.shape[0] if mer_s.shape[0] < k_s else k_s # pre-defined rank k or min shape for current weight
            k = max(1, k_cur // (task_num + 1))
            tvs = [torch.linalg.multi_dot((u_, torch.diag(s_), v_)) for u_,s_,v_ in zip(sep_u, sep_s, sep_v)] + [tar_vec[key].to(accelerator)]
            

            
            sum_u = torch.cat((*[u[:, :k] for u in sep_u], tar_u[:, :k]), dim=1)
            sum_s = torch.cat((*[s[:k]    for s in sep_s], tar_s[:k]),    dim=0)
            sum_v = torch.cat((*[v[:k, :] for v in sep_v], tar_v[:k, :]), dim=0)

            if sum_s.shape[0] != k_cur:
                margin = k_cur - sum_s.shape[0]
                
                sort_result = torch.topk(
                    torch.cat((*[s[k:] for s in sep_s], tar_s[k:]), dim=0), 
                    margin)
                
                cover_u = torch.cat((*[u[:, k:] for u in sep_u], tar_u[:, k:]), dim=1)[:, sort_result.indices] 
                cover_s = sort_result.values
                cover_v = torch.cat((*[v[k:, :] for v in sep_v], tar_v[k:, :]), dim=0)[sort_result.indices, :]

                sum_u = torch.cat((sum_u, cover_u), dim=1)
                sum_s = torch.cat((sum_s, cover_s), dim=0)
                sum_v = torch.cat((sum_v, cover_v), dim=0)

            last_sort = torch.topk(sum_s, sum_s.shape[0])
            sum_u = sum_u[:, last_sort.indices]
            sum_s = last_sort.values
            sum_v = sum_v[last_sort.indices, :]

            if procrustes:
                u_u, s_u, v_u = canon_svd(sum_u, full_matrices=False) 
                u_v, s_v, v_v = canon_svd(sum_v, full_matrices=False) 

                
                # Procrustes Solution and Rotation Vector Storing
                pro_u = torch.linalg.multi_dot((u_u, v_u))
                pro_v = torch.linalg.multi_dot((u_v, v_v))
                pro_u, pro_v = change_sign(pro_u, pro_v) 

                recon = torch.linalg.multi_dot((pro_u, torch.diag(sum_s), pro_v))
                
                if pro_u.shape[0] < pro_u.shape[1]:
                    R_u = choose_R(torch.eye(u_u.shape[1], device=u_u.device), u_u, pro_u, v_u)
                    RsR_u = torch.linalg.multi_dot((R_u, torch.diag(s_u), R_u.T))  
                else:
                    R_u = choose_R(pro_u, u_u, torch.eye(v_u.shape[1], device=v_u.device), v_u)
                    RsR_u = torch.linalg.multi_dot((R_u, torch.diag(s_u), R_u.T))

                if pro_v.shape[0] < pro_v.shape[1]:
                    R_v = choose_R(torch.eye(u_v.shape[1], device=u_v.device), u_v, pro_v, v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T))    
                else: 
                    R_v = choose_R(pro_v, u_v, torch.eye(v_v.shape[1], device=v_v.device), v_v)
                    RsR_v = torch.linalg.multi_dot((R_v, torch.diag(s_v), R_v.T))  
          

                RsR_u = save_symmetric(RsR_u)
                RsR_v = save_symmetric(RsR_v) 

                
                # Save Data
                new_vector[key] = recon
                singular_values[key] = [RsR_u, RsR_v]

                # Optmized version
                print(recon.shape)
                if (opt_only_last and (idx != (ntask-1))):
                    print('Pass Optimization')
                elif 'shared' in key or 'embed_tokens' in key or 'lm_head' in key:
                    print(f'{key} : Embedding layer or head, skip')
                else:
                    recon_ = copy.deepcopy(recon)
                    print(f'{key} - optimization')
                    if init_recon:
                        print(f'Init with Recon - {key}')
                        new_vector[key] = torch.nn.Parameter(recon)
                    else:
                        new_vector[key] = torch.nn.Parameter(torch.linalg.multi_dot((sum_u, torch.diag(sum_s), sum_v)))
                    new_vector[key] = wudi_merging_flan(new_vector[key], tvs, iter_num=iter_num)
                    del(tvs)
                    torch.cuda.empty_cache()


                    singular_values[key].append(recon_.to('cpu')) # Same as saving delta matrix
                    del(recon_)
                    


                del(sep_u)
                del(sep_s)
                del(sep_v)
                torch.cuda.empty_cache()

            else:
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v
                    )
                )
            
        else:
            new_vector[key] = (mer_vec[key] + tar_vec[key]) / 2

        new_vector[key] = new_vector[key].to(
            device=original_device, dtype=original_dtype, non_blocking=True
        )

    return new_vector, singular_values




































