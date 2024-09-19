import torch

import numpy as np

import dirichlet, niw
import utils


def update_Nk(r_nk):
    # Bishop eq 10.51
    return torch.sum(r_nk, dim=0)


def update_xk(x, r_nk, N_k):
    # Bishop eq 10.52
    x_k = torch.einsum("nk,nd->kd", r_nk, x)
    x_k_normed = x_k / N_k.unsqueeze(1)
    return torch.where(torch.isnan(x_k_normed), x_k, x_k_normed)


def update_Sk(x, r_nk, N_k, x_k):
    # Bishop eq 10.53
    x_xk = x.unsqueeze(1) - x_k.unsqueeze(0)
    S = torch.einsum("nk,nkde->kde", r_nk, torch.einsum("nkd,nke->nkde", x_xk, x_xk))
    S_normed = S / N_k.unsqueeze(1).unsqueeze(2)
    return torch.where(torch.isnan(S_normed), S, S_normed)


def update_alphak(alpha_0, N_k):
    # Bishop eq 10.58
    return torch.add(alpha_0, N_k)


def update_betak(beta_0, N_k):
    # Bishop eq 10.60
    return torch.add(beta_0, N_k)


def update_mk(beta_0, m_0, N_k, x_k, beta_k):
    # Bishop eq 10.61
    if len(beta_0.shape) == 1:
        beta_0 = torch.reshape(beta_0, (-1, 1))

    Nk_xk = N_k.unsqueeze(1) * x_k
    beta0_m0 = beta_0 * m_0
    return (beta0_m0 + Nk_xk) / beta_k.unsqueeze(1)


def update_Ck(C_0, x_k, N_k, m_0, beta_0, beta_k, S_k):
    # Bishop eq 10.62
    C = C_0 + N_k.unsqueeze(1).unsqueeze(2) * S_k
    Q0 = x_k - m_0
    q = torch.einsum("kd,ke->kde", Q0, Q0)
    return C + torch.einsum("k,kde->kde", (beta_0 * N_k) / beta_k, q)


def update_vk(v_0, N_k):
    # Bishop eq 10.63
    return (v_0 + N_k + 1).clone()


def m_step(x, r_nk, alpha_0, beta_0, m_0, C_0, v_0):
    N_k = update_Nk(r_nk)  # Bishop eq 10.51
    x_k = update_xk(x, r_nk, N_k)  # Bishop eq 10.52
    S_k = update_Sk(x, r_nk, N_k, x_k)  # Bishop eq 10.53

    alpha_k = update_alphak(alpha_0, N_k)  # Bishop eq 10.58
    beta_k = update_betak(beta_0, N_k)  # Bishop eq 10.60
    m_k = update_mk(beta_0, m_0, N_k, x_k, beta_k)  # Bishop eq 10.61
    C_k = update_Ck(C_0, x_k, N_k, m_0, beta_0, beta_k, S_k)  # Bishop eq 10.62
    v_k = update_vk(v_0, N_k)  # Bishop eq 10.63

    return alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k
