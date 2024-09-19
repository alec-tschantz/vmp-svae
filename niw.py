from math import pi

import numpy as np

import torch


def natural_to_standard(m_o, phi_o, beta, v_hat):
    m = phi_o / beta.unsqueeze(-1)

    K, D = m.shape
    D = int(D)

    C = m_o - _outer(phi_o, m)
    v = v_hat - D - 2
    return beta, m, C, v


def standard_to_natural(beta, m, C, v):
    K, D = m.shape
    D = int(D)

    b = beta.unsqueeze(-1) * m
    A = C + _outer(b, m)
    v_hat = v + D + 2

    return A, b, beta, v_hat


def _outer(a, b):
    a_ = a.unsqueeze(-1)
    b_ = b.unsqueeze(-2)
    return a_ * b_


def expected_values(niw_standard_params):
    beta, m, C, v = niw_standard_params
    exp_m = m.clone()
    C_inv = C.inverse()
    C_inv_sym = C_inv + torch.transpose(C_inv, dim0=2, dim1=1) / 2.0
    expected_precision = C_inv_sym * v.unsqueeze(1).unsqueeze(2)
    expected_covariance = expected_precision.inverse()

    return exp_m, expected_covariance
