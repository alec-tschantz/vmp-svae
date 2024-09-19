import math
from math import pi

import torch
import numpy as np


eps = torch.tensor(1e-10)


def natural_to_standard(eta1: torch.Tensor, eta2: torch.Tensor):
    sigma = torch.inverse(-2.0 * eta2)
    mu = sigma @ eta1.unsqueeze(2)
    return torch.reshape(mu, eta1.shape), sigma


def standard_to_natural(mu: torch.Tensor, sigma: torch.Tensor):
    eta_2 = -0.5 * sigma.inverse()
    eta_1 = -2 * (eta_2 @ mu.unsqueeze(-1))
    return (torch.reshape(eta_1, mu.shape), eta_2)


def log_prob(y: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, weights: torch.Tensor):
    M, K, S, L = mean.shape
    y = y.unsqueeze(1).unsqueeze(1)
    sample_mean = torch.einsum("nksd,nk->", torch.pow(y - mean, 2) / var + torch.log(var + 1e-8), weights)
    return -0.5 * (sample_mean / S) - M * L / 2.0 * math.log(2.0 * pi)


def natural_log_prob(x: torch.Tensor, eta1: torch.Tensor, eta2: torch.Tensor, weights: torch.Tensor):
    N, D = x.shape

    logprob = torch.einsum("nd,nkd->nk", x, eta1)
    logprob += torch.einsum("nkd,nd->nk", torch.einsum("nd,nkde->nke", x, eta2), x)
    logprob -= D / 2.0 * torch.log(torch.tensor(2.0 * pi))

    eta1 = eta1.unsqueeze(3)
    logprob += 0.25 * torch.einsum("nkdi,nkdi->nk", eta2.inverse() @ eta1, eta1)

    logprob += -0.5 * logdet(-2.0 * eta2 + eps + torch.eye(D))
    logprob += weights.unsqueeze(0)

    max_logprob = torch.max(logprob, dim=1, keepdim=True)[0]
    normalizer = max_logprob + torch.log(torch.sum(torch.exp(logprob - max_logprob), dim=1, keepdim=True))
    return logprob - normalizer


def natural_log_prob_per_sample(x_samples: torch.Tensor, eta1: torch.Tensor, eta2: torch.Tensor):
    N, K, S, D = x_samples.shape

    log_normal = torch.einsum("nksd,nksd->nks", torch.einsum("nkij,nksj->nksi", eta2, x_samples), x_samples)
    log_normal += torch.einsum("nki,nksi->nks", eta1, x_samples)

    log_normal += torch.tensor(1.0 / 4) * torch.einsum("nkdi,nkd->nki", eta2.inverse() @ eta1.unsqueeze(-1), eta1)
    log_pi = torch.tensor(np.log(2 * pi))
    log_normal -= torch.tensor(D / 2.0) * log_pi

    log_sigma = logdet(-2.0 * eta2 + 1e-20 * torch.eye(D))
    log_normal += torch.tensor(1.0 / 2) * log_sigma.unsqueeze(2)
    return log_normal


def logdet(a: torch.Tensor):
    return 2.0 * torch.sum(torch.log(torch.diagonal(torch.linalg.cholesky(a), dim1=-2, dim2=-1)), dim=-1)
